import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_mimo_pytorch_repro/mamba3_mimo_pytorch_repro.py")


def load_module():
    spec = importlib.util.spec_from_file_location("mamba3_mimo_pytorch_repro", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_tensors_reports_zero_for_identical_tensors():
    module = load_module()
    x = torch.tensor([1.0, 2.0, 3.0])
    row = module.compare_tensors("x", x, x.clone())
    assert row["max_abs"] == 0.0
    assert row["mean_abs"] == 0.0
    assert row["max_rel"] == 0.0


def test_apply_rope_mimo_rotates_first_and_middle_halves():
    module = load_module()
    x = torch.zeros(1, 1, 1, 1, 8)
    x[..., 0] = 1.0
    x[..., 4] = 2.0
    angles = torch.tensor([[[[torch.pi / 2, 0.0]]]])

    out = module.apply_rope_mimo(x, angles, rotary_dim_divisor=4)

    expected = x.clone()
    expected[..., 0] = -2.0
    expected[..., 4] = 1.0
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_compute_dacs_segsum_torch_matches_reference_shape_and_values():
    module = load_module()
    da = torch.tensor([[[-1.0, -2.0, -3.0, -4.0]]])

    da_cs, da_cs_rev, segsum = module.compute_dacs_segsum_torch(da, chunk_size=4)

    assert torch.equal(da_cs, torch.tensor([[[-1.0, -3.0, -6.0, -10.0]]]))
    assert torch.equal(da_cs_rev, torch.tensor([[[-9.0, -7.0, -4.0, 0.0]]]))
    assert segsum.shape == (1, 1, 1, 4, 4)
    assert segsum[0, 0, 0, 2, 0].item() == -5.0
    assert segsum[0, 0, 0, 2, 1].item() == -3.0
    assert segsum[0, 0, 0, 0, 0].item() == 0.0


def test_prepare_mimo_inputs_has_expected_shapes_on_cpu():
    module = load_module()
    device = torch.device("cpu")
    model = module.build_demo_mimo_model(device)
    u = module.make_input(device)

    prepared, intermediates = module.prepare_mimo_inputs(model, u)

    assert prepared["Q"].shape == (module.BATCH, module.SEQ_LEN, module.MIMO_RANK, model.num_bc_heads, module.D_STATE)
    assert prepared["K"].shape == (module.BATCH, module.SEQ_LEN, module.MIMO_RANK, model.num_bc_heads, module.D_STATE)
    assert prepared["V"].shape == (module.BATCH, module.SEQ_LEN, model.nheads, module.HEADDIM)
    assert prepared["Z"].shape == (module.BATCH, module.SEQ_LEN, model.nheads, module.HEADDIM)
    assert prepared["ADT"].shape == (module.BATCH, model.nheads, module.SEQ_LEN)
    assert intermediates["in_proj_out"].shape[0] == module.BATCH


def test_mimo_kernel_torch_returns_expected_shape_and_debug_tensors_on_cpu():
    module = load_module()
    device = torch.device("cpu")
    model = module.build_demo_mimo_model(device)
    u = module.make_input(device)
    prepared, _ = module.prepare_mimo_inputs(model, u)

    y, debug = module.mimo_kernel_torch(model, prepared)

    assert y.shape == (module.BATCH, module.SEQ_LEN, model.nheads, module.HEADDIM)
    assert debug["DA_CS"].shape == (module.BATCH, model.nheads, module.SEQ_LEN)
    assert debug["Segsum"].shape[:3] == (module.BATCH, model.nheads, module.SEQ_LEN // module.CHUNK_SIZE)
    assert debug["chunks"][0]["psi_v"].shape == (module.CHUNK_SIZE, module.MIMO_RANK, module.HEADDIM)


def test_run_pytorch_demo_only_writes_report_without_official_reference(tmp_path):
    module = load_module()

    result = module.run_pytorch_demo_only(tmp_path, torch.device("cpu"), reason="missing tilelang")

    report = tmp_path / "comparison_report.md"
    intermediates = tmp_path / "intermediates.pt"
    assert report.exists()
    assert intermediates.exists()
    assert result["official_available"] is False
    assert "official MIMO reference unavailable" in report.read_text(encoding="utf-8")
