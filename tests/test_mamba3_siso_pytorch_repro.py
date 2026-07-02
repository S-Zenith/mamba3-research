import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py")


def load_module():
    spec = importlib.util.spec_from_file_location("mamba3_siso_pytorch_repro", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_rope_pairwise_preserves_shape_and_rotates_pairs():
    module = load_module()
    x = torch.tensor([[[[1.0, 0.0, 2.0, 0.0]]]])
    angles = torch.tensor([[[[0.0, torch.pi / 2]]]])
    out = module.apply_rope_pairwise(x, angles)
    expected = torch.tensor([[[[1.0, 0.0, 0.0, 2.0]]]])
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_compare_tensors_reports_zero_for_identical_tensors():
    module = load_module()
    x = torch.tensor([1.0, 2.0, 3.0])
    row = module.compare_tensors("x", x, x.clone())
    assert row["max_abs"] == 0.0
    assert row["mean_abs"] == 0.0
    assert row["max_rel"] == 0.0


def test_angle_dt_cumsum_applies_tanh_dt_cumsum_and_modulo():
    module = load_module()
    angles = torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]])
    dt = torch.tensor([[[0.5, 0.25, 0.25]]])
    out = module.angle_dt_cumsum(angles, dt)
    expected_step = torch.tanh(angles) * torch.pi * dt.permute(0, 2, 1).unsqueeze(-1)
    expected = torch.cumsum(expected_step, dim=1) % (2 * torch.pi)
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)
