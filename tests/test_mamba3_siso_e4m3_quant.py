import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/E4M3-quant-test/e4m3_quant_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("e4m3_quant_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_e4m3_quantize_dequantize_returns_fp32_with_same_shape():
    module = load_module()
    x = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0])
    q = module.quantize_e4m3(x)
    deq = module.dequantize_e4m3(q)

    assert deq.dtype == torch.float32
    assert deq.shape == x.shape
    assert torch.all(torch.isfinite(deq))
    assert torch.allclose(deq, x, atol=0.08, rtol=0)


def test_quantize_e4m3_state_dict_returns_records_for_floating_weights():
    module = load_module()
    state = {"w": torch.tensor([1.0, -1.0]), "idx": torch.tensor([1], dtype=torch.int64)}
    quantized = module.quantize_e4m3_state_dict(state)

    assert "dequant" in quantized["w"]
    assert quantized["w"]["dequant"].dtype == torch.float32
    assert torch.equal(quantized["idx"], state["idx"])
