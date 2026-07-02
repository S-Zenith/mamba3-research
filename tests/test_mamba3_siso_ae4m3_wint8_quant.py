import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/AE4M3-WINT8-quant-test/ae4m3_wint8_quant_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("ae4m3_wint8_quant_test", SCRIPT)
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


def test_quantize_activation_tree_quantizes_floating_tensors_only():
    module = load_module()
    value = {"x": torch.tensor([1.0]), "i": torch.tensor([1], dtype=torch.int64)}
    q = module.quantize_activation_tree(value)

    assert "dequant" in q["x"]
    assert q["x"]["dequant"].dtype == torch.float32
    assert torch.equal(q["i"], value["i"])
