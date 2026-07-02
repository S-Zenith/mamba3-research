import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/INT8-quant-test/int8_quant_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("int8_quant_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_symmetric_int8_quantize_dequantize_round_trips_simple_tensor():
    module = load_module()
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    q = module.quantize_symmetric_int8(x)

    assert q["q"].dtype == torch.int8
    assert torch.isclose(q["scale"], torch.tensor(1.0 / 127.0))
    deq = module.dequantize_symmetric_int8(q)
    assert torch.allclose(deq, x, atol=q["scale"].item(), rtol=0)


def test_quantize_zero_tensor_uses_unit_scale_and_stays_zero():
    module = load_module()
    x = torch.zeros(4)
    q = module.quantize_symmetric_int8(x)

    assert q["scale"].item() == 1.0
    assert torch.equal(q["q"], torch.zeros(4, dtype=torch.int8))
    assert torch.equal(module.dequantize_symmetric_int8(q), x)
