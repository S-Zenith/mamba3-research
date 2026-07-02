import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/INT8-StateBF16-quant-test/int8_statebf16_quant_test.py")


def load_module():
    spec = importlib.util.spec_from_file_location("int8_statebf16_quant_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_state_bf16_roundtrip_returns_fp32_and_records_bf16():
    module = load_module()
    x = torch.tensor([1.0, 1.125, -2.5])
    record = module.quantize_state_bf16(x)
    deq = module.dequantize_state_bf16(record)

    assert record["q"].dtype == torch.bfloat16
    assert deq.dtype == torch.float32
    assert deq.shape == x.shape
    assert torch.allclose(deq, x, atol=0.01, rtol=0)


def test_is_state_name_detects_recurrent_state_names():
    module = load_module()

    assert module.is_state_name("states")
    assert module.is_state_name("states_before")
    assert module.is_state_name("states_after")
    assert not module.is_state_name("Q_rot")


def test_statebf16_kernel_uses_shared_op_level_quant_kernel():
    module = load_module()
    source = inspect.getsource(module.siso_kernel_int8_statebf16)

    assert "siso_kernel_op_quant" in source
    assert "state_quant" in source
