import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/op_quant_kernel.py")


def load_module():
    spec = importlib.util.spec_from_file_location("op_quant_kernel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fake_quant_tracker_counts_quantized_ops():
    module = load_module()
    tracker = module.FakeQuantTracker(lambda x: x)
    x = torch.tensor([1.0])

    assert tracker(x).item() == 1.0
    assert tracker(x).item() == 1.0
    assert tracker.count == 2


def test_state_quantizer_can_differ_from_regular_quantizer():
    module = load_module()
    tracker = module.FakeQuantTracker(lambda x: x + 1)
    state_tracker = module.FakeQuantTracker(lambda x: x.to(torch.bfloat16).to(torch.float32))

    assert tracker(torch.tensor([1.0])).item() == 2.0
    assert state_tracker(torch.tensor([1.0])).dtype == torch.float32
