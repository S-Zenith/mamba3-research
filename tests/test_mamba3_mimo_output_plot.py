import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_mimo_pytorch_repro/plot_output_comparison.py")


def load_module():
    spec = importlib.util.spec_from_file_location("mamba3_mimo_output_plot", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_save_reference_sorted_actual_regression_writes_png(tmp_path):
    module = load_module()
    reference = torch.tensor([[0.3, -0.2], [0.1, 0.0]])
    actual = torch.tensor([[0.31, -0.18], [0.09, 0.01]])
    path = tmp_path / "sorted_scatter.png"

    module.save_reference_sorted_actual_regression(reference, actual, "demo", path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_save_reference_sorted_actual_regression_near_zero_writes_png(tmp_path):
    module = load_module()
    reference = torch.tensor([[0.3, -0.02], [0.01, 0.0]])
    actual = torch.tensor([[0.31, -0.018], [0.012, 0.003]])
    path = tmp_path / "near_zero.png"

    module.save_reference_sorted_actual_regression_near_zero(reference, actual, "near zero", path, threshold=0.05)

    assert path.exists()
    assert path.stat().st_size > 0
