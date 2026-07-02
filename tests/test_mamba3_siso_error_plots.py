import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/plot_error_surfaces.py")


def load_module():
    spec = importlib.util.spec_from_file_location("plot_error_surfaces", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_error_matrices_returns_abs_and_thresholded_relative_error():
    module = load_module()
    actual = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    reference = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    matrices = module.build_error_matrices(actual, reference, rel_threshold=1.0)

    assert torch.equal(matrices["actual"], actual.float())
    assert torch.equal(matrices["reference"], reference.float())
    assert torch.equal(matrices["abs_error"], torch.tensor([[0.0, 1.0], [2.0, 8.0]]))
    assert torch.equal(matrices["relative_error"], torch.tensor([[0.0, 1.0], [1.0, 8.0]]))


def test_expected_plot_paths_exclude_standalone_surfaces_and_include_regression_plot():
    module = load_module()
    names = [path.name for path in module.expected_plot_paths("demo")]

    assert "demo_surface_reference_with_abs_error.png" in names
    assert "demo_reference_sorted_actual_regression.png" in names
    assert "demo_reference_sorted_actual_regression_near_zero.png" in names
    assert "demo_surface_reference.png" not in names
    assert "demo_surface_torch.png" not in names
    assert "demo_surface_abs_error.png" not in names
