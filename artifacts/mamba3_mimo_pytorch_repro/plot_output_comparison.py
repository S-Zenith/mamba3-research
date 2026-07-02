from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
REPRO_DIR = ROOT / "artifacts" / "mamba3_mimo_pytorch_repro"
INTERMEDIATES = REPRO_DIR / "intermediates.pt"
PLOT_DIR = REPRO_DIR / "plots"
NEAR_ZERO_THRESHOLD = 0.05


def save_reference_sorted_actual_regression(reference: torch.Tensor, actual: torch.Tensor, title: str, path: Path) -> None:
    ref = reference.detach().float().cpu().flatten()
    act = actual.detach().float().cpu().flatten()
    order = torch.argsort(ref)
    ref_sorted = ref[order].numpy()
    act_sorted = act[order].numpy()
    lo = min(float(ref.min().item()), float(act.min().item()))
    hi = max(float(ref.max().item()), float(act.max().item()))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_sorted, act_sorted, s=6, alpha=0.35, label="PyTorch actual vs sorted official reference")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="y = x")
    ax.set_title(title)
    ax.set_xlabel("official/reference value (sorted ascending)")
    ax.set_ylabel("PyTorch/actual value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reference_sorted_actual_regression_near_zero(
    reference: torch.Tensor,
    actual: torch.Tensor,
    title: str,
    path: Path,
    threshold: float = NEAR_ZERO_THRESHOLD,
) -> None:
    ref = reference.detach().float().cpu().flatten()
    act = actual.detach().float().cpu().flatten()
    mask = ref.abs() <= threshold
    ref = ref[mask]
    act = act[mask]
    if ref.numel() == 0:
        raise ValueError(f"no points found for near-zero threshold {threshold}")

    order = torch.argsort(ref)
    ref_sorted = ref[order].numpy()
    act_sorted = act[order].numpy()
    lo = min(float(ref.min().item()), float(act.min().item()), -threshold)
    hi = max(float(ref.max().item()), float(act.max().item()), threshold)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_sorted, act_sorted, s=8, alpha=0.45, label=f"|official/reference| <= {threshold}")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="y = x")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title}\nnear zero: |official/reference| <= {threshold}, points={ref.numel()}")
    ax.set_xlabel("official/reference value near zero (sorted ascending)")
    ax.set_ylabel("PyTorch/actual value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots() -> list[Path]:
    if not INTERMEDIATES.exists():
        raise FileNotFoundError(f"missing {INTERMEDIATES}; run mamba3_mimo_pytorch_repro.py first")

    data = torch.load(INTERMEDIATES, map_location="cpu")
    if "official_out" not in data:
        raise KeyError("intermediates.pt does not contain official_out; rerun after official MIMO kernel comparison succeeds")
    if "torch_out" not in data:
        raise KeyError("intermediates.pt does not contain torch_out")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output = PLOT_DIR / "final_output_reference_sorted_actual_regression.png"
    near_zero_output = PLOT_DIR / "final_output_reference_sorted_actual_regression_near_zero.png"
    save_reference_sorted_actual_regression(
        data["official_out"],
        data["torch_out"],
        "Mamba3 MIMO final output: PyTorch actual vs sorted official reference",
        output,
    )
    save_reference_sorted_actual_regression_near_zero(
        data["official_out"],
        data["torch_out"],
        "Mamba3 MIMO final output: PyTorch actual vs sorted official reference",
        near_zero_output,
    )
    (PLOT_DIR / "README.md").write_text(
        "# Mamba3 MIMO Output Comparison Plots\n\n"
        f"- `{output.name}`: x-axis is sorted official/reference output, y-axis is PyTorch/actual output, dashed line is `y = x`.\n"
        f"- `{near_zero_output.name}`: same scatter plot zoomed to `|official/reference| <= {NEAR_ZERO_THRESHOLD}`.\n",
        encoding="utf-8",
    )
    return [output, near_zero_output]


if __name__ == "__main__":
    paths = make_plots()
    for path in paths:
        print(path)
