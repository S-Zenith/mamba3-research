from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib import cm


ROOT = Path(__file__).resolve().parents[2]
REPRO_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
INTERMEDIATES = REPRO_DIR / "intermediates.pt"
PLOT_DIR = REPRO_DIR / "plots"
NEAR_ZERO_THRESHOLD = 0.05


def build_error_matrices(actual: torch.Tensor, reference: torch.Tensor, rel_threshold: float = 1e-3) -> dict[str, torch.Tensor]:
    actual_f = actual.detach().float().cpu()
    reference_f = reference.detach().float().cpu()
    abs_error = (actual_f - reference_f).abs()
    # 相对误差分母加阈值，避免 reference 接近 0 时图像被单个点拉爆。
    relative_error = abs_error / reference_f.abs().clamp_min(rel_threshold)
    return {
        "actual": actual_f,
        "reference": reference_f,
        "abs_error": abs_error,
        "relative_error": relative_error,
    }


def make_xy_grid(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_size, x_size = matrix.shape
    xs = torch.arange(x_size)
    ys = torch.arange(y_size)
    return torch.meshgrid(xs, ys, indexing="xy")


def save_surface_reference_with_abs_error(reference: torch.Tensor, abs_error: torch.Tensor, title: str, path: Path) -> None:
    ref_z = reference.numpy()
    err_z = abs_error.numpy()
    x_grid, y_grid = make_xy_grid(reference)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    ref_surface = ax.plot_surface(
        x_grid.numpy(),
        y_grid.numpy(),
        ref_z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.72,
    )
    err_surface = ax.plot_surface(
        x_grid.numpy(),
        y_grid.numpy(),
        err_z,
        cmap=cm.autumn,
        linewidth=0,
        antialiased=True,
        alpha=0.62,
    )
    ax.set_title(
        f"{title}\n"
        f"reference range=[{reference.min().item():.3e}, {reference.max().item():.3e}], "
        f"abs_error range=[{abs_error.min().item():.3e}, {abs_error.max().item():.3e}]"
    )
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel("reference / abs error")
    fig.colorbar(ref_surface, shrink=0.55, aspect=12, pad=0.02, label="reference")
    fig.colorbar(err_surface, shrink=0.55, aspect=12, pad=0.12, label="abs error")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_surface(matrix: torch.Tensor, title: str, z_label: str, path: Path) -> None:
    z = matrix.numpy()
    y_size, x_size = z.shape
    xs = torch.arange(x_size).numpy()
    ys = torch.arange(y_size).numpy()
    x_grid, y_grid = torch.meshgrid(torch.from_numpy(xs), torch.from_numpy(ys), indexing="xy")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x_grid.numpy(), y_grid.numpy(), z, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_zlabel(z_label)
    fig.colorbar(surface, shrink=0.6, aspect=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(matrix: torch.Tensor, title: str, color_label: str, path: Path, cmap: str = "magma") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.colorbar(image, ax=ax, label=color_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_value_error_scatter(reference: torch.Tensor, abs_error: torch.Tensor, title: str, path: Path) -> None:
    ref_abs = reference.abs().flatten().numpy()
    err = abs_error.flatten().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ref_abs, err, s=6, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("abs(reference value)")
    ax.set_ylabel("abs(error)")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_reference_sorted_actual_regression(reference: torch.Tensor, actual: torch.Tensor, title: str, path: Path) -> None:
    ref = reference.flatten()
    act = actual.flatten()
    order = torch.argsort(ref)
    ref_sorted = ref[order].numpy()
    act_sorted = act[order].numpy()
    lo = min(float(ref.min().item()), float(act.min().item()))
    hi = max(float(ref.max().item()), float(act.max().item()))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ref_sorted, act_sorted, s=6, alpha=0.35, label="actual vs sorted reference")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="y = x")
    ax.set_title(title)
    ax.set_xlabel("reference value (sorted ascending)")
    ax.set_ylabel("actual/PyTorch value")
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
    ref = reference.flatten()
    act = actual.flatten()
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
    ax.scatter(ref_sorted, act_sorted, s=8, alpha=0.45, label=f"|reference| <= {threshold}")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, linestyle="--", label="y = x")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title}\nnear zero: |reference| <= {threshold}, points={ref.numel()}")
    ax.set_xlabel("reference value near zero (sorted ascending)")
    ax.set_ylabel("actual/PyTorch value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def expected_plot_paths(name: str) -> list[Path]:
    prefix = PLOT_DIR / name
    return [
        prefix.with_name(f"{name}_surface_reference_with_abs_error.png"),
        prefix.with_name(f"{name}_abs_error_heatmap.png"),
        prefix.with_name(f"{name}_relative_error_heatmap.png"),
        prefix.with_name(f"{name}_value_vs_error_scatter.png"),
        prefix.with_name(f"{name}_reference_sorted_actual_regression.png"),
        prefix.with_name(f"{name}_reference_sorted_actual_regression_near_zero.png"),
    ]


def remove_obsolete_surface_files(name: str) -> None:
    for suffix in ["surface_reference", "surface_torch", "surface_abs_error"]:
        path = PLOT_DIR / f"{name}_{suffix}.png"
        if path.exists():
            path.unlink()


def save_matrix_plot_set(name: str, actual: torch.Tensor, reference: torch.Tensor, rel_threshold: float = 1e-3) -> list[Path]:
    matrices = build_error_matrices(actual, reference, rel_threshold=rel_threshold)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    remove_obsolete_surface_files(name)
    outputs = expected_plot_paths(name)
    save_surface_reference_with_abs_error(
        matrices["reference"],
        matrices["abs_error"],
        f"{name}: reference surface with absolute error overlay",
        outputs[0],
    )
    save_heatmap(matrices["abs_error"], f"{name}: absolute error heatmap", "abs error", outputs[1])
    save_heatmap(matrices["relative_error"], f"{name}: relative error heatmap, threshold={rel_threshold}", "relative error", outputs[2])
    save_value_error_scatter(matrices["reference"], matrices["abs_error"], f"{name}: value magnitude vs abs error", outputs[3])
    save_reference_sorted_actual_regression(
        matrices["reference"],
        matrices["actual"],
        f"{name}: actual vs sorted reference regression",
        outputs[4],
    )
    save_reference_sorted_actual_regression_near_zero(
        matrices["reference"],
        matrices["actual"],
        f"{name}: actual vs sorted reference regression",
        outputs[5],
    )
    return outputs


def make_plots() -> list[Path]:
    if not INTERMEDIATES.exists():
        raise FileNotFoundError(f"missing {INTERMEDIATES}; run mamba3_siso_pytorch_repro.py first")

    data = torch.load(INTERMEDIATES, map_location="cpu")
    outputs: list[Path] = []

    # final_output 本身是 [seq_len, d_model] 二维矩阵，最适合作为整体误差图。
    outputs.extend(save_matrix_plot_set("final_output", data["torch_out"][0], data["official_out"][0]))

    # kernel 内部张量选第 0 个 batch、第 0 个 head，观察 RoPE 与 state 输出误差。
    outputs.extend(save_matrix_plot_set("q_rot_b0_h0", data["kernel_debug"]["Q_rot"][0, :, 0, :], data["official_kernel"]["Q_rot"][0, :, 0, :]))
    outputs.extend(save_matrix_plot_set("k_scaled_b0_h0", data["kernel_debug"]["K_scaled"][0, :, 0, :], data["official_kernel"]["K_scaled"][0, :, 0, :]))
    outputs.extend(save_matrix_plot_set("out_v_b0_h0", data["kernel_debug"]["out_v"][0, :, 0, :], data["official_kernel"]["out_v"][0, :, 0, :]))

    index_lines = ["# Mamba3 SISO Error Plots", ""]
    for output in outputs:
        index_lines.append(f"- `{output.name}`")
    (PLOT_DIR / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return outputs


if __name__ == "__main__":
    paths = make_plots()
    for path in paths:
        print(path)
