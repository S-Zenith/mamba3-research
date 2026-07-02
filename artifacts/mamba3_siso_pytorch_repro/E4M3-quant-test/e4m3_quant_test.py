from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[3]
REPRO_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
OUT_DIR = REPRO_DIR / "E4M3-quant-test"
PLOT_DIR = OUT_DIR / "plots"
REPRO_SCRIPT = REPRO_DIR / "mamba3_siso_pytorch_repro.py"
PLOT_SCRIPT = REPRO_DIR / "plot_error_surfaces.py"
AE4M3_SCRIPT = REPRO_DIR / "AE4M3-WINT8-quant-test" / "ae4m3_wint8_quant_test.py"
OP_QUANT_SCRIPT = REPRO_DIR / "op_quant_kernel.py"
FLOW_SCRIPT = REPRO_DIR / "pytorch_siso_flow.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repro = load_module(REPRO_SCRIPT, "mamba3_siso_pytorch_repro")
plots = load_module(PLOT_SCRIPT, "plot_error_surfaces")
ae4m3 = load_module(AE4M3_SCRIPT, "ae4m3_wint8_quant_test")
opq = load_module(OP_QUANT_SCRIPT, "op_quant_kernel")
flow = load_module(FLOW_SCRIPT, "pytorch_siso_flow")


def quantize_e4m3(x: torch.Tensor) -> dict[str, torch.Tensor | str]:
    return ae4m3.quantize_e4m3(x)


def dequantize_e4m3(record: dict[str, torch.Tensor | str]) -> torch.Tensor:
    return ae4m3.dequantize_e4m3(record)


def qdq_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
    record = quantize_e4m3(x)
    return dequantize_e4m3(record).to(x.device), record


def quantize_activation_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return quantize_e4m3(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: quantize_activation_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [quantize_activation_tree(v) for v in value]
    return value


def quantize_e4m3_state_dict(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, tensor in state.items():
        out[name] = quantize_e4m3(tensor) if tensor.is_floating_point() else tensor.detach().cpu().contiguous()
    return out


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    row = repro.compare_tensors(name, actual, expected)
    actual_f = actual.detach().to(torch.float64).cpu()
    expected_f = expected.detach().to(torch.float64).cpu()
    row["rel_l2"] = float(torch.linalg.vector_norm(actual_f - expected_f) / torch.linalg.vector_norm(expected_f).clamp_min(1e-12))
    return row


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# E4M3-quant-test", ""]
    lines.extend(notes)
    lines.append("")
    lines.append("| tensor | shape | max_abs | mean_abs | max_rel | rel_l2 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['shape']}` | "
            f"{row['max_abs']:.8e} | {row['mean_abs']:.8e} | {row['max_rel']:.8e} | {row['rel_l2']:.8e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_e4m3_model_from_fp32(device: torch.device):
    fp32_model = flow.load_siso_weights(device)
    quantized_state = quantize_e4m3_state_dict(flow.state_dict_from_weights(fp32_model))
    qdq_state = {}
    for name, tensor in flow.state_dict_from_weights(fp32_model).items():
        if tensor.is_floating_point():
            qdq_state[name] = dequantize_e4m3(quantized_state[name]).to(device=device, dtype=tensor.dtype)
        else:
            qdq_state[name] = tensor
    e4m3_model = flow.load_siso_weights(device, qdq_state)
    return fp32_model, e4m3_model, quantized_state


def run_e4m3_quant_test() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model, e4m3_model, weights_e4m3 = build_e4m3_model_from_fp32(device)
    u = flow.make_input(device)

    baseline_prepared, baseline_intermediates = flow.prepare_siso_inputs(fp32_model, u)
    baseline_y, baseline_kernel_debug = repro.siso_kernel_torch(fp32_model, baseline_prepared)
    baseline_out = flow.out_proj(fp32_model, rearrange(baseline_y, "b l h p -> b l (h p)").to(torch.float32))

    u_e4m3_deq, u_e4m3 = qdq_e4m3(u)
    e4m3_prepared, e4m3_intermediates = flow.prepare_siso_inputs(e4m3_model, u_e4m3_deq)

    e4m3_prepared_deq: dict[str, torch.Tensor] = {}
    e4m3_prepared_quant: dict[str, Any] = {}
    for name, tensor in e4m3_prepared.items():
        deq, record = qdq_e4m3(tensor)
        e4m3_prepared_deq[name] = deq.to(tensor.device)
        e4m3_prepared_quant[name] = record

    e4m3_y, e4m3_kernel_debug = opq.siso_kernel_op_quant(e4m3_model, e4m3_prepared_deq, lambda x: qdq_e4m3(x)[0])
    e4m3_y_deq, e4m3_y_quant = qdq_e4m3(e4m3_y)
    e4m3_out = flow.out_proj(e4m3_model, rearrange(e4m3_y_deq, "b l h p -> b l (h p)").to(torch.float32))
    e4m3_out_deq, e4m3_out_quant = qdq_e4m3(e4m3_out)

    rows = [
        compare_tensors("final_output", e4m3_out_deq, baseline_out),
        compare_tensors("kernel_y", e4m3_y_deq, baseline_y),
        compare_tensors("out_v", e4m3_kernel_debug["out_v"], baseline_kernel_debug["out_v"]),
        compare_tensors("Q_rot", e4m3_kernel_debug["Q_rot"], baseline_kernel_debug["Q_rot"]),
        compare_tensors("K_scaled", e4m3_kernel_debug["K_scaled"], baseline_kernel_debug["K_scaled"]),
    ]

    activation_format = quantize_e4m3(torch.tensor([1.0]))["format"]
    payload = {
        "experiment": "E4M3-quant-test",
        "baseline_dtype": str(baseline_out.dtype),
        "e4m3_format": activation_format,
        "device": str(device),
        "u_e4m3": u_e4m3,
        "weights_e4m3": weights_e4m3,
        "prepared_e4m3": e4m3_prepared_quant,
        "baseline_out": repro.detach_cpu(baseline_out),
        "e4m3_out_dequant": repro.detach_cpu(e4m3_out_deq),
        "baseline_y": repro.detach_cpu(baseline_y),
        "e4m3_y_dequant": repro.detach_cpu(e4m3_y_deq),
        "e4m3_y_quant": e4m3_y_quant,
        "e4m3_out_quant": e4m3_out_quant,
        "baseline_intermediates": repro.to_cpu_tree(baseline_intermediates),
        "e4m3_intermediates_quant": quantize_activation_tree(e4m3_intermediates),
        "baseline_kernel_debug": repro.to_cpu_tree(baseline_kernel_debug),
        "e4m3_kernel_debug_quant": quantize_activation_tree(e4m3_kernel_debug),
        "comparison": rows,
    }
    torch.save(payload, OUT_DIR / "quantized_intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        rows,
        [
            "- experiment: `E4M3-quant-test`",
            f"- baseline: PyTorch SISO FP32 output, dtype `{baseline_out.dtype}`",
            f"- weights/activations/intermediates/states: `{activation_format}`",
            "- arithmetic model: op-level fake quant; each major kernel op output is quantized back to E4M3",
        ],
    )

    old_plot_dir = plots.PLOT_DIR
    plots.PLOT_DIR = PLOT_DIR
    try:
        plot_outputs = plots.save_matrix_plot_set("final_output", e4m3_out_deq[0], baseline_out[0])
    finally:
        plots.PLOT_DIR = old_plot_dir

    readme = ["# E4M3-quant-test Plots", "", "Reference is PyTorch FP32 baseline; actual is full E4M3 dequantized output.", ""]
    readme.extend(f"- `{path.name}`" for path in plot_outputs)
    (PLOT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_e4m3_quant_test()
    print(OUT_DIR / "comparison_report.md")
    print(OUT_DIR / "quantized_intermediates.pt")
