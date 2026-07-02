from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[3]
REPRO_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
OUT_DIR = REPRO_DIR / "INT8-quant-test"
PLOT_DIR = OUT_DIR / "plots"
REPRO_SCRIPT = REPRO_DIR / "mamba3_siso_pytorch_repro.py"
PLOT_SCRIPT = REPRO_DIR / "plot_error_surfaces.py"
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
opq = load_module(OP_QUANT_SCRIPT, "op_quant_kernel")
flow = load_module(FLOW_SCRIPT, "pytorch_siso_flow")


def quantize_symmetric_int8(x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_f = x.detach().to(torch.float32)
    max_abs = x_f.abs().max()
    scale = torch.tensor(1.0, dtype=torch.float32, device=x_f.device) if max_abs == 0 else max_abs / 127.0
    q = torch.round(x_f / scale).clamp(-127, 127).to(torch.int8)
    return {"q": q, "scale": scale.detach().to(torch.float32), "shape": torch.tensor(x.shape, dtype=torch.int64)}


def dequantize_symmetric_int8(record: dict[str, torch.Tensor]) -> torch.Tensor:
    return record["q"].to(torch.float32) * record["scale"].to(torch.float32)


def qdq_tensor(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    record = quantize_symmetric_int8(x)
    return dequantize_symmetric_int8(record).to(x.device), record


def quantize_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        record = quantize_symmetric_int8(value.cpu())
        record["dequant"] = dequantize_symmetric_int8(record)
        return record
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: quantize_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [quantize_tree(v) for v in value]
    return value


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    row = repro.compare_tensors(name, actual, expected)
    actual_f = actual.detach().to(torch.float64).cpu()
    expected_f = expected.detach().to(torch.float64).cpu()
    row["rel_l2"] = float(torch.linalg.vector_norm(actual_f - expected_f) / torch.linalg.vector_norm(expected_f).clamp_min(1e-12))
    return row


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# INT8-quant-test", ""]
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


def build_int8_model_from_fp32(device: torch.device):
    fp32_model = flow.load_siso_weights(device)
    quantized_state: dict[str, dict[str, torch.Tensor]] = {}
    qdq_state = {}
    for name, tensor in flow.state_dict_from_weights(fp32_model).items():
        if tensor.is_floating_point():
            deq, record = qdq_tensor(tensor)
            qdq_state[name] = deq.to(tensor.dtype)
            quantized_state[name] = {k: v.detach().cpu() for k, v in record.items()}
        else:
            qdq_state[name] = tensor
    int8_model = flow.load_siso_weights(device, qdq_state)
    return fp32_model, int8_model, quantized_state


def run_int8_quant_test() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model, int8_model, quantized_state = build_int8_model_from_fp32(device)
    u = flow.make_input(device)

    baseline_prepared, baseline_intermediates = flow.prepare_siso_inputs(fp32_model, u)
    baseline_y, baseline_kernel_debug = repro.siso_kernel_torch(fp32_model, baseline_prepared)
    baseline_out = flow.out_proj(fp32_model, rearrange(baseline_y, "b l h p -> b l (h p)").to(torch.float32))

    u_int8_deq, u_quant = qdq_tensor(u)
    int8_prepared, int8_intermediates = flow.prepare_siso_inputs(int8_model, u_int8_deq)

    quantized_prepared: dict[str, dict[str, torch.Tensor]] = {}
    int8_prepared_deq: dict[str, torch.Tensor] = {}
    for name, tensor in int8_prepared.items():
        deq, record = qdq_tensor(tensor)
        int8_prepared_deq[name] = deq.to(tensor.device)
        quantized_prepared[name] = {k: v.detach().cpu() for k, v in record.items()}

    int8_y, int8_kernel_debug = opq.siso_kernel_op_quant(int8_model, int8_prepared_deq, lambda x: qdq_tensor(x)[0])
    int8_y_deq, int8_y_quant = qdq_tensor(int8_y)
    int8_out = flow.out_proj(int8_model, rearrange(int8_y_deq, "b l h p -> b l (h p)").to(torch.float32))
    int8_out_deq, int8_out_quant = qdq_tensor(int8_out)

    rows = [
        compare_tensors("final_output", int8_out_deq, baseline_out),
        compare_tensors("kernel_y", int8_y_deq, baseline_y),
        compare_tensors("out_v", int8_kernel_debug["out_v"], baseline_kernel_debug["out_v"]),
        compare_tensors("Q_rot", int8_kernel_debug["Q_rot"], baseline_kernel_debug["Q_rot"]),
        compare_tensors("K_scaled", int8_kernel_debug["K_scaled"], baseline_kernel_debug["K_scaled"]),
    ]

    payload = {
        "experiment": "INT8-quant-test",
        "baseline_dtype": str(baseline_out.dtype),
        "device": str(device),
        "u_quant": {k: v.detach().cpu() for k, v in u_quant.items()},
        "weights_quant": quantized_state,
        "prepared_quant": quantized_prepared,
        "baseline_out": repro.detach_cpu(baseline_out),
        "int8_out_dequant": repro.detach_cpu(int8_out_deq),
        "baseline_y": repro.detach_cpu(baseline_y),
        "int8_y_dequant": repro.detach_cpu(int8_y_deq),
        "int8_y_quant": {k: v.detach().cpu() for k, v in int8_y_quant.items()},
        "int8_out_quant": {k: v.detach().cpu() for k, v in int8_out_quant.items()},
        "baseline_intermediates": repro.to_cpu_tree(baseline_intermediates),
        "int8_intermediates_quant": quantize_tree(int8_intermediates),
        "baseline_kernel_debug": repro.to_cpu_tree(baseline_kernel_debug),
        "int8_kernel_debug_quant": quantize_tree(int8_kernel_debug),
        "comparison": rows,
    }
    torch.save(payload, OUT_DIR / "quantized_intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        rows,
        [
            "- experiment: `INT8-quant-test`",
            f"- baseline: PyTorch SISO FP32 output, dtype `{baseline_out.dtype}`",
            "- quantization: symmetric per-tensor int8",
            "- arithmetic model: op-level fake quant; each major kernel op output is quantized back to INT8",
        ],
    )

    old_plot_dir = plots.PLOT_DIR
    plots.PLOT_DIR = PLOT_DIR
    try:
        plot_outputs = plots.save_matrix_plot_set("final_output", int8_out_deq[0], baseline_out[0])
    finally:
        plots.PLOT_DIR = old_plot_dir

    readme = ["# INT8-quant-test Plots", "", "Reference is PyTorch FP32 baseline; actual is INT8 dequantized output.", ""]
    readme.extend(f"- `{path.name}`" for path in plot_outputs)
    (PLOT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    return payload


if __name__ == "__main__":
    run_int8_quant_test()
    print(OUT_DIR / "comparison_report.md")
    print(OUT_DIR / "quantized_intermediates.pt")
