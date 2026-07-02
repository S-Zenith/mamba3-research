from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange


ROOT = Path(__file__).resolve().parents[3]
REPRO_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
OUT_DIR = REPRO_DIR / "INT8-StateBF16-quant-test"
PLOT_DIR = OUT_DIR / "plots"
REPRO_SCRIPT = REPRO_DIR / "mamba3_siso_pytorch_repro.py"
PLOT_SCRIPT = REPRO_DIR / "plot_error_surfaces.py"
INT8_SCRIPT = REPRO_DIR / "INT8-quant-test" / "int8_quant_test.py"
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
int8 = load_module(INT8_SCRIPT, "int8_quant_test")
opq = load_module(OP_QUANT_SCRIPT, "op_quant_kernel")
flow = load_module(FLOW_SCRIPT, "pytorch_siso_flow")


def quantize_state_bf16(x: torch.Tensor) -> dict[str, torch.Tensor]:
    q = x.detach().to(torch.float32).cpu().to(torch.bfloat16)
    return {"q": q, "dequant": q.to(torch.float32)}


def dequantize_state_bf16(record: dict[str, torch.Tensor]) -> torch.Tensor:
    return record["q"].to(torch.float32)


def is_state_name(name: str) -> bool:
    return name in {"states", "states_before", "states_after", "ssm_state", "state"} or name.endswith("_state")


def qdq_int8(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return int8.qdq_tensor(x)


def quantize_tree_mixed(value: Any, name: str = "") -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return quantize_state_bf16(value) if is_state_name(name) else int8.quantize_tree(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: quantize_tree_mixed(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [quantize_tree_mixed(v, name) for v in value]
    return value


def siso_kernel_int8_statebf16(model, prepared: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    return opq.siso_kernel_op_quant(
        model,
        prepared,
        lambda x: qdq_int8(x)[0],
        state_quant=lambda x: x.to(torch.bfloat16).to(torch.float32),
    )


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    row = repro.compare_tensors(name, actual, expected)
    actual_f = actual.detach().to(torch.float64).cpu()
    expected_f = expected.detach().to(torch.float64).cpu()
    row["rel_l2"] = float(torch.linalg.vector_norm(actual_f - expected_f) / torch.linalg.vector_norm(expected_f).clamp_min(1e-12))
    return row


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# INT8-StateBF16-quant-test", ""]
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
            deq, record = int8.qdq_tensor(tensor)
            qdq_state[name] = deq.to(tensor.dtype)
            quantized_state[name] = {k: v.detach().cpu() for k, v in record.items()}
        else:
            qdq_state[name] = tensor
    int8_model = flow.load_siso_weights(device, qdq_state)
    return fp32_model, int8_model, quantized_state


def run_int8_statebf16_quant_test() -> dict[str, Any]:
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

    u_int8_deq, u_quant = qdq_int8(u)
    prepared, intermediates = flow.prepare_siso_inputs(int8_model, u_int8_deq)
    prepared_deq: dict[str, torch.Tensor] = {}
    prepared_quant: dict[str, Any] = {}
    for name, tensor in prepared.items():
        deq, record = qdq_int8(tensor)
        prepared_deq[name] = deq.to(tensor.device)
        prepared_quant[name] = {k: v.detach().cpu() for k, v in record.items()}

    quant_y, quant_kernel_debug = siso_kernel_int8_statebf16(int8_model, prepared_deq)
    quant_y_deq, quant_y_record = qdq_int8(quant_y)
    quant_out = flow.out_proj(int8_model, rearrange(quant_y_deq, "b l h p -> b l (h p)").to(torch.float32))
    quant_out_deq, quant_out_record = qdq_int8(quant_out)

    rows = [
        compare_tensors("final_output", quant_out_deq, baseline_out),
        compare_tensors("kernel_y", quant_y_deq, baseline_y),
        compare_tensors("out_v", quant_kernel_debug["out_v"], baseline_kernel_debug["out_v"]),
        compare_tensors("Q_rot", quant_kernel_debug["Q_rot"], baseline_kernel_debug["Q_rot"]),
        compare_tensors("K_scaled", quant_kernel_debug["K_scaled"], baseline_kernel_debug["K_scaled"]),
    ]

    payload = {
        "experiment": "INT8-StateBF16-quant-test",
        "baseline_dtype": str(baseline_out.dtype),
        "device": str(device),
        "u_int8": {k: v.detach().cpu() for k, v in u_quant.items()},
        "weights_int8": quantized_state,
        "prepared_int8": prepared_quant,
        "state_format": "torch.bfloat16",
        "baseline_out": repro.detach_cpu(baseline_out),
        "int8_statebf16_out_dequant": repro.detach_cpu(quant_out_deq),
        "baseline_y": repro.detach_cpu(baseline_y),
        "int8_statebf16_y_dequant": repro.detach_cpu(quant_y_deq),
        "quant_y_int8": {k: v.detach().cpu() for k, v in quant_y_record.items()},
        "quant_out_int8": {k: v.detach().cpu() for k, v in quant_out_record.items()},
        "baseline_intermediates": repro.to_cpu_tree(baseline_intermediates),
        "quant_intermediates_mixed": quantize_tree_mixed(intermediates),
        "baseline_kernel_debug": repro.to_cpu_tree(baseline_kernel_debug),
        "quant_kernel_debug_mixed": quantize_tree_mixed(quant_kernel_debug),
        "comparison": rows,
    }
    torch.save(payload, OUT_DIR / "quantized_intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        rows,
        [
            "- experiment: `INT8-StateBF16-quant-test`",
            f"- baseline: PyTorch SISO FP32 output, dtype `{baseline_out.dtype}`",
            "- weights/non-state intermediates: symmetric per-tensor INT8",
            "- recurrent states: BF16 storage",
            "- arithmetic model: op-level fake quant; each major non-state kernel op output is quantized back to INT8",
        ],
    )

    old_plot_dir = plots.PLOT_DIR
    plots.PLOT_DIR = PLOT_DIR
    try:
        plot_outputs = plots.save_matrix_plot_set("final_output", quant_out_deq[0], baseline_out[0])
    finally:
        plots.PLOT_DIR = old_plot_dir

    readme = ["# INT8-StateBF16-quant-test Plots", "", "Reference is PyTorch FP32 baseline; actual is INT8 non-state + BF16 state dequantized output.", ""]
    readme.extend(f"- `{path.name}`" for path in plot_outputs)
    (PLOT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_int8_statebf16_quant_test()
    print(OUT_DIR / "comparison_report.md")
    print(OUT_DIR / "quantized_intermediates.pt")
