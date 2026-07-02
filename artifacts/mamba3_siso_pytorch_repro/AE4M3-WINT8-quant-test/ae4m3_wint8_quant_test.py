from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[3]
REPRO_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
OUT_DIR = REPRO_DIR / "AE4M3-WINT8-quant-test"
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


def _native_e4m3_dtype():
    return getattr(torch, "float8_e4m3fn", None)


def _software_e4m3_dequant(x: torch.Tensor) -> torch.Tensor:
    # 简化的软件 E4M3 模拟：按 E4M3 的 3-bit mantissa 量化到近似网格。
    x_f = x.detach().to(torch.float32).cpu()
    sign = torch.sign(x_f)
    ax = x_f.abs().clamp(max=448.0)
    out = torch.zeros_like(ax)
    mask = ax > 0
    if mask.any():
        exp = torch.floor(torch.log2(ax[mask])).clamp(-6, 8)
        step = torch.pow(torch.tensor(2.0), exp - 3)
        out[mask] = torch.round(ax[mask] / step) * step
    return out * sign


def quantize_e4m3(x: torch.Tensor) -> dict[str, torch.Tensor | str]:
    x_cpu = x.detach().to(torch.float32).cpu()
    fp8_dtype = _native_e4m3_dtype()
    if fp8_dtype is not None:
        q = x_cpu.to(fp8_dtype)
        deq = q.to(torch.float32)
        return {"q": q, "dequant": deq, "format": "torch.float8_e4m3fn"}

    deq = _software_e4m3_dequant(x_cpu)
    return {"q": deq, "dequant": deq, "format": "software_e4m3_approx"}


def dequantize_e4m3(record: dict[str, torch.Tensor | str]) -> torch.Tensor:
    q = record["q"]
    assert isinstance(q, torch.Tensor)
    if q.is_floating_point() and str(q.dtype).startswith("torch.float8"):
        return q.to(torch.float32)
    deq = record.get("dequant")
    assert isinstance(deq, torch.Tensor)
    return deq.to(torch.float32)


def qdq_activation(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
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


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    row = repro.compare_tensors(name, actual, expected)
    actual_f = actual.detach().to(torch.float64).cpu()
    expected_f = expected.detach().to(torch.float64).cpu()
    row["rel_l2"] = float(torch.linalg.vector_norm(actual_f - expected_f) / torch.linalg.vector_norm(expected_f).clamp_min(1e-12))
    return row


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# AE4M3-WINT8-quant-test", ""]
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


def build_wint8_model_from_fp32(device: torch.device):
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
    wint8_model = flow.load_siso_weights(device, qdq_state)
    return fp32_model, wint8_model, quantized_state


def run_ae4m3_wint8_quant_test() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp32_model, wint8_model, quantized_state = build_wint8_model_from_fp32(device)
    u = flow.make_input(device)

    baseline_prepared, baseline_intermediates = flow.prepare_siso_inputs(fp32_model, u)
    baseline_y, baseline_kernel_debug = repro.siso_kernel_torch(fp32_model, baseline_prepared)
    baseline_out = flow.out_proj(fp32_model, rearrange(baseline_y, "b l h p -> b l (h p)").to(torch.float32))

    u_ae4m3_deq, u_ae4m3 = qdq_activation(u)
    ae4m3_prepared, ae4m3_intermediates = flow.prepare_siso_inputs(wint8_model, u_ae4m3_deq)

    ae4m3_prepared_deq: dict[str, torch.Tensor] = {}
    ae4m3_prepared_quant: dict[str, Any] = {}
    for name, tensor in ae4m3_prepared.items():
        deq, record = qdq_activation(tensor)
        ae4m3_prepared_deq[name] = deq.to(tensor.device)
        ae4m3_prepared_quant[name] = record

    ae4m3_y, ae4m3_kernel_debug = opq.siso_kernel_op_quant(wint8_model, ae4m3_prepared_deq, lambda x: qdq_activation(x)[0])
    ae4m3_y_deq, ae4m3_y_quant = qdq_activation(ae4m3_y)
    ae4m3_out = flow.out_proj(wint8_model, rearrange(ae4m3_y_deq, "b l h p -> b l (h p)").to(torch.float32))
    ae4m3_out_deq, ae4m3_out_quant = qdq_activation(ae4m3_out)

    rows = [
        compare_tensors("final_output", ae4m3_out_deq, baseline_out),
        compare_tensors("kernel_y", ae4m3_y_deq, baseline_y),
        compare_tensors("out_v", ae4m3_kernel_debug["out_v"], baseline_kernel_debug["out_v"]),
        compare_tensors("Q_rot", ae4m3_kernel_debug["Q_rot"], baseline_kernel_debug["Q_rot"]),
        compare_tensors("K_scaled", ae4m3_kernel_debug["K_scaled"], baseline_kernel_debug["K_scaled"]),
    ]

    payload = {
        "experiment": "AE4M3-WINT8-quant-test",
        "baseline_dtype": str(baseline_out.dtype),
        "activation_format": quantize_e4m3(torch.tensor([1.0]))["format"],
        "device": str(device),
        "u_ae4m3": u_ae4m3,
        "weights_int8": quantized_state,
        "prepared_ae4m3": ae4m3_prepared_quant,
        "baseline_out": repro.detach_cpu(baseline_out),
        "ae4m3_wint8_out_dequant": repro.detach_cpu(ae4m3_out_deq),
        "baseline_y": repro.detach_cpu(baseline_y),
        "ae4m3_y_dequant": repro.detach_cpu(ae4m3_y_deq),
        "ae4m3_y_quant": ae4m3_y_quant,
        "ae4m3_out_quant": ae4m3_out_quant,
        "baseline_intermediates": repro.to_cpu_tree(baseline_intermediates),
        "ae4m3_intermediates_quant": quantize_activation_tree(ae4m3_intermediates),
        "baseline_kernel_debug": repro.to_cpu_tree(baseline_kernel_debug),
        "ae4m3_kernel_debug_quant": quantize_activation_tree(ae4m3_kernel_debug),
        "comparison": rows,
    }
    torch.save(payload, OUT_DIR / "quantized_intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        rows,
        [
            "- experiment: `AE4M3-WINT8-quant-test`",
            f"- baseline: PyTorch SISO FP32 output, dtype `{baseline_out.dtype}`",
            f"- activations/intermediates/states: `{payload['activation_format']}`",
            "- weights: symmetric per-tensor INT8",
            "- arithmetic model: op-level fake quant; each major kernel op output is quantized back to activation format",
        ],
    )

    old_plot_dir = plots.PLOT_DIR
    plots.PLOT_DIR = PLOT_DIR
    try:
        plot_outputs = plots.save_matrix_plot_set("final_output", ae4m3_out_deq[0], baseline_out[0])
    finally:
        plots.PLOT_DIR = old_plot_dir

    readme = ["# AE4M3-WINT8-quant-test Plots", "", "Reference is PyTorch FP32 baseline; actual is AE4M3 activation + WINT8 weight dequantized output.", ""]
    readme.extend(f"- `{path.name}`" for path in plot_outputs)
    (PLOT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_ae4m3_wint8_quant_test()
    print(OUT_DIR / "comparison_report.md")
    print(OUT_DIR / "quantized_intermediates.pt")
