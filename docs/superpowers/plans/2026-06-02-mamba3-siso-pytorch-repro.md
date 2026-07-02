# Mamba3 SISO PyTorch Repro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure PyTorch Mamba3 SISO block forward reproduction that saves intermediates and compares its output with the official package.

**Architecture:** Add one self-contained script under `artifacts/mamba3_siso_pytorch_repro/`. The script instantiates the official Mamba3 block for reference, loads the existing demo state dict, runs a hand-written PyTorch SISO forward translated from official `mamba3.py` and `mamba3_siso_fwd.py`, saves intermediate tensors, and writes a comparison report.

**Tech Stack:** Python, PyTorch, einops, mamba-ssm official package, existing project virtualenv.

---

## File Structure

- Create: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`
  - Owns the full reproduction workflow, helper functions, intermediate capture, comparison, and report writing.
- Runtime output: `artifacts/mamba3_siso_pytorch_repro/intermediates.pt`
  - Created by the script. Stores config, input, official output, PyTorch output, and PyTorch-side intermediates.
- Runtime output: `artifacts/mamba3_siso_pytorch_repro/comparison_report.md`
  - Created by the script. Stores numeric comparison metrics and notes.
- Existing input: `artifacts/mamba3_model_info/mamba3_demo_state_dict.pt`
  - Read-only. Provides the demo Mamba3 block parameters.

## Task 1: Create Reproduction Script Skeleton

**Files:**
- Create: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Verify output directory parent exists**

Run: `ls artifacts`

Expected: output includes `mamba3_model_info`.

- [ ] **Step 2: Create script with imports, constants, path setup, and report helpers**

Add a Python file with these top-level pieces:

```python
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "mamba3_siso_pytorch_repro"
STATE_DICT_PATH = ROOT / "artifacts" / "mamba3_model_info" / "mamba3_demo_state_dict.pt"

SEED = 20260602
BATCH = 1
SEQ_LEN = 128
D_MODEL = 256
D_STATE = 64
EXPAND = 2
HEADDIM = 64
CHUNK_SIZE = 64


def detach_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().contiguous()


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    actual_f = actual.detach().to(torch.float64).cpu()
    expected_f = expected.detach().to(torch.float64).cpu()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-12)
    rel = diff / denom
    return {
        "name": name,
        "shape": str(tuple(actual.shape)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# Mamba3 SISO PyTorch Repro Comparison", ""]
    lines.extend(notes)
    lines.append("")
    lines.append("| tensor | shape | max_abs | mean_abs | max_rel |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['shape']}` | "
            f"{row['max_abs']:.8e} | {row['mean_abs']:.8e} | {row['max_rel']:.8e} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

- [ ] **Step 3: Run syntax check**

Run: `python -m py_compile artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: command exits with status 0.

## Task 2: Add Official Model Loader And Input Generator

**Files:**
- Modify: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Add model creation helper**

Add this function:

```python
def build_official_model(device: torch.device) -> Mamba3:
    if not STATE_DICT_PATH.exists():
        raise FileNotFoundError(f"missing state dict: {STATE_DICT_PATH}")

    model = Mamba3(
        d_model=D_MODEL,
        d_state=D_STATE,
        expand=EXPAND,
        headdim=HEADDIM,
        chunk_size=CHUNK_SIZE,
        is_mimo=False,
        device=device,
        dtype=torch.float32,
    )
    state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

- [ ] **Step 2: Add deterministic input helper**

Add this function:

```python
def make_input(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device, dtype=torch.float32, generator=generator)
```

- [ ] **Step 3: Add minimal main that saves official output**

Add this function and entrypoint:

```python
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_official_model(device)
    u = make_input(device)
    official_out = model(u)

    torch.save({"u": detach_cpu(u), "official_out": detach_cpu(official_out)}, OUT_DIR / "intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        [],
        [f"- device: `{device}`", "- status: official reference generated"],
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script once**

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: `intermediates.pt` and `comparison_report.md` are created.

## Task 3: Implement PyTorch Pre-Kernel Forward Preparation

**Files:**
- Modify: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Add pure PyTorch RMSNorm helper**

Add this function:

```python
def rms_norm_no_gate(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # 官方 B_norm/C_norm 在这里没有 gate，只需要按最后一维做 RMSNorm。
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(torch.float32)).to(x.dtype)
```

- [ ] **Step 2: Add projection and split helper**

Add this function:

```python
def prepare_siso_inputs(model: Mamba3, u: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    intermediates: dict[str, torch.Tensor] = {}
    zx = model.in_proj(u)
    intermediates["zxBCdtAtrap"] = zx

    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zx,
        [
            model.d_inner,
            model.d_inner,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.d_state * model.num_bc_heads * model.mimo_rank,
            model.nheads,
            model.nheads,
            model.nheads,
            model.num_rope_angles,
        ],
        dim=-1,
    )
    intermediates.update({"z_flat": z, "x_flat": x, "B_flat": B, "C_flat": C, "dd_dt": dd_dt, "dd_A": dd_A, "trap_flat": trap, "angles_flat": angles})

    z = rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=model.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    A = -F.softplus(dd_A.to(torch.float32))
    A = torch.clamp(A, max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)
    ADT = A * DT
    DT = rearrange(DT, "b l h -> b h l")
    ADT = rearrange(ADT, "b l h -> b h l")
    angles = angles.unsqueeze(-2).expand(-1, -1, model.nheads, -1).to(torch.float32)

    B_norm = rms_norm_no_gate(B, model.B_norm.weight).squeeze(2)
    C_norm = rms_norm_no_gate(C, model.C_norm.weight).squeeze(2)

    intermediates.update({
        "z": z,
        "x": x,
        "B_norm": B_norm,
        "C_norm": C_norm,
        "A": A,
        "DT": DT,
        "ADT": ADT,
        "trap": trap,
        "angles": angles,
    })
    prepared = {"Q": C_norm, "K": B_norm, "V": x, "ADT": ADT, "DT": DT, "Trap": trap, "Angles": angles, "Z": z}
    return prepared, intermediates
```

- [ ] **Step 3: Run script after wiring helper into main**

Update `main()` to call `prepare_siso_inputs(model, u)` and include these intermediates in `intermediates.pt`.

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: `intermediates.pt` includes `B_norm`, `C_norm`, `DT`, and `ADT`.

## Task 4: Implement PyTorch SISO Kernel Translation

**Files:**
- Modify: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Add RoPE helper matching SISO layout**

Add this function:

```python
def apply_rope_pairwise(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # SISO Triton kernel按相邻二元组旋转：[x0, x1] -> [x0*cos-x1*sin, x0*sin+x1*cos]。
    out = x.clone()
    rotary_pairs = angles.shape[-1]
    pair_part = out[..., : rotary_pairs * 2].reshape(*out.shape[:-1], rotary_pairs, 2)
    x0 = pair_part[..., 0]
    x1 = pair_part[..., 1]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).reshape(*out.shape[:-1], rotary_pairs * 2)
    out[..., : rotary_pairs * 2] = rotated
    return out
```

- [ ] **Step 2: Add pure PyTorch SISO forward helper**

Add this function:

```python
def siso_kernel_torch(model: Mamba3, prepared: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    Q = prepared["Q"]
    K = prepared["K"]
    V = prepared["V"]
    ADT = prepared["ADT"]
    DT = prepared["DT"]
    Trap = prepared["Trap"]
    Angles = prepared["Angles"]
    Z = prepared["Z"]

    batch, seqlen, nheads, headdim_qk = Q.shape
    headdim_v = V.shape[-1]
    q_bias = model.C_bias.squeeze(1).to(torch.float32)
    k_bias = model.B_bias.squeeze(1).to(torch.float32)

    Q_pre = Q.to(torch.float32) + q_bias[None, None, :, :]
    K_pre = K.to(torch.float32) + k_bias[None, None, :, :]
    Q_rot = apply_rope_pairwise(Q_pre, Angles)
    K_rot_unscaled = apply_rope_pairwise(K_pre, Angles)

    trap_sigmoid = torch.sigmoid(Trap.to(torch.float32))
    gamma = DT.to(torch.float32) * trap_sigmoid
    shifted_gamma = torch.zeros_like(gamma)
    shifted_gamma[:, :, :-1] = DT[:, :, 1:].to(torch.float32) * (1.0 - trap_sigmoid[:, :, 1:])
    scale = gamma + shifted_gamma
    K_scaled = K_rot_unscaled * rearrange(scale, "b h l -> b l h 1")
    qk_store = (Q_pre * K_pre).sum(dim=-1) * rearrange(gamma, "b h l -> b l h")

    out_v = torch.empty((batch, seqlen, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    states = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    chunk_records: list[dict[str, torch.Tensor]] = []

    for chunk_start in range(0, seqlen, model.chunk_size):
        chunk_end = min(chunk_start + model.chunk_size, seqlen)
        q = Q_rot[:, chunk_start:chunk_end].to(torch.float32)
        k = K_scaled[:, chunk_start:chunk_end].to(torch.float32)
        v = V[:, chunk_start:chunk_end].to(torch.float32)
        adt = ADT[:, :, chunk_start:chunk_end].to(torch.float32)
        qk = qk_store[:, chunk_start:chunk_end].to(torch.float32)
        chunk_len = chunk_end - chunk_start

        da_cs = torch.cumsum(adt, dim=-1)
        da_cs_last = adt.sum(dim=-1)
        da_cs_rev = da_cs_last[:, :, None] - da_cs
        states_before = states.clone()

        prev = torch.einsum("blhn,bhpn->blhp", q, states)
        prev = prev * rearrange(torch.exp(da_cs), "b h l -> b l h 1")

        scores = torch.einsum("bihn,bjhn->bhij", q, k)
        decay = torch.exp(torch.minimum(da_cs[:, :, :, None] - da_cs[:, :, None, :], torch.zeros((), device=Q.device)))
        causal = torch.tril(torch.ones(chunk_len, chunk_len, device=Q.device, dtype=torch.bool), diagonal=-1)
        scores = scores * decay * causal[None, None, :, :]
        intra = torch.einsum("bhij,bjhp->bihp", scores, v)

        y = prev + intra + (model.D.to(torch.float32)[None, None, :, None] + qk[..., None]) * v
        out_v[:, chunk_start:chunk_end] = y

        v_scaled = v * rearrange(torch.exp(da_cs_rev), "b h l -> b l h 1")
        states = states * rearrange(torch.exp(da_cs_last), "b h -> b h 1 1") + torch.einsum("blhp,blhn->bhpn", v_scaled, k)
        chunk_records.append({"chunk_start": torch.tensor(chunk_start), "da_cs": da_cs.detach(), "states_before": states_before.detach(), "states_after": states.detach(), "pre_gate": y.detach()})

    y = out_v * F.silu(Z.to(torch.float32))
    debug = {
        "Q_pre_bias": Q_pre,
        "K_pre_bias": K_pre,
        "Q_rot": Q_rot,
        "K_rot_unscaled": K_rot_unscaled,
        "K_scaled": K_scaled,
        "gamma": gamma,
        "shifted_gamma": shifted_gamma,
        "scale": scale,
        "QK_store": qk_store,
        "out_v": out_v,
        "chunk_records": chunk_records,
    }
    return y.to(V.dtype), debug
```

- [ ] **Step 3: Wire PyTorch kernel and output projection into main**

Add these lines after `prepare_siso_inputs`:

```python
torch_y, kernel_debug = siso_kernel_torch(model, prepared)
torch_y_flat = rearrange(torch_y, "b l h p -> b l (h p)")
torch_out = model.out_proj(torch_y_flat.to(u.dtype))
```

Then save `torch_y`, `torch_out`, and `kernel_debug` into `intermediates.pt`.

- [ ] **Step 4: Run script and inspect final-output comparison**

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: script completes and report includes `final_output` metrics.

## Task 5: Save Complete Intermediates And Comparison Report

**Files:**
- Modify: `artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Add recursive CPU conversion helper**

Add this function:

```python
def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return detach_cpu(value)
    if isinstance(value, dict):
        return {k: to_cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(v) for v in value]
    return value
```

- [ ] **Step 2: Replace save/report block in main**

Use this block:

```python
rows = [compare_tensors("final_output", torch_out, official_out)]
payload = {
    "config": {
        "seed": SEED,
        "batch": BATCH,
        "seq_len": SEQ_LEN,
        "d_model": D_MODEL,
        "d_state": D_STATE,
        "expand": EXPAND,
        "headdim": HEADDIM,
        "chunk_size": CHUNK_SIZE,
        "is_mimo": False,
    },
    "u": u,
    "official_out": official_out,
    "torch_y": torch_y,
    "torch_out": torch_out,
    "prepared": prepared,
    "intermediates": prep_intermediates,
    "kernel_debug": kernel_debug,
    "comparison": rows,
}
torch.save(to_cpu_tree(payload), OUT_DIR / "intermediates.pt")
write_report(
    OUT_DIR / "comparison_report.md",
    rows,
    [
        f"- device: `{device}`",
        "- official path: `Mamba3.forward`",
        "- PyTorch path: translated SISO forward without calling official fused kernel",
    ],
)
```

- [ ] **Step 3: Run script**

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: `comparison_report.md` has one table row for `final_output`; `intermediates.pt` loads with `torch.load`.

## Task 6: Add Shape And Smoke Tests

**Files:**
- Create: `tests/test_mamba3_siso_pytorch_repro.py`

- [ ] **Step 1: Add tests**

Create this test file:

```python
import importlib.util
from pathlib import Path

import torch


SCRIPT = Path("artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py")


def load_module():
    spec = importlib.util.spec_from_file_location("mamba3_siso_pytorch_repro", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_rope_pairwise_preserves_shape_and_rotates_pairs():
    module = load_module()
    x = torch.tensor([[[[1.0, 0.0, 2.0, 0.0]]]])
    angles = torch.tensor([[[[0.0, torch.pi / 2]]]])
    out = module.apply_rope_pairwise(x, angles)
    expected = torch.tensor([[[[1.0, 0.0, 0.0, 2.0]]]])
    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-6, rtol=0)


def test_compare_tensors_reports_zero_for_identical_tensors():
    module = load_module()
    x = torch.tensor([1.0, 2.0, 3.0])
    row = module.compare_tensors("x", x, x.clone())
    assert row["max_abs"] == 0.0
    assert row["mean_abs"] == 0.0
    assert row["max_rel"] == 0.0
```

- [ ] **Step 2: Run new tests**

Run: `pytest tests/test_mamba3_siso_pytorch_repro.py -v`

Expected: both tests pass.

- [ ] **Step 3: Run script again**

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: script completes and writes updated outputs.

## Task 7: Final Verification

**Files:**
- Read runtime outputs only.

- [ ] **Step 1: Run targeted tests**

Run: `pytest tests/test_mamba3_siso_pytorch_repro.py -v`

Expected: tests pass.

- [ ] **Step 2: Run reproduction script**

Run: `python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py`

Expected: script completes successfully.

- [ ] **Step 3: Run existing test suite**

Run: `pytest tests -v`

Expected: existing tests pass or failures are documented with exact error output.

- [ ] **Step 4: Inspect output files**

Check that these files exist:

```text
artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py
artifacts/mamba3_siso_pytorch_repro/intermediates.pt
artifacts/mamba3_siso_pytorch_repro/comparison_report.md
tests/test_mamba3_siso_pytorch_repro.py
```

Expected: all files exist.

## Self-Review

- Spec coverage: The plan creates the requested new subfolder, implements SISO-only PyTorch flow, saves intermediates, writes Chinese comments in the script, compares with official package output, and runs verification.
- Placeholder scan: No `TBD`, no unbounded future work in implementation tasks, and MIMO is explicitly excluded.
- Type consistency: Helper names used in later tasks are defined earlier: `prepare_siso_inputs`, `siso_kernel_torch`, `to_cpu_tree`, `compare_tensors`, and `write_report`.
