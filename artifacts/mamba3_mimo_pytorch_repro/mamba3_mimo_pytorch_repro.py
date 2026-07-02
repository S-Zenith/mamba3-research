from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from mamba_ssm.modules.mamba3 import Mamba3


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "mamba3_mimo_pytorch_repro"

SEED = 20260604
BATCH = 2
SEQ_LEN = 128
D_MODEL = 256
D_STATE = 64
EXPAND = 2
HEADDIM = 64
MIMO_RANK = 4
NGROUPS = 2
CHUNK_SIZE = 8


class _WeightOnlyNorm(nn.Module):
    def __init__(self, dim: int, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))


class DemoMimoBlock(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.d_model = D_MODEL
        self.d_state = D_STATE
        self.expand = EXPAND
        self.headdim = HEADDIM
        self.chunk_size = CHUNK_SIZE
        self.A_floor = 1e-4
        self.is_mimo = True
        self.mimo_rank = MIMO_RANK
        self.d_inner = self.d_model * self.expand
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = NGROUPS
        self.rotary_dim_divisor = 4
        self.num_rope_angles = (self.d_state // self.rotary_dim_divisor)
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, device=device, dtype=torch.float32)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, device=device, dtype=torch.float32))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state, device=device, dtype=torch.float32))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state, device=device, dtype=torch.float32))
        self.B_norm = _WeightOnlyNorm(self.d_state, device)
        self.C_norm = _WeightOnlyNorm(self.d_state, device)
        self.mimo_x = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device, dtype=torch.float32) / self.mimo_rank)
        self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device, dtype=torch.float32))
        self.mimo_o = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim, device=device, dtype=torch.float32) / self.mimo_rank)
        self.D = nn.Parameter(torch.ones(self.nheads, device=device, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, device=device, dtype=torch.float32)


def detach_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().contiguous()


def to_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return detach_cpu(value)
    if isinstance(value, dict):
        return {k: to_cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(v) for v in value]
    return value


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


def apply_rope_mimo(x: torch.Tensor, angles: torch.Tensor, rotary_dim_divisor: int) -> torch.Tensor:
    out = x.clone()
    rotary = x.shape[-1] // rotary_dim_divisor
    first = out[..., :rotary].clone()
    second_start = x.shape[-1] // 2
    second = out[..., second_start : second_start + rotary].clone()
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
    out[..., :rotary] = cos * first - sin * second
    out[..., second_start : second_start + rotary] = sin * first + cos * second
    return out


def angle_dt_cumsum(angles: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    increments = torch.tanh(angles.to(torch.float32)) * torch.pi * dt.to(torch.float32).permute(0, 2, 1).unsqueeze(-1)
    return torch.remainder(torch.cumsum(increments, dim=1), 2 * torch.pi).to(angles.dtype)


def compute_dacs_segsum_torch(da: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, nheads, seqlen = da.shape
    if seqlen % chunk_size != 0:
        raise ValueError(f"SEQ_LEN={seqlen} must be divisible by CHUNK_SIZE={chunk_size} for this repro script")
    nchunks = seqlen // chunk_size
    da_chunks = da.view(bsz, nheads, nchunks, chunk_size)
    da_cs = torch.cumsum(da_chunks, dim=-1)
    da_cs_sum = torch.sum(da_chunks, dim=-1)
    da_cs_rev = da_cs_sum[..., None] - da_cs
    segsum = repeat(da_chunks, "... d -> ... d e", e=chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da.device, dtype=torch.bool), diagonal=-1)
    segsum = segsum.masked_fill(~mask, 0)
    segsum = torch.cumsum(segsum, dim=-2)
    return da_cs.view(bsz, nheads, seqlen), da_cs_rev.view(bsz, nheads, seqlen), segsum


def build_official_model(device: torch.device) -> Mamba3:
    model = Mamba3(
        d_model=D_MODEL,
        d_state=D_STATE,
        expand=EXPAND,
        headdim=HEADDIM,
        ngroups=NGROUPS,
        chunk_size=CHUNK_SIZE,
        is_mimo=True,
        mimo_rank=MIMO_RANK,
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    return model


def build_demo_mimo_model(device: torch.device) -> DemoMimoBlock:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    model = DemoMimoBlock(device)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, device=device, dtype=parameter.dtype, generator=generator) * 0.02)
        model.dt_bias.zero_()
        model.B_bias.fill_(1.0)
        model.C_bias.fill_(1.0)
        model.B_norm.weight.fill_(1.0)
        model.C_norm.weight.fill_(1.0)
        model.mimo_x.fill_(1.0 / model.mimo_rank)
        model.mimo_z.fill_(1.0)
        model.mimo_o.fill_(1.0 / model.mimo_rank)
        model.D.fill_(1.0)
    model.eval()
    return model


def make_input(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device, dtype=torch.float32, generator=generator)


def rms_norm_no_gate(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(torch.float32)).to(x.dtype)


def prepare_mimo_inputs(model: Mamba3, u: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    intermediates: dict[str, torch.Tensor] = {}
    zx = model.in_proj(u)
    intermediates["in_proj_out"] = zx

    z, x, b_raw, c_raw, dd_dt, dd_A, trap, angles = torch.split(
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
    intermediates.update(
        {
            "z_flat": z,
            "x_flat": x,
            "B_flat": b_raw,
            "C_flat": c_raw,
            "dd_dt": dd_dt,
            "dd_A": dd_A,
            "trap_flat": trap,
            "angles_flat": angles,
        }
    )

    z = rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=model.headdim)
    b_raw = rearrange(b_raw, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    c_raw = rearrange(c_raw, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    a = -F.softplus(dd_A.to(torch.float32))
    a = torch.clamp(a, max=-model.A_floor)
    dt = F.softplus(dd_dt + model.dt_bias)
    adt = a * dt
    dt = rearrange(dt, "b l h -> b h l")
    adt = rearrange(adt, "b l h -> b h l")
    angles = angles.unsqueeze(-2).expand(-1, -1, model.nheads, -1).to(torch.float32)

    b_norm = rms_norm_no_gate(b_raw, model.B_norm.weight)
    c_norm = rms_norm_no_gate(c_raw, model.C_norm.weight)
    angles_cumsum = angle_dt_cumsum(angles, dt)

    intermediates.update(
        {
            "z": z,
            "x": x,
            "B_norm": b_norm,
            "C_norm": c_norm,
            "A": a,
            "DT": dt,
            "ADT": adt,
            "trap": trap,
            "angles": angles,
            "angles_cumsum": angles_cumsum,
        }
    )
    prepared = {
        "Q": c_norm,
        "K": b_norm,
        "V": x,
        "ADT": adt,
        "DT": dt,
        "Trap": trap,
        "Angles": angles,
        "Angles_Cumsum": angles_cumsum,
        "Z": z,
    }
    return prepared, intermediates


def silu_tilelang_style(z: torch.Tensor) -> torch.Tensor:
    half = z * 0.5
    return half * torch.tanh(half) + half


def _shifted_gamma(dt: torch.Tensor, trap: torch.Tensor, start: int, end: int, seqlen: int) -> torch.Tensor:
    shifted = torch.zeros(end - start, device=dt.device, dtype=torch.float32)
    if end - start > 1:
        shifted[:-1] = dt[start + 1 : end].to(torch.float32) * torch.sigmoid(-trap[start + 1 : end].to(torch.float32))
    if end < seqlen:
        shifted[-1] = dt[end].to(torch.float32) * torch.sigmoid(-trap[end].to(torch.float32))
    return shifted


def mimo_kernel_torch(model: Mamba3, prepared: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    q_in = prepared["Q"].to(torch.bfloat16)
    k_in = prepared["K"].to(torch.bfloat16)
    v_in = prepared["V"].to(torch.bfloat16)
    z_in = prepared["Z"].to(torch.bfloat16)
    dt = prepared["DT"]
    adt = prepared["ADT"]
    trap = prepared["Trap"].to(torch.bfloat16)
    angles_cumsum = prepared["Angles_Cumsum"]

    bsz, seqlen, rank, qk_heads, d_state = q_in.shape
    nheads, headdim = v_in.shape[-2], v_in.shape[-1]
    chunk_size = model.chunk_size
    nchunks = seqlen // chunk_size
    da_cs, da_cs_rev, segsum = compute_dacs_segsum_torch(adt, chunk_size)

    out = torch.empty((bsz, seqlen, nheads, headdim), device=v_in.device, dtype=torch.bfloat16)
    debug: dict[str, Any] = {"DA_CS": da_cs, "DA_CS_REV": da_cs_rev, "Segsum": segsum, "chunks": []}

    causal_steps = torch.arange(chunk_size, device=v_in.device)
    causal_mask = causal_steps[:, None] > causal_steps[None, :]
    causal_mask = causal_mask.repeat_interleave(rank, dim=0).repeat_interleave(rank, dim=1)

    for ib in range(bsz):
        for ih in range(nheads):
            ih_qk = ih // (nheads // qk_heads)
            state = torch.zeros(d_state, headdim, device=v_in.device, dtype=torch.float32)
            q_bias = model.C_bias[ih].to(torch.bfloat16)
            k_bias = model.B_bias[ih].to(torch.bfloat16)
            psi = model.mimo_x[ih].to(torch.bfloat16)
            phi = model.mimo_o[ih].to(torch.float32)
            zeta = model.mimo_z[ih].to(torch.float32)

            for chunk_idx in range(nchunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                q = q_in[ib, start:end, :, ih_qk, :] + q_bias
                k = k_in[ib, start:end, :, ih_qk, :] + k_bias
                v = v_in[ib, start:end, ih, :]
                z = z_in[ib, start:end, ih, :]

                gamma = dt[ib, ih, start:end].to(torch.float32) * torch.sigmoid(trap[ib, ih, start:end].to(torch.float32))
                shifted = _shifted_gamma(dt[ib, ih], trap[ib, ih], start, end, seqlen)
                trap_scale = (gamma + shifted).to(torch.bfloat16)

                psi_v = v[:, None, :] * psi[None, :, :]
                angle = angles_cumsum[ib, start:end, ih, :].view(1, chunk_size, 1, 1, -1)
                q_rot = apply_rope_mimo(q.view(1, chunk_size, rank, 1, d_state), angle, model.rotary_dim_divisor).view(chunk_size, rank, d_state)
                k_rot = apply_rope_mimo(k.view(1, chunk_size, rank, 1, d_state), angle, model.rotary_dim_divisor).view(chunk_size, rank, d_state)

                q_flat = q_rot.reshape(chunk_size * rank, d_state).to(torch.float32)
                k_flat = k_rot.reshape(chunk_size * rank, d_state).to(torch.float32)
                psi_v_flat = psi_v.reshape(chunk_size * rank, headdim).to(torch.float32)

                inter = q_flat @ state
                inter = inter * torch.exp(da_cs[ib, ih, start:end]).repeat_interleave(rank)[:, None]

                k_scaled = k_flat * trap_scale.repeat_interleave(rank)[:, None].to(torch.float32)
                qk_intrachunk = q_flat @ k_scaled.T
                seg = segsum[ib, ih, chunk_idx].repeat_interleave(rank, dim=0).repeat_interleave(rank, dim=1)
                masked_qk = torch.where(causal_mask, qk_intrachunk * torch.exp(seg), torch.zeros_like(qk_intrachunk))
                intra = masked_qk @ psi_v_flat

                qk_diag = q_flat @ k_flat.T
                qk_diag_r = qk_diag.view(chunk_size, rank, chunk_size, rank)
                psi_v_r = psi_v.to(torch.float32)
                diag_term = torch.empty(chunk_size, rank, headdim, device=v_in.device, dtype=torch.float32)
                for cs in range(chunk_size):
                    diag_term[cs] = torch.einsum("ri,ip->rp", qk_diag_r[cs, :, cs, :], psi_v_r[cs]) * gamma[cs]
                diag_term = diag_term + model.D[ih].to(torch.float32) * psi_v_r

                o_rank = (inter + intra).view(chunk_size, rank, headdim) + diag_term
                z_gate = silu_tilelang_style(z.to(torch.float32)[:, None, :] * zeta[None, :, :])
                o_rank = o_rank * phi[None, :, :] * z_gate
                out[ib, start:end, ih, :] = o_rank.sum(dim=1).to(torch.bfloat16)

                k_state = k_scaled * torch.exp(da_cs_rev[ib, ih, start:end]).repeat_interleave(rank)[:, None]
                state = state * torch.exp(da_cs[ib, ih, end - 1])
                state = state + k_state.T @ psi_v_flat

                if ib == 0 and ih == 0:
                    debug["chunks"].append(
                        {
                            "chunk": chunk_idx,
                            "gamma": gamma,
                            "shifted_gamma": shifted,
                            "trap_scale": trap_scale,
                            "q_rot": q_rot,
                            "k_rot": k_rot,
                            "psi_v": psi_v,
                            "qk_intrachunk": qk_intrachunk,
                            "out_rank": o_rank,
                            "state_after": state.clone(),
                        }
                    )

    debug["torch_y"] = out
    return out, debug


def write_report(path: Path, rows: list[dict[str, float | str]], notes: list[str]) -> None:
    lines = ["# Mamba3 MIMO PyTorch Repro Comparison", ""]
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


def run_pytorch_demo_only(out_dir: Path, device: torch.device, reason: str) -> dict[str, bool | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    model = build_demo_mimo_model(device)
    u = make_input(device)
    with torch.no_grad():
        prepared, prep_intermediates = prepare_mimo_inputs(model, u)
        y, kernel_intermediates = mimo_kernel_torch(model, prepared)
        torch_out = model.out_proj(y.reshape(BATCH, SEQ_LEN, model.d_inner).to(prepared["V"].dtype))

    notes = [
        f"- official MIMO reference unavailable: {reason}",
        "- generated PyTorch MIMO flow with a local demo MIMO block so tensor layouts and intermediates can still be inspected.",
        "- this fallback does not claim numerical agreement with the official kernel.",
    ]
    torch.save(
        to_cpu_tree(
            {
                "config": {
                    "seed": SEED,
                    "batch": BATCH,
                    "seq_len": SEQ_LEN,
                    "d_model": D_MODEL,
                    "d_state": D_STATE,
                    "headdim": HEADDIM,
                    "mimo_rank": MIMO_RANK,
                    "chunk_size": CHUNK_SIZE,
                    "official_available": False,
                },
                "input": u,
                "torch_y": y,
                "torch_out": torch_out,
                "prepared": prepared,
                "prep_intermediates": prep_intermediates,
                "kernel_intermediates": kernel_intermediates,
            }
        ),
        out_dir / "intermediates.pt",
    )
    write_report(out_dir / "comparison_report.md", [], notes)
    return {"official_available": False, "reason": reason}


def official_mimo_kernel_out(model: Mamba3, prepared: dict[str, torch.Tensor]) -> torch.Tensor:
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo

    y = mamba3_mimo(
        Q=prepared["Q"].contiguous(),
        K=prepared["K"].contiguous(),
        V=prepared["V"].contiguous(),
        ADT=prepared["ADT"].contiguous(),
        DT=prepared["DT"].contiguous(),
        Trap=prepared["Trap"].contiguous(),
        Q_bias=model.C_bias.contiguous(),
        K_bias=model.B_bias.contiguous(),
        MIMO_V=model.mimo_x.contiguous(),
        MIMO_Z=model.mimo_z.contiguous(),
        MIMO_Out=model.mimo_o.contiguous(),
        Angles=prepared["Angles"].contiguous(),
        D=model.D.contiguous(),
        Z=prepared["Z"].contiguous(),
        chunk_size=model.chunk_size,
        rotary_dim_divisor=model.rotary_dim_divisor,
        dtype=prepared["V"].dtype,
        return_state=False,
        cu_seqlens=None,
    )
    return model.out_proj(y.reshape(BATCH, SEQ_LEN, model.d_inner).to(prepared["V"].dtype))


def main() -> None:
    if not torch.cuda.is_available():
        result = run_pytorch_demo_only(OUT_DIR, torch.device("cpu"), "CUDA is unavailable; official MIMO TileLang kernel requires CUDA")
        print(f"wrote {OUT_DIR / 'comparison_report.md'}")
        print(result["reason"])
        return
    device = torch.device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    try:
        model = build_official_model(device)
    except (AssertionError, ModuleNotFoundError) as exc:
        result = run_pytorch_demo_only(OUT_DIR, torch.device("cpu"), str(exc))
        print(f"wrote {OUT_DIR / 'comparison_report.md'}")
        print(result["reason"])
        return
    u = make_input(device)

    with torch.no_grad():
        prepared, prep_intermediates = prepare_mimo_inputs(model, u)
        official_out = official_mimo_kernel_out(model, prepared)
        y, kernel_intermediates = mimo_kernel_torch(model, prepared)
        torch_out = model.out_proj(y.reshape(BATCH, SEQ_LEN, model.d_inner).to(prepared["V"].dtype))

    rows = [compare_tensors("final_output", torch_out, official_out)]
    notes = [
        "- note: PyTorch path keeps the MIMO rank dimension explicit and does not call the official fused MIMO forward kernel.",
        "- note: official TileLang uses mixed precision and fast math; small numerical differences are expected.",
        "- note: SISO rank is effectively 1, while MIMO expands V/Z by rank and then reduces with MIMO_Out.",
    ]
    torch.save(
        to_cpu_tree(
            {
                "config": {
                    "seed": SEED,
                    "batch": BATCH,
                    "seq_len": SEQ_LEN,
                    "d_model": D_MODEL,
                    "d_state": D_STATE,
                    "headdim": HEADDIM,
                    "mimo_rank": MIMO_RANK,
                    "chunk_size": CHUNK_SIZE,
                },
                "input": u,
                "official_out": official_out,
                "torch_y": y,
                "torch_out": torch_out,
                "prepared": prepared,
                "prep_intermediates": prep_intermediates,
                "kernel_intermediates": kernel_intermediates,
                "comparison": rows,
            }
        ),
        OUT_DIR / "intermediates.pt",
    )
    write_report(OUT_DIR / "comparison_report.md", rows, notes)
    print(f"wrote {OUT_DIR / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
