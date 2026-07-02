from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange


ROOT = Path(__file__).resolve().parents[2]
STATE_DICT_PATH = ROOT / "artifacts" / "mamba3_model_info" / "mamba3_demo_state_dict.pt"

D_MODEL = 256
D_STATE = 64
EXPAND = 2
HEADDIM = 64
CHUNK_SIZE = 64
A_FLOOR = 1e-4


@dataclass
class SisoWeights:
    in_proj_weight: torch.Tensor
    out_proj_weight: torch.Tensor
    dt_bias: torch.Tensor
    B_bias: torch.Tensor
    C_bias: torch.Tensor
    B_norm_weight: torch.Tensor
    C_norm_weight: torch.Tensor
    D: torch.Tensor
    d_model: int = D_MODEL
    d_state: int = D_STATE
    expand: int = EXPAND
    headdim: int = HEADDIM
    chunk_size: int = CHUNK_SIZE
    A_floor: float = A_FLOOR
    is_mimo: bool = False
    mimo_rank: int = 1
    num_bc_heads: int = 1
    num_rope_angles: int = 16

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def nheads(self) -> int:
        return self.d_inner // self.headdim


def load_siso_weights(device: torch.device, state_dict: dict[str, torch.Tensor] | None = None) -> SisoWeights:
    if state_dict is None:
        state_dict = torch.load(STATE_DICT_PATH, map_location=device)
    return SisoWeights(
        in_proj_weight=state_dict["in_proj.weight"].to(device=device, dtype=torch.float32),
        out_proj_weight=state_dict["out_proj.weight"].to(device=device, dtype=torch.float32),
        dt_bias=state_dict["dt_bias"].to(device=device, dtype=torch.float32),
        B_bias=state_dict["B_bias"].to(device=device, dtype=torch.float32),
        C_bias=state_dict["C_bias"].to(device=device, dtype=torch.float32),
        B_norm_weight=state_dict["B_norm.weight"].to(device=device, dtype=torch.float32),
        C_norm_weight=state_dict["C_norm.weight"].to(device=device, dtype=torch.float32),
        D=state_dict["D"].to(device=device, dtype=torch.float32),
    )


def quantize_weights(weights: SisoWeights, qdq: Callable[[torch.Tensor], torch.Tensor]) -> SisoWeights:
    return replace(
        weights,
        in_proj_weight=qdq(weights.in_proj_weight),
        out_proj_weight=qdq(weights.out_proj_weight),
        dt_bias=qdq(weights.dt_bias),
        B_bias=qdq(weights.B_bias),
        C_bias=qdq(weights.C_bias),
        B_norm_weight=qdq(weights.B_norm_weight),
        C_norm_weight=qdq(weights.C_norm_weight),
        D=qdq(weights.D),
    )


def make_input(device: torch.device, seed: int = 20260602) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(1, 128, D_MODEL, device=device, dtype=torch.float32, generator=generator)


def rms_norm_no_gate(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(torch.float32)).to(x.dtype)


def prepare_siso_inputs(weights: SisoWeights, u: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    intermediates: dict[str, torch.Tensor] = {}
    zx = F.linear(u, weights.in_proj_weight)
    intermediates["zxBCdtAtrap"] = zx

    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zx,
        [
            weights.d_inner,
            weights.d_inner,
            weights.d_state * weights.num_bc_heads * weights.mimo_rank,
            weights.d_state * weights.num_bc_heads * weights.mimo_rank,
            weights.nheads,
            weights.nheads,
            weights.nheads,
            weights.num_rope_angles,
        ],
        dim=-1,
    )
    intermediates.update({"z_flat": z, "x_flat": x, "B_flat": B, "C_flat": C, "dd_dt": dd_dt, "dd_A": dd_A, "trap_flat": trap, "angles_flat": angles})

    z = rearrange(z, "b l (h p) -> b l h p", p=weights.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=weights.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=weights.mimo_rank, g=weights.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=weights.mimo_rank, g=weights.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    A = -F.softplus(dd_A.to(torch.float32))
    A = torch.clamp(A, max=-weights.A_floor)
    DT = F.softplus(dd_dt + weights.dt_bias)
    ADT = A * DT
    DT = rearrange(DT, "b l h -> b h l")
    ADT = rearrange(ADT, "b l h -> b h l")
    angles = angles.unsqueeze(-2).expand(-1, -1, weights.nheads, -1).to(torch.float32)

    B_norm = rms_norm_no_gate(B, weights.B_norm_weight).squeeze(2)
    C_norm = rms_norm_no_gate(C, weights.C_norm_weight).squeeze(2)
    intermediates.update({"z": z, "x": x, "B_norm": B_norm, "C_norm": C_norm, "A": A, "DT": DT, "ADT": ADT, "trap": trap, "angles": angles})
    prepared = {"Q": C_norm, "K": B_norm, "V": x, "ADT": ADT, "DT": DT, "Trap": trap, "Angles": angles, "Z": z}
    return prepared, intermediates


def out_proj(weights: SisoWeights, y_flat: torch.Tensor) -> torch.Tensor:
    return F.linear(y_flat, weights.out_proj_weight)


def state_dict_from_weights(weights: SisoWeights) -> dict[str, torch.Tensor]:
    return {
        "in_proj.weight": weights.in_proj_weight,
        "out_proj.weight": weights.out_proj_weight,
        "dt_bias": weights.dt_bias,
        "B_bias": weights.B_bias,
        "C_bias": weights.C_bias,
        "B_norm.weight": weights.B_norm_weight,
        "C_norm.weight": weights.C_norm_weight,
        "D": weights.D,
    }
