from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
from mamba_ssm.ops.triton.mamba3.mamba3_siso_fwd import mamba3_siso_fwd


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


def append_known_limits(notes: list[str]) -> list[str]:
    return notes + [
        "- note: 官方 SISO kernel 使用 PTX `tanh.approx.f32/cos.approx.f32/sin.approx.f32`；纯 PyTorch 使用 `torch.tanh/cos/sin`，因此 RoPE 相关张量会有小幅数值差异。",
        "- note: 本脚本没有调用官方 Mamba3 fused forward kernel 来生成 PyTorch 路径，只用官方 block 生成 reference 输出。",
    ]


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


def make_input(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device, dtype=torch.float32, generator=generator)


def rms_norm_no_gate(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # 官方 B_norm/C_norm 这里没有门控输入，只需要在最后一维做 RMSNorm。
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(torch.float32)).to(x.dtype)


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
    intermediates.update(
        {
            "z_flat": z,
            "x_flat": x,
            "B_flat": B,
            "C_flat": C,
            "dd_dt": dd_dt,
            "dd_A": dd_A,
            "trap_flat": trap,
            "angles_flat": angles,
        }
    )

    z = rearrange(z, "b l (h p) -> b l h p", p=model.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=model.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=model.mimo_rank, g=model.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    # A/DT/ADT 与官方 forward 保持同一顺序，ADT 是每步状态衰减的对数域增量。
    A = -F.softplus(dd_A.to(torch.float32))
    A = torch.clamp(A, max=-model.A_floor)
    DT = F.softplus(dd_dt + model.dt_bias)
    ADT = A * DT
    DT = rearrange(DT, "b l h -> b h l")
    ADT = rearrange(ADT, "b l h -> b h l")
    angles = angles.unsqueeze(-2).expand(-1, -1, model.nheads, -1).to(torch.float32)

    B_norm = rms_norm_no_gate(B, model.B_norm.weight).squeeze(2)
    C_norm = rms_norm_no_gate(C, model.C_norm.weight).squeeze(2)

    intermediates.update(
        {
            "z": z,
            "x": x,
            "B_norm": B_norm,
            "C_norm": C_norm,
            "A": A,
            "DT": DT,
            "ADT": ADT,
            "trap": trap,
            "angles": angles,
        }
    )
    prepared = {"Q": C_norm, "K": B_norm, "V": x, "ADT": ADT, "DT": DT, "Trap": trap, "Angles": angles, "Z": z}
    return prepared, intermediates


def apply_rope_pairwise(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    # SISO Triton kernel 按相邻二元组旋转：[x0, x1] -> [x0*cos-x1*sin, x0*sin+x1*cos]。
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


def angle_dt_cumsum(angles: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    # 官方 angle_dt_fwd 会先 tanh(raw_angle) * pi，再乘 DT，并沿序列做累计角度。
    increments = torch.tanh(angles.to(torch.float32)) * torch.pi * rearrange(dt.to(torch.float32), "b h l -> b l h 1")
    return torch.remainder(torch.cumsum(increments, dim=1), 2 * torch.pi).to(angles.dtype)


def siso_kernel_torch(model: Mamba3, prepared: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    # 这个函数是官方 SISO Triton kernel 的 PyTorch 复现版。
    # 你可以把它理解成一个“手写 attention/SSM 混合层”：
    # - Q/K 用来算当前位置和历史位置之间的相似度；
    # - V 是真正被累加/传播的内容；
    # - states 是跨 chunk 保存的递归状态，类似 RNN 的 hidden state；
    # - ADT/DT/Trap 控制每个时间步的衰减和权重。
    #
    # prepared 里的主要形状：
    # - Q, K: [batch, seqlen, qk_heads, d_state]，这里 qk_heads 通常是 1。
    # - V, Z: [batch, seqlen, nheads, headdim]，这里 nheads=8, headdim=64。
    # - DT, ADT, Trap: [batch, nheads, seqlen]。
    # - Angles: [batch, seqlen, nheads, num_rope_angles]。
    #
    # 官方 mamba3_siso_combined 会先把这些输入转成 bfloat16，再进入 Triton kernel。
    # 这里也这么做，是为了让 PyTorch 复现尽量接近官方 kernel 的 dtype 边界。
    Q = prepared["Q"].to(torch.bfloat16)
    K = prepared["K"].to(torch.bfloat16)
    V = prepared["V"].to(torch.bfloat16)
    ADT = prepared["ADT"]
    DT = prepared["DT"]
    Trap = prepared["Trap"].to(torch.bfloat16)
    Angles = angle_dt_cumsum(prepared["Angles"].to(torch.bfloat16), DT)
    Z = prepared["Z"].to(torch.bfloat16)

    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    # Q/K 的 head 数可以少于 V 的 head 数。比如 qk_heads=1，V heads=8。
    # 后面加 bias 时会把 Q/K broadcast 成每个 V head 都有一份。
    assert nheads % nheads_qk == 0

    # C_bias 对应 Q 的 bias，B_bias 对应 K 的 bias。
    # squeeze(1) 是因为 SISO 的 mimo_rank=1，原始形状里有一个长度为 1 的 rank 维度。
    q_bias = model.C_bias.squeeze(1).to(torch.float32)
    k_bias = model.B_bias.squeeze(1).to(torch.float32)

    # Q/K 只有 num_bc_heads 个输入头；加 per-V-head bias 后广播到所有 V heads。
    # PyTorch 里的 None 表示“在这里插入一个新维度”，便于自动广播。
    # 例如 q_bias[None, None, :, :] 会从 [nheads, d_state]
    # 变成 [1, 1, nheads, d_state]，可以加到 [batch, seqlen, 1, d_state] 上。
    Q_pre = Q.to(torch.float32) + q_bias[None, None, :, :]
    K_pre = K.to(torch.float32) + k_bias[None, None, :, :]
    # RoPE 旋转：把 Q/K 的前若干维按二维 pair 做旋转，用于注入位置信息。
    # apply_rope_pairwise 不改变张量形状，只改变数值。
    Q_rot = apply_rope_pairwise(Q_pre, Angles).to(torch.bfloat16)
    K_rot_unscaled = apply_rope_pairwise(K_pre, Angles).to(torch.bfloat16)

    # Trap 控制 K 的缩放方式。sigmoid 后落在 [0, 1]。
    # gamma 表示当前 token 自己贡献的权重。
    # shifted_gamma 表示下一个 token 的一部分权重，被挪到当前 K 上。
    # scale = gamma + shifted_gamma 是最终乘在 K 上的缩放因子。
    trap_sigmoid = torch.sigmoid(Trap.to(torch.float32))
    gamma = DT.to(torch.float32) * trap_sigmoid
    shifted_gamma = torch.zeros_like(gamma)
    shifted_gamma[:, :, :-1] = DT[:, :, 1:].to(torch.float32) * (1.0 - trap_sigmoid[:, :, 1:])
    scale = gamma + shifted_gamma
    # rearrange(scale, "b h l -> b l h 1") 只是调换维度：
    # 从 [batch, heads, seqlen] 变成 [batch, seqlen, heads, 1]，
    # 这样可以和 K_rot_unscaled 的 [batch, seqlen, heads, d_state] 相乘。
    K_scaled = (K_rot_unscaled.to(torch.float32) * rearrange(scale, "b h l -> b l h 1")).to(torch.bfloat16)
    # qk_store 是 Q/K 在同一位置上的点积项。
    # (Q_pre * K_pre).sum(dim=-1) 表示最后一维 d_state 上逐元素相乘再求和。
    # 它后面会作为一个 diagonal/skip 项加到 V 上。
    qk_store = (Q_pre * K_pre).sum(dim=-1) * rearrange(gamma, "b h l -> b l h")

    # out_v 用来保存每个时间步、每个 head 的 pre-gate 输出。
    # states 是跨 chunk 的递归状态，形状 [batch, heads, headdim_v, d_state]。
    # 可以把它看成历史 K/V 信息压缩后的“记忆矩阵”。
    out_v = torch.empty((batch, seqlen, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    states = torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32)
    chunk_records: list[dict[str, torch.Tensor]] = []

    # 按 chunk 处理序列。chunk_size=64 时，长度 128 的序列会分成两个 chunk。
    # 这么做是官方 kernel 的组织方式：chunk 内用并行矩阵运算，chunk 间用 states 串起来。
    for chunk_start in range(0, seqlen, model.chunk_size):
        chunk_end = min(chunk_start + model.chunk_size, seqlen)
        # 取当前 chunk 的切片。
        # q/k/v 的第一维 batch 保留，第二维只取当前 chunk 的 token。
        q = Q_rot[:, chunk_start:chunk_end].to(torch.float32)
        k = K_scaled[:, chunk_start:chunk_end].to(torch.float32)
        v = V[:, chunk_start:chunk_end].to(torch.float32)
        adt = ADT[:, :, chunk_start:chunk_end].to(torch.float32)
        qk = qk_store[:, chunk_start:chunk_end].to(torch.float32)
        chunk_len = chunk_end - chunk_start

        # ADT 是 log-domain 的衰减增量。
        # da_cs 是 chunk 内从开头到当前位置的累计衰减。
        # da_cs_last 是整个 chunk 的总衰减，用于把 states 推进到下一个 chunk。
        # da_cs_rev 是“从当前位置到 chunk 末尾还剩多少衰减”，用于更新 states。
        da_cs = torch.cumsum(adt, dim=-1)
        da_cs_last = adt.sum(dim=-1)
        da_cs_rev = da_cs_last[:, :, None] - da_cs
        states_before = states.clone()

        # 跨 chunk 项：当前 Q 读取上一个 chunk 累积下来的 SSM state。
        # einsum("blhn,bhpn->blhp") 的含义：
        # - q:      b=batch, l=chunk位置, h=head, n=d_state
        # - states: b=batch, h=head, p=headdim_v, n=d_state
        # - 输出:   b=batch, l=chunk位置, h=head, p=headdim_v
        # 也就是每个 token 的 q 向量去读 states 里的记忆矩阵。
        prev = torch.einsum("blhn,bhpn->blhp", q, states)
        # 历史 state 读出来后，还要乘当前位置累计衰减 exp(da_cs)。
        prev = prev * rearrange(torch.exp(da_cs), "b h l -> b l h 1")

        # chunk 内项：严格因果，只允许当前位置 i 读取 j < i 的 K/V。
        # scores[b,h,i,j] = 当前 chunk 内第 i 个 token 的 q，和第 j 个 token 的 k 的点积。
        scores = torch.einsum("bihn,bjhn->bhij", q, k)
        # decay[b,h,i,j] 控制 j 到 i 之间的状态衰减。
        # minimum(..., 0) 是为了避免 exp 正数导致增长，只保留衰减。
        decay = torch.exp(torch.minimum(da_cs[:, :, :, None] - da_cs[:, :, None, :], torch.zeros((), device=Q.device)))
        # causal 是下三角 mask。diagonal=-1 表示不包括当前位置自己，只允许 j < i。
        causal = torch.tril(torch.ones(chunk_len, chunk_len, device=Q.device, dtype=torch.bool), diagonal=-1)
        scores = scores * decay * causal[None, None, :, :]
        # intra 是 chunk 内历史 token 的 V 加权和。
        intra = torch.einsum("bhij,bjhp->bihp", scores, v)

        # y 是进 gate 前的输出，由三部分组成：
        # 1. prev: 来自以前 chunk 的 state；
        # 2. intra: 当前 chunk 内严格因果的历史 token；
        # 3. (D + qk) * v: 当前 token 自身的 skip/diagonal 项。
        y = prev + intra + (model.D.to(torch.float32)[None, None, :, None] + qk[..., None]) * v
        out_v[:, chunk_start:chunk_end] = y

        # 更新 recurrent state，供后续 chunk 使用。
        # v_scaled 先把当前 chunk 的 V 按“到 chunk 末尾的剩余衰减”缩放。
        v_scaled = v * rearrange(torch.exp(da_cs_rev), "b h l -> b l h 1")
        # 新 states = 旧 states 经过整个 chunk 衰减 + 当前 chunk 的 V/K 外积累加。
        # einsum("blhp,blhn->bhpn")：对 chunk 位置 l 求和，形成 [b,h,p,n] 的状态矩阵。
        states = states * rearrange(torch.exp(da_cs_last), "b h -> b h 1 1") + torch.einsum("blhp,blhn->bhpn", v_scaled, k)
        chunk_records.append(
            {
                "chunk_start": torch.tensor(chunk_start),
                "da_cs": da_cs.detach(),
                "states_before": states_before.detach(),
                "states_after": states.detach(),
                "pre_gate": y.detach(),
            }
        )

    # 最后用 Z 做 gate。F.silu(x)=x*sigmoid(x)，也叫 Swish。
    # out_v 和 Z 形状相同，逐元素相乘。
    y = out_v * F.silu(Z.to(torch.float32))
    # debug 保存中间值，方便和官方 kernel 或量化实验逐项对比。
    debug = {
        "Q_pre_bias": Q_pre,
        "K_pre_bias": K_pre,
        "Q_rot": Q_rot,
        "Angles_cumsum": Angles,
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


def official_siso_intermediates(model: Mamba3, prepared: dict[str, torch.Tensor]) -> dict[str, Any]:
    # 这里只作为 reference 调官方 wrapper 暴露中间结果，PyTorch 复现路径不依赖它。
    angles_cumsum = angle_dt_fwd(
        prepared["Angles"].to(torch.bfloat16),
        prepared["DT"],
        chunk_size=model.chunk_size,
    )
    (
        official_y,
        out_v,
        ssm_states,
        da_cs,
        da_cs_sum,
        q_rot,
        k_scaled,
        qk_store,
        scale,
        gamma,
        final_states,
    ) = mamba3_siso_fwd(
        prepared["Q"].to(torch.bfloat16),
        prepared["K"].to(torch.bfloat16),
        prepared["V"].to(torch.bfloat16),
        prepared["ADT"],
        prepared["DT"],
        prepared["Trap"].to(torch.bfloat16),
        model.C_bias.squeeze(1),
        model.B_bias.squeeze(1),
        angles_cumsum,
        model.D,
        prepared["Z"].to(torch.bfloat16),
        None,
        chunk_size=model.chunk_size,
        store_states_adt_outv=True,
        return_final_states=False,
    )
    return {
        "Angles_cumsum": angles_cumsum,
        "official_y": official_y,
        "out_v": out_v,
        "ssm_states": ssm_states,
        "da_cs": da_cs,
        "da_cs_sum": da_cs_sum,
        "Q_rot": q_rot,
        "K_scaled": k_scaled,
        "QK_store": qk_store,
        "Scale": scale,
        "Gamma": gamma,
        "final_states": final_states,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_official_model(device)
    u = make_input(device)

    official_out = model(u)
    prepared, prep_intermediates = prepare_siso_inputs(model, u)
    torch_y, kernel_debug = siso_kernel_torch(model, prepared)
    torch_y_flat = rearrange(torch_y, "b l h p -> b l (h p)")
    torch_out = model.out_proj(torch_y_flat.to(u.dtype))
    official_kernel = official_siso_intermediates(model, prepared)

    rows = [
        compare_tensors("final_output", torch_out, official_out),
        compare_tensors("kernel_y", torch_y, official_kernel["official_y"]),
        compare_tensors("out_v", kernel_debug["out_v"], official_kernel["out_v"]),
        compare_tensors("Q_rot", kernel_debug["Q_rot"], official_kernel["Q_rot"]),
        compare_tensors("K_scaled", kernel_debug["K_scaled"], official_kernel["K_scaled"]),
        compare_tensors("QK_store", kernel_debug["QK_store"].permute(0, 2, 1), official_kernel["QK_store"]),
        compare_tensors("Scale", kernel_debug["scale"], official_kernel["Scale"]),
        compare_tensors("Gamma", kernel_debug["gamma"], official_kernel["Gamma"]),
    ]
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
        "official_kernel": official_kernel,
        "comparison": rows,
    }
    torch.save(to_cpu_tree(payload), OUT_DIR / "intermediates.pt")
    write_report(
        OUT_DIR / "comparison_report.md",
        rows,
        append_known_limits(
            [
                f"- device: `{device}`",
                "- official path: `Mamba3.forward`",
                "- PyTorch path: translated SISO forward without calling official fused kernel",
            ]
        ),
    )


if __name__ == "__main__":
    main()
