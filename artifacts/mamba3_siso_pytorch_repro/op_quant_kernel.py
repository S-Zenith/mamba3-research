from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from einops import rearrange


ROOT = Path(__file__).resolve().parents[2]
REPRO_SCRIPT = ROOT / "artifacts" / "mamba3_siso_pytorch_repro" / "mamba3_siso_pytorch_repro.py"


def load_repro():
    spec = importlib.util.spec_from_file_location("mamba3_siso_pytorch_repro", REPRO_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


repro = load_repro()


class FakeQuantTracker:
    # 这个小类只是包一层量化函数，并记录调用次数。
    # fn 是外面传进来的“量化再反量化”函数，例如：
    # - INT8: 先 round/clamp 到 int8，再乘 scale 反量化回 float；
    # - E4M3: 先转 torch.float8_e4m3fn，再转回 float32；
    # - BF16 state: 先转 bfloat16，再转回 float32。
    #
    # 为什么还转回 float32？因为 PyTorch 的很多算子不支持直接用 int8/float8 做完整计算。
    # 所以这里模拟的是 fake quant：每个 op 后把数值压到目标精度，再用反量化值继续算。
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.fn = fn
        self.count = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # 每调用一次，就表示这里人为插入了一个“低精度存储/输出边界”。
        self.count += 1
        return self.fn(x)


def siso_kernel_op_quant(model, prepared: dict[str, torch.Tensor], quant: Callable[[torch.Tensor], torch.Tensor], state_quant: Callable[[torch.Tensor], torch.Tensor] | None = None):
    # 这是 siso_kernel_torch 的 op-level fake-quant 版本。
    # 和原始 siso_kernel_torch 的数学结构相同，但每个主要 op 后都会调用 qdq(...)
    # 把结果压回目标精度。
    #
    # 参数说明：
    # - model: 纯 PyTorch 权重容器 SisoWeights，或者字段兼容的对象。
    # - prepared: prepare_siso_inputs() 产生的 Q/K/V/ADT/DT/Trap/Angles/Z。
    # - quant: 普通中间变量使用的量化函数。
    # - state_quant: recurrent states 使用的量化函数；如果不传，就和 quant 相同。
    #
    # 四个实验对应关系：
    # - INT8-quant-test: quant=INT8, state_quant=INT8。
    # - AE4M3-WINT8-quant-test: 权重 INT8，quant=E4M3, state_quant=E4M3。
    # - E4M3-quant-test: 权重 E4M3，quant=E4M3, state_quant=E4M3。
    # - INT8-StateBF16-quant-test: quant=INT8, state_quant=BF16。
    qdq = FakeQuantTracker(quant)
    state_qdq = FakeQuantTracker(state_quant or quant)

    # 入口张量先按目标精度 fake-quant 一次。
    # 注意：to(torch.bfloat16) 是为了保持和官方 SISO kernel 的输入 dtype 边界接近；
    # 随后 qdq 会根据实验配置再压到 INT8/E4M3 等目标格式。
    Q = qdq(prepared["Q"].to(torch.bfloat16))
    K = qdq(prepared["K"].to(torch.bfloat16))
    V = qdq(prepared["V"].to(torch.bfloat16))
    ADT = qdq(prepared["ADT"])
    DT = qdq(prepared["DT"])
    Trap = qdq(prepared["Trap"].to(torch.bfloat16))
    Angles = qdq(repro.angle_dt_cumsum(qdq(prepared["Angles"].to(torch.bfloat16)), DT))
    Z = qdq(prepared["Z"].to(torch.bfloat16))

    # 读出形状信息，后面创建 states/out_v 需要。
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    assert nheads % nheads_qk == 0

    # bias 和 D 也属于参数，会经过同样的 quant。
    q_bias = qdq(model.C_bias.squeeze(1).to(torch.float32))
    k_bias = qdq(model.B_bias.squeeze(1).to(torch.float32))
    D = qdq(model.D.to(torch.float32))

    # Q/K 加 bias 后立即 qdq，模拟低精度 op 输出。
    Q_pre = qdq(Q.to(torch.float32) + q_bias[None, None, :, :])
    K_pre = qdq(K.to(torch.float32) + k_bias[None, None, :, :])
    # RoPE 的输出也立刻 qdq。
    Q_rot = qdq(repro.apply_rope_pairwise(Q_pre, Angles)).to(torch.bfloat16)
    K_rot_unscaled = qdq(repro.apply_rope_pairwise(K_pre, Angles)).to(torch.bfloat16)

    # trap/sigmoid/gamma/scale 这些标量控制项也在每个主要 op 后 qdq。
    trap_sigmoid = qdq(torch.sigmoid(Trap.to(torch.float32)))
    gamma = qdq(DT.to(torch.float32) * trap_sigmoid)
    shifted_gamma = torch.zeros_like(gamma)
    shifted_gamma[:, :, :-1] = qdq(DT[:, :, 1:].to(torch.float32) * (1.0 - trap_sigmoid[:, :, 1:]))
    scale = qdq(gamma + shifted_gamma)
    K_scaled = qdq(K_rot_unscaled.to(torch.float32) * rearrange(scale, "b h l -> b l h 1")).to(torch.bfloat16)
    qk_store = qdq((Q_pre * K_pre).sum(dim=-1) * rearrange(gamma, "b h l -> b l h"))

    # out_v 是每个 token 的 pre-gate 输出缓存。
    out_v = torch.empty((batch, seqlen, nheads, headdim_v), device=Q.device, dtype=torch.float32)
    # states_store 是跨 chunk 的 recurrent state。
    # 这里不用 qdq，而用 state_qdq，因为 state 可以有单独精度，例如 BF16。
    states_store = state_qdq(torch.zeros((batch, nheads, headdim_v, headdim_qk), device=Q.device, dtype=torch.float32))
    chunk_records: list[dict[str, torch.Tensor]] = []

    # 按 chunk 做 SISO 扫描。每个 chunk 内的主要算术输出都 qdq。
    for chunk_start in range(0, seqlen, model.chunk_size):
        chunk_end = min(chunk_start + model.chunk_size, seqlen)
        # 当前 chunk 的切片也 qdq 一次，模拟从低精度缓存中读出。
        q = qdq(Q_rot[:, chunk_start:chunk_end].to(torch.float32))
        k = qdq(K_scaled[:, chunk_start:chunk_end].to(torch.float32))
        v = qdq(V[:, chunk_start:chunk_end].to(torch.float32))
        adt = qdq(ADT[:, :, chunk_start:chunk_end].to(torch.float32))
        qk = qdq(qk_store[:, chunk_start:chunk_end].to(torch.float32))
        chunk_len = chunk_end - chunk_start

        # cumsum/sum/减法后都 qdq，避免这些控制量在 FP32 中“偷偷变准”。
        da_cs = qdq(torch.cumsum(adt, dim=-1))
        da_cs_last = qdq(adt.sum(dim=-1))
        da_cs_rev = qdq(da_cs_last[:, :, None] - da_cs)
        # state 从自己的存储精度恢复为 float32，用于 PyTorch 算子计算；
        # 计算后的新 state 会再次通过 state_qdq 存回目标精度。
        states = states_store.to(torch.float32)
        states_before = states.clone()

        # 跨 chunk state 读取项：einsum 输出 qdq，乘 decay 后再 qdq。
        prev = qdq(torch.einsum("blhn,bhpn->blhp", q, states))
        prev = qdq(prev * rearrange(torch.exp(da_cs), "b h l -> b l h 1"))

        # chunk 内因果项：scores、decay、mask 后 scores、intra 全都 qdq。
        scores = qdq(torch.einsum("bihn,bjhn->bhij", q, k))
        decay = qdq(torch.exp(torch.minimum(da_cs[:, :, :, None] - da_cs[:, :, None, :], torch.zeros((), device=Q.device))))
        causal = torch.tril(torch.ones(chunk_len, chunk_len, device=Q.device, dtype=torch.bool), diagonal=-1)
        scores = qdq(scores * decay * causal[None, None, :, :])
        intra = qdq(torch.einsum("bhij,bjhp->bihp", scores, v))

        # 三项相加后的 y 也 qdq，表示 pre-gate 输出以目标精度保存。
        y = qdq(prev + intra + (D[None, None, :, None] + qk[..., None]) * v)
        out_v[:, chunk_start:chunk_end] = y

        # 更新 state：v_scaled qdq，新 states 的计算输出先 qdq，再 state_qdq 存储。
        # 对 INT8-StateBF16 来说，这里 qdq 是 INT8 算术输出，state_qdq 是 BF16 状态存储。
        v_scaled = qdq(v * rearrange(torch.exp(da_cs_rev), "b h l -> b l h 1"))
        states = qdq(states * rearrange(torch.exp(da_cs_last), "b h -> b h 1 1") + torch.einsum("blhp,blhn->bhpn", v_scaled, k))
        states_store = state_qdq(states)
        chunk_records.append(
            {
                "chunk_start": torch.tensor(chunk_start),
                "da_cs": da_cs.detach(),
                "states_before": states_before.detach(),
                "states_after": states_store.to(torch.float32).detach(),
                "pre_gate": y.detach(),
            }
        )

    # 整个 out_v 缓存再 qdq，然后和 gate Z 做 SiLU 门控，门控输出也 qdq。
    out_v = qdq(out_v)
    y = qdq(out_v * F.silu(Z.to(torch.float32)))
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
        # 记录调用次数，方便确认确实进入了 op-level fake quant 路径。
        "quant_op_count": torch.tensor(qdq.count),
        "state_quant_op_count": torch.tensor(state_qdq.count),
    }
    return y.to(torch.float32), debug
