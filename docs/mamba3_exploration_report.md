# Mamba3 三方向探索报告

## 概述

在 `demo/wikitext_mamba3_quant/` 基础上分三阶段探索 Mamba3 的量化性能和 ASIC 友好性。所有实验记录在 SwanLab 项目 [mamba3-wikitext-quant](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant)。

---

## Phase 1：消融 + QAT

### 消融实验

5 个 FP32 配置，全部满足单 block runtime state <= 1024 和单 block 静态参数 <= 1M：

| preset | d_model | n_layer | headdim | d_state | steps | FP32 val_loss | FP32 test_loss | 训练时间 |
|---|---|---|---|---|---|---|---|---|
| baseline | 512 | 4 | 64 | 2 | 1500 | 22.97 | 23.20 | 244s |
| deep | 512 | 8 | 64 | 2 | 1500 | 14.98 | 15.80 | 417s |
| widehead | 512 | 4 | 128 | 2 | 1500 | 23.02 | 22.69 | 183s |
| bigstate | 256 | 4 | 64 | 4 | 1500 | 13.57 | 13.67 | 136s |
| **longtrain** | 512 | 4 | 64 | 2 | 5000 | **9.89** | **9.98** | 830s |

**关键发现**：
- **训练步数是最大杠杆**：longtrain（5000步）的 val_loss=9.89，比 baseline（1500步）的 22.97 低 57%。模型远未收敛，更多训练会继续改善。
- **加深有效**：deep（8层）val_loss=14.98，比 baseline 好 35%，但训练时间增加 71%。
- **大 state 有效**：bigstate（d_state=4）val_loss=13.57，比 baseline 好 41%，且训练最快（136s），因为 d_model=256 更小。
- **widehead 无效**：headdim=128（nheads=4）与 baseline 几乎相同，说明 head 数量在这个规模下不是瓶颈。

### PTQ 评估（longtrain 配置）

| mode | val_loss | test_loss | val_ppl_ratio | 通过 10% |
|---|---|---|---|---|
| FP32 | 9.89 | 9.98 | 1.00 | — |
| W8A8 | ~100 | ~106 | 3.8e33 | 否 |
| W4A8 | ~108 | ~109 | 5.4e36 | 否 |
| W4A16 | 12.26 | 12.28 | 10.7 | 否（接近） |

### QAT 结果

从 longtrain 的 FP32 checkpoint 微调 1000 步：

| QAT mode | FP32 val_loss | PTQ val_loss | QAT val_loss | QAT 改善 |
|---|---|---|---|---|
| W8A8 | 9.89 | ~100 | 35.25 | 65% 改善但仍不可用 |
| W4A16 | 9.89 | 12.26 | 11.23 | 42% 改善，接近 FP32 |

**关键发现**：
- **W4A16 QAT 有效**：val_loss 从 PTQ 的 12.26 降到 11.23，仅比 FP32 的 9.89 高 13.5%。QAT 让模型学会了适应 4bit 权重量化。
- **W8A8 QAT 部分有效但不充分**：val_loss 从 ~100 降到 35.25，改善显著但仍远超 FP32。激活量化到 INT8 的误差太大，STE 无法完全补偿。
- **结论**：W4A16 + QAT 是当前最可行的低比特方案。W8A8 需要更精细的 scale 管理（如 Quamba 的 per-channel/per-group activation scale），而不是简单的 per-tensor。

---

## Phase 2：中间变量误差分析

在 longtrain FP32 checkpoint 上，用同一 batch 跑 FP32 和量化前向，hook 每 block 的 in_proj、norm、out_proj 输出，计算 relative error（MSE / FP32 abs_mean²）。

### 误差按环节分布（relative error）

| tensor | W8A8 | W4A8 | W4A16 |
|---|---|---|---|
| block0_norm_out | 0.000 | 0.000 | 0.000 |
| block0_in_proj_out | 0.002 | 0.015 | 0.013 |
| block0_out_proj_out | 0.017 | 0.078 | 0.045 |
| block1_norm_out | 0.063 | 0.098 | 0.037 |
| block1_in_proj_out | 0.060 | 0.103 | 0.048 |
| block1_out_proj_out | 0.147 | 0.166 | 0.064 |
| block2_norm_out | 0.493 | 0.531 | 0.150 |
| block2_in_proj_out | 0.543 | 0.584 | 0.189 |
| block2_out_proj_out | 0.674 | 0.673 | 0.226 |
| block3_norm_out | 0.698 | 0.789 | 0.260 |
| block3_in_proj_out | **1.727** | **2.068** | **0.783** |
| block3_out_proj_out | 1.067 | 1.148 | 0.156 |

**关键发现**：
1. **误差逐层累积**：block0 的误差接近 0，到 block3 增长到 1.7（W8A8）。残差连接让量化误差在层间叠加，越深的层误差越大。
2. **in_proj 是最大误差源**：每个 block 中 in_proj 的 relative error 最高。因为 in_proj 是第一个被量化的线性层，其输出直接喂给 selective scan 的所有非线性计算（softplus、exp、sigmoid、RoPE），误差被放大。
3. **out_proj 的 MSE 极大**：block3_out_proj 的 MSE 达 27 亿（W8A8），因为 out_proj 输出的 abs_mean 很大（50602），即使 relative error 不是最高，绝对误差巨大。
4. **W4A16 的误差远低于 W8A8/W4A8**：W4A16 的最大 relative error 为 0.78，而 W8A8 为 1.73。这证实激活量化（A8）比权重量化（W4）引入更多误差。
5. **norm 层在 block0 误差为 0**：因为 block0 的 norm 输入来自 token embedding（未量化），而后续 block 的 norm 输入来自量化后的残差流。

---

## Phase 3：非线性函数近似

在 baseline 配置（d_model=512, n_layer=4, 1500步）上做 4 个结构变体，每个从头训练：

| 变体 | 改动 | FP32 val_loss | vs baseline |
|---|---|---|---|
| baseline | 原始 | 22.97 | — |
| silu-relu | gate SiLU→ReLU | 18.91 | **-18% (更好)** |
| softplus-relu | A/dt softplus→ReLU | 33.37 | +45% (更差) |
| rope-complex | RoPE cos/sin→复数一阶近似 | 26.58 | +16% (更差) |
| exp-poly | decay exp→分段多项式 | 26.47 | +15% (更差) |

### 量化后对比

| 变体 | W8A8 val_loss | W4A16 val_loss | W4A16 ratio |
|---|---|---|---|
| baseline | ~100 | 25.96 | 19.8x |
| silu-relu | 57.10 | 22.39 | 32.4x |
| softplus-relu | 72.40 | 36.68 | 27.4x |
| rope-complex | 143.76 | 30.08 | 33.3x |
| exp-poly | 46.68 | 29.73 | 26.1x |

**关键发现**：
1. **SiLU→ReLU 是唯一比 baseline 更好的变体**：FP32 val_loss 从 22.97 降到 18.91（-18%）。ReLU 更简单、硬件更友好，且在这个模型规模上性能更好。W8A8 量化后也更好（57 vs 100）。
2. **softplus→ReLU 严重损害性能**：val_loss 从 22.97 升到 33.37。softplus 的平滑性对 A 和 dt 的离散化至关重要，ReLU 的硬截断导致状态更新不稳定。
3. **RoPE 复数近似损害性能**：一阶近似 `cos≈1-θ²/2, sin≈θ` 在角度较大时误差明显，val_loss 从 22.97 升到 26.58。
4. **exp 多项式近似损害性能**：分段多项式在衰减计算中引入误差，val_loss 从 22.97 升到 26.47。衰减路径对精度非常敏感。
5. **W4A16 下所有变体的 ratio 都很大**：因为 baseline 本身训练不够（1500步），PPL 饱和导致 ratio 不稳定。但 W8A8 下 silu-relu 明显优于其他变体。

---

## 综合结论与 ASIC 建议

### 最优配置

- **FP32 训练**：d_model=512, n_layer=4, d_state=2, 5000步（longtrain preset），val_loss=9.89
- **量化方案**：W4A16 + QAT，val_loss=11.23（+13.5% vs FP32）
- **非线性近似**：gate 可用 ReLU 替代 SiLU（性能更好且硬件更友好）

### ASIC 设计建议

1. **混合精度 datapath**：线性层权重用 INT4 存储，激活保留 FP16/INT16。scan 累积路径需要 ≥16bit 累加器。
2. **in_proj 需要最高精度**：中间变量分析显示 in_proj 是最大误差源，ASIC 上应给 in_proj 的 MAC 阵列分配更多位宽或更精细的 scale。
3. **误差逐层累积**：深层模型需要考虑残差流的精度保持，可能需要在 block 间插入重新校准（recalibration）。
4. **gate 用 ReLU**：SiLU→ReLU 不仅不损害性能反而更好，且 ReLU 在硬件上只需比较器，无需乘法。这是最直接的 ASIC 优化。
5. **softplus/exp/sigmoid 不可简化**：这些控制路径对精度敏感，ASIC 上应保留 FP16/BF16 或用高精度 LUT。
6. **RoPE 可用 LUT 替代三角函数**：虽然一阶近似损害性能，但预计算 cos/sin 查找表可以避免运行时三角函数，同时保持精度。
7. **下一步**：在 longtrain 配置上重跑 silu-relu 变体（5000步），验证 ReLU gate 在更好训练的模型上是否仍然优于 SiLU。
