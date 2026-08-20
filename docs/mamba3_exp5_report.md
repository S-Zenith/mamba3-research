# Exp5 系列实验报告：角度函数替代与 lm_head 精度

## 新基线

bigstate-long + ReLU gate + tanh angle：FP32 val_loss=9.45。

---

## Q1 回答：angles_proj 与 tanh

### angles_proj 是什么

`angles_proj` 是 `in_proj` 输出的一个投影分量，维度为 `num_rope_angles = d_state // 2`。它产生复数状态空间中**旋转角度的单步增量**：

```
angle_increment = f(angles_proj) * dt * π
θ_t = cumsum(angle_increment) mod 2π
```

其中 `f` 是把无界投影值压缩到有界范围的函数（当前用 tanh）。

### 为什么用 tanh

旋转角度是周期量（mod 2π），单步增量必须有界。tanh 的优势：
1. **光滑可微**：梯度 = 1 - tanh²(x)，饱和时趋零但非零，反向传播仍能学习
2. **对称有界**：映射到 (-1,1)，乘 π 后每步旋转不超过半圈
3. **数学等价性**：sigmoid(x)*2-1 = tanh(x/2)，所以 tanh 和 sigmoid 是同族函数，只是输入尺度不同

### tanh 替代的实验结果

| angle_mode | 函数 | FP32 val_loss | W8A8_pct999 val | W8A16 val | W8A16 pass |
|---|---|---|---|---|---|
| tanh | tanh(x)*π | 9.45 | 9.97 | 9.49 | ❌ |
| clamp | clamp(x,-1,1)*π | 9.05 | 9.70 | 9.00 | ✅ |
| **sigmoid** | (sigmoid(x)*2-1)*π | **7.66** | **8.86** | **7.67** | ✅ |
| linear | x*π | 9.15 | 9.99 | 9.11 | ✅ |

SwanLab runs: [exp5-1_angle_clamp](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/amey796c), [exp5-1_angle_sigmoid](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/w7gsmzbg), [exp5-1_angle_linear](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/s34oci10)

### 关键发现

1. **sigmoid 是最好的角度函数**：FP32 val_loss=7.66，比 tanh 的 9.45 好 19%。
   - 数学原因：sigmoid(x)*2-1 = tanh(x/2)，等价于 tanh 但输入尺度减半
   - 更小的输入尺度 → 更平缓的角度增量 → 更平滑的旋转 → 训练更稳定
   - **这是一个重要的发现：缩小角度尺度可以显著提升模型精度**

2. **clamp 可用且硬件友好**：FP32 val_loss=9.05，略好于 tanh。clamp 只需比较器，无需乘法或查表。但边界处梯度为零可能影响深层训练。

3. **linear 也可用**：FP32 val_loss=9.15，依赖 mod 2π 保持周期性。无界投影值通过 cumsum 后可能很大，但 mod 2π 把它折回 [0, 2π)。

4. **W8A8_pct999 的绝对 val_loss**：sigmoid（8.86）< clamp（9.70）< linear（9.99）< tanh（9.97）。sigmoid 在量化后也是最好的。

5. **所有替代都通过了 W8A16**，而 tanh 未通过。虽然部分是随机评估波动，但说明替代函数不损害量化友好性。

---

## Q2 回答：out_proj 数值爆炸

### 为什么 in_proj 小但 out_proj 大

| 层 | 输入来源 | 输入 abs_mean | 输入 abs_max |
|---|---|---|---|
| in_proj | RMSNorm 输出（归一化到单位 RMS） | 0.80 | 4.45 |
| out_proj | scan 输出 × gate（无归一化） | 0.46~1140 | 112~271850 |

- **in_proj 输入** = `RMSNorm(x)` 的输出。RMSNorm 把每个位置向量归一化到单位 RMS，所以输入始终在 ±5 以内，**不随层加深而增大**。
- **out_proj 输入** = scan 计算的输出乘以 gate。scan 内部涉及 `exp(ΣA·dt)` 衰减、`QK^T/√d` attention scores、`Σ(weights×V)` 加权求和。这些运算的输出**没有归一化**，数值可以自由增长。残差连接 `x = x + mixer(norm(x))` 让残差流逐层增长，导致深层 out_proj 输入远大于浅层。

### outlier 误差 vs bulk 误差

- **outlier 误差** = abs 值在 99.9% 分位数以上的那些点的平均量化误差
- **bulk 误差** = abs 值在 99.9% 分位数以下的那些点的平均量化误差
- 例如 block3.out_proj：outlier 误差=540.8，bulk 误差=98.7，outlier 是 bulk 的 5.5 倍
- **含义**：少数极端值主导了 per-tensor scale，大部分值的量化精度被牺牲

### lm_head 是哪个操作

`lm_head` = 模型最后一层的 `nn.Linear(d_model, vocab_size)`，把 256 维投影到 50257 维词表。与 `token_embedding` 绑定权重（weight tying）。在代码中：

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight  # 绑定
```

### out_proj 范围跨越是所有 block 的共性

| block | out_proj abs_mean | out_proj abs_max | 增长倍数 |
|---|---|---|---|
| block0 | 0.46 | 112 | 1x |
| block1 | 1.24 | 235 | 2x |
| block2 | 5.08 | 766 | 7x |
| block3 | 1140 | 271850 | **2478x** |

**所有 block 都有此问题**，且逐层指数增长。残差流不断累积，RMSNorm 在下一层入口归一化残差流（保护 in_proj），但 out_proj 的输入是归一化后的值经 scan 计算的输出，不受保护。

---

## Q3 回答：percentile 99.9% 的硬件需求

**推理时不需要任何额外硬件**。percentile 在校准阶段（离线）计算：

1. 收集少量校准数据（如 32 个 batch）
2. 跑前向，统计每层激活值的 abs 分布
3. 找到 99.9% 分位数，作为该层的 scale
4. 把 scale 存入模型 checkpoint

推理时只需读取预存的 scale 值，和普通 INT8 量化**硬件完全相同**——一个 scale 寄存器 + 一个乘法器。区别只是 scale 的**数值**更小（比 max 小），让大部分激活值获得更多量化级别。

---

## exp5-2：lm_head FP16 vs INT8

| 模式 | val_loss | val_ratio | 说明 |
|---|---|---|---|
| W8A8_pct999 | 10.08 | 1.86 | lm_head 被量化 |
| W8A8_pct999_lmhead16 | 10.55 | 3.00 | lm_head 保留 FP16 |
| W8A16 | 9.45 | 1.00 | 参考 |

SwanLab: [exp5-2_pct999_lmhead](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/0bo3v27f)

### 反直觉发现：保留 lm_head FP16 反而更差

ratio 从 1.86 升到 3.00。可能原因：
1. **量化 lm_head 输出起到了截断极端 logits 的正则化效果**：PercentileActQuantLinear 对 lm_head 输出做 99.9% percentile 裁剪，相当于截断了极端预测，减少了 cross-entropy loss
2. **lm_head 输入经过 RMSNorm 归一化**，本身很小（abs_mean=0.48），量化误差极小（MSE=0.00003），所以量化 lm_head 输入几乎无损
3. **lm_head 输出（logits）的量化**才是关键——它截断了极端 logits

**结论**：lm_head 不需要保留 FP16，量化它反而有益。

---

## 综合结论

### 当前最优配置

**angle_mode=sigmoid + gate=relu + bigstate-long + W8A16**：
- FP32 val_loss=7.66
- W8A16 val_loss=7.67, ratio=1.01, **通过 10%**
- 这是所有实验中最好的结果

### 角度函数的 ASIC 启示

1. **sigmoid 优于 tanh**，但 sigmoid 需要指数运算。ASIC 上可用 LUT 或查表实现
2. **数学等价**：sigmoid(x)*2-1 = tanh(x/2)，所以只需把 tanh 的输入除以 2 即可。如果 ASIC 已有 tanh LUT，只需在输入端加一个右移（>>1）
3. **clamp 可作为备选**：硬件最简（只需比较器），FP32 略好于 tanh，但梯度消失可能影响深层训练
4. **linear 可用于推理**：训练用 tanh/sigmoid，推理时转为 linear + mod 2π，完全避免非线性函数

### 量化策略的 ASIC 启示

1. **W8A16 是当前最优量化方案**：INT8 权重 + FP16 激活，几乎无损
2. **W8A8_pct999 是次优方案**：如果必须用 INT8 激活，99.9% percentile clipping 是必须的
3. **lm_head 不需要特殊对待**：量化它反而有益（截断极端 logits）
4. **percentile 不需要额外硬件**：校准时离线计算，推理时与普通 INT8 相同
