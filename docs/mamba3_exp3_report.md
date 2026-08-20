# Exp3 系列实验报告：精度提升探索

## 新基线

bigstate-long + ReLU gate：d_model=256, n_layer=4, headdim=64, d_state=4, 5000步。FP32 val_loss=9.45, test_loss=9.39。

## exp3-0：W8A8 QAT

从 bigstate-long ReLU 的 FP32 checkpoint 微调 1000 步。

| 指标 | FP32 | W8A8 PTQ | W8A8 QAT |
|---|---|---|---|
| val_loss | 9.45 | 12.81 | **12.27** |
| test_loss | 9.39 | 12.57 | 11.50 |

QAT 将 val_loss 从 12.81 降到 12.27（改善 4.2%）。改善有限，因为 per-tensor 激活量化误差太大，STE 无法完全补偿。SwanLab: [exp3-0_W8A8_relu_QAT](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/76qgwz8x)

## exp3-1：权重/激活精度分离实验

在 FP32 checkpoint 上分别测试权重和激活的不同精度组合：

| 模式 | 权重精度 | 激活精度 | val_loss | test_loss | val_ratio | 通过 10% |
|---|---|---|---|---|---|---|
| FP32 | FP32 | FP32 | 9.45 | 9.39 | 1.00 | — |
| **W8A16** | **INT8** | **FP16** | **9.55** | **9.43** | **1.09** | **✅** |
| W4A16 | INT4 | FP16 | 10.09 | 10.20 | 1.88 | ❌ |
| W16A8 | FP16 | INT8 | 12.56 | 12.63 | 22.20 | ❌ |
| W8A8 | INT8 | INT8 | 13.06 | 12.91 | 36.64 | ❌ |

SwanLab: [exp3-1_ptq_eval_relu](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/lbjwey7b)

### 关键发现

1. **W8A16 通过了 10% 标准**（ratio=1.09）。INT8 权重 + FP16 激活是可行方案，val_loss 仅从 9.45 升到 9.55（+1%）。
2. **激活量化是唯一瓶颈**：W16A8（FP16 权重 + INT8 激活）ratio=22.2，而 W8A16（INT8 权重 + FP16 激活）ratio=1.09。即使权重保持 FP16，INT8 激活仍导致 22 倍 PPL 恶化。
3. **W8A8 的误差主要来自激活侧**：W8A8 ratio=36.6，而 W8A16 ratio=1.09。两者差异完全来自激活量化。
4. **W4A16 接近但未通过**：ratio=1.88，4bit 权重会损失更多信息，但仍远好于任何包含激活量化的方案。

## exp3-2：权重值分析

对 FP32 checkpoint 的每个 Linear 层分析 W8/W4 量化后的权重值变化。

### 各层 W8/W4 量化误差

| 层 | FP32 abs_mean | W8 rel_mse | W4 rel_mse | W4 零值比例 | W4 max_err |
|---|---|---|---|---|---|
| block0.in_proj | 0.036 | 0.000034 | 0.010 | 76% | 0.012 |
| block0.out_proj | 0.035 | 0.000033 | 0.010 | 73% | 0.010 |
| block1.in_proj | 0.035 | 0.000032 | 0.010 | 71% | 0.011 |
| block1.out_proj | 0.035 | 0.000031 | 0.009 | 68% | 0.010 |
| block2.in_proj | 0.035 | 0.000027 | 0.008 | 66% | 0.012 |
| block2.out_proj | 0.035 | 0.000029 | 0.008 | 65% | 0.010 |
| block3.in_proj | 0.036 | 0.000024 | 0.007 | 60% | 0.019 |
| block3.out_proj | 0.035 | 0.000036 | 0.009 | 64% | 0.025 |
| lm_head | 0.784 | 0.000048 | 0.014 | 16% | 0.371 |

### 关键发现

1. **W8 权量化几乎无损**：所有层的 rel_mse < 0.00005，max_err < 0.02。INT8 足以精确表示所有权重。
2. **W4 的主要信息丢失是"小值归零"**：block 层 60-76% 的权重值在 W4 后变成零。因为 block 层的 abs_mean=0.035，很多权重值很小，在 4bit scale 下被 round 到 0。
3. **lm_head 对 W4 最敏感**：abs_mean=0.784（远大于 block 的 0.035），W4 max_err=0.371。lm_head 与 token_embedding 绑定，权重范围大，4bit 的 16 个级别不足以覆盖。
4. **深层 block 的 W4 误差更大**：block3 的 max_err=0.019-0.025，大于 block0 的 0.010-0.012。说明深层权重可能更分散。
5. **per-channel 变异系数（CV）**：scale_per_channel_cv 字段显示各通道 scale 的变异程度。CV 越大，per-tensor scale 越不合适，需要 per-channel scale。

## 结论与 ASIC 建议

### 最优量化方案

**W8A16（INT8 权重 + FP16 激活）**是当前最佳方案：
- val_ratio=1.09，通过 10% 标准
- 权重存储减半（FP32→INT8），激活保持 FP16
- ASIC 上只需 INT8 权重 SRAM + FP16 激活 datapath

### 激活量化的根本问题

激活量化到 INT8 会导致 22 倍 PPL 恶化（W16A8），这不是权重精度的问题，而是激活分布的离群值导致 per-tensor scale 过大，大部分激活值被压缩到很少的几个 INT8 级别。解决方案：
1. **per-channel activation scale**（Quamba 风格）：为每个通道单独校准 scale
2. **percentile clipping**：用 99.9% 分位数而非 max 作为 scale
3. **Hadamard 变换**：平滑激活分布，减少离群值

### ASIC 设计参数

| 组件 | 推荐精度 | 理由 |
|---|---|---|
| 权重存储 | INT8 | W8 rel_mse < 0.0001，几乎无损 |
| 激活 datapath | FP16/BF16 | INT8 激活导致 22x PPL 恶化 |
| 累加器 | FP16/BF16 | scan 路径累积需要高精度 |
| cos/sin (RoPE) | LUT | 需要精确值，多项式近似不够 |
| softplus/exp/sigmoid | FP16/BF16 | 控制路径对精度敏感 |
| gate | ReLU (INT1) | 硬件最简，且量化鲁棒性好 200x |
