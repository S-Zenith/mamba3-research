# Exp4 系列实验报告：激活值量化误差分析与策略

## 新基线

bigstate-long + ReLU gate：FP32 val_loss=9.45, test_loss=9.39。

## exp4-1：激活值 FP32→INT8 量化误差分析

### 方法

在训练好的 FP32 checkpoint 上，用固定 batch 跑前向，hook 每个 Linear 层的输入激活值。对每个激活张量：
1. 应用 per-tensor INT8 量化
2. 记录原始值、量化值、误差
3. 画散点图（FP32 vs INT8，颜色=误差大小）
4. 计算放大因子：`||W(x-x_q)|| / ||x-x_q||`，即量化误差经权重矩阵后的放大倍数

输出文件：
- `activation_scatter.png`：9 个 Linear 层的 FP32→INT8 散点图
- `amplification_factor.png`：各层放大因子柱状图
- `activation_analysis.json`：完整统计数据

### 各层激活值统计

| 层 | FP32 abs_mean | FP32 abs_max | INT8 唯一值数 | MSE | 放大因子 | outlier误差 | bulk误差 |
|---|---|---|---|---|---|---|---|
| block0.in_proj | 0.80 | 4.45 | 228 | 0.0001 | 1.00 | 0.010 | 0.009 |
| block0.out_proj | 0.46 | 112.6 | 133 | 0.024 | 0.68 | 0.211 | 0.086 |
| block1.in_proj | 0.81 | 4.46 | 230 | 0.0001 | 0.98 | 0.009 | 0.009 |
| block1.out_proj | 1.24 | 235.5 | 202 | 0.098 | 0.69 | 0.461 | 0.175 |
| block2.in_proj | 0.82 | 4.30 | 231 | 0.0001 | 0.97 | 0.008 | 0.009 |
| block2.out_proj | 5.08 | 766.0 | 171 | 0.916 | 0.73 | 1.519 | 0.506 |
| block3.in_proj | 0.85 | 4.18 | 211 | 0.0001 | 0.99 | 0.008 | 0.008 |
| **block3.out_proj** | **1140** | **271850** | 214 | **57236** | 0.97 | **540.8** | 98.7 |
| **lm_head** | 0.48 | 2.24 | 232 | 0.00003 | **220.3** | 0.004 | 0.004 |

### 关键发现

1. **out_proj 激活值逐层爆炸**：
   - abs_mean 从 block0 的 0.46 涨到 block3 的 1140（2478 倍）
   - abs_max 从 112 涨到 271850（2422 倍）
   - 残差流不断累积，深层 out_proj 的输入激活值远大于浅层

2. **per-tensor scale 被离群值主导**：
   - block3.out_proj 的 abs_max=271850，但 abs_mean=1140
   - INT8 scale = 271850/127 = 2141，但大部分值在 1140 附近
   - 1140 附近的值只有 1140/2141*127 ≈ 67 个量化级别可用
   - **outlier 误差（540.8）是 bulk 误差（98.7）的 5.5 倍**

3. **lm_head 放大因子高达 220 倍**：
   - 输入激活值很小（abs_mean=0.48），量化误差也很小（MSE=0.00003）
   - 但权重矩阵 [50257, 256] 将 256 维误差投影到 50257 维
   - 误差在 50257 个输出维度上累加，放大 220 倍
   - **lm_head 是激活量化误差被放大的最关键环节**

4. **in_proj 量化几乎无损**：
   - 激活值小且分布均匀（abs_mean~0.8, abs_max~4.5）
   - 228/255 个 INT8 级别被使用
   - 放大因子~1.0，无误差放大

5. **FP32 唯一值数量 vs INT8**：
   - FP32 有 13-26 万个唯一值
   - INT8 只有 133-232 个唯一值
   - 信息压缩比约 1000:1

### 误差放大分析

放大因子 = `||W(x - x_q)||₂ / ||x - x_q||₂`，衡量量化误差经权重矩阵后的放大程度。

- **放大因子 > 1**：误差被放大（lm_head: 220x）
- **放大因子 < 1**：误差被压缩（out_proj: 0.68-0.73）
- **放大因子 ≈ 1**：误差不变传递（in_proj: 0.97-1.00）

虽然 out_proj 的放大因子 < 1，但其绝对误差极大（block3 MSE=57236），因为输入激活值本身极大。**问题不在放大，而在激活值分布的范围过大**。

## exp4-2：量化策略实验

### 策略设计

基于 exp4-1 的分析，问题根源是 per-tensor scale 被离群值主导。设计三种策略：

1. **W8A8_perchan**：per-channel 激活 scale，每个输入特征维度独立 scale
2. **W8A8_pct95/99/999**：percentile clipping，用 95%/99%/99.9% 分位数替代 max 作为 scale

### 结果

| 模式 | val_loss | test_loss | val_ratio | vs W8A8改善 |
|---|---|---|---|---|
| FP32 | 9.45 | 9.39 | 1.00 | — |
| W8A8 | 13.06 | 12.91 | 36.64 | — |
| **W8A8_perchan** | **10.01** | 9.94 | **1.74** | **21x** |
| W8A8_pct95 | 14.21 | 14.10 | 116.35 | 更差 |
| W8A8_pct99 | 10.17 | 10.26 | 2.05 | 18x |
| **W8A8_pct999** | **9.97** | 10.03 | **1.68** | **22x** |
| W8A16 (无激活量化) | 9.49 | 9.61 | 1.03 | 参考 |

SwanLab: [exp4-2_quant_strategies](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/9kyemyiw)

### 关键发现

1. **per-channel scale 是最大单项改善**：PPL ratio 从 36.6 降到 1.74（21 倍改善）。每个通道独立 scale，离群通道不再主导其他通道的 scale。

2. **99.9% percentile 略优于 per-channel**：ratio=1.68 vs 1.74。因为 percentile 同时处理通道间和通道内离群值，而 per-channel 只处理通道间差异。

3. **95% percentile 过于激进**：ratio=116，比不裁剪还差。裁掉 5% 的值引入的截断误差超过了 scale 改善带来的收益。

4. **所有 W8A8 策略仍未通过 10%**：最好的 W8A8_pct999 ratio=1.68。剩余误差来自 INT8 本身的 256 级精度限制。

5. **W8A16 仍是最优**（ratio=1.03），但 W8A8_pct999（1.68）已接近。

### 信息丢失的根源

从散点图和数据可以定位信息丢失的三个层面：

1. **值域层面**：out_proj 激活值范围跨越 5 个数量级（0.1 到 271850），per-tensor INT8 的 256 级无法覆盖
2. **通道层面**：不同通道的激活幅度差异大，per-tensor scale 被最大通道主导
3. **层间层面**：深层 out_proj 的激活值远大于浅层，但所有层共用相同的量化策略

## 结论与下一步

### 当前最优方案

| 方案 | val_ratio | 硬件代价 |
|---|---|---|
| W8A16 | 1.03 | INT8权重 + FP16激活 |
| W8A8_pct999 | 1.68 | INT8权重 + INT8激活(999裁剪) |
| W8A8_perchan | 1.74 | INT8权重 + INT8激活(per-channel scale) |

### 下一步建议

1. **组合 per-channel + percentile**：per-channel scale 基础上加 percentile clipping，可能进一步降到 ratio < 1.3
2. **lm_head 保持 FP16**：放大因子 220x，即使其他层用 INT8 激活，lm_head 应保留 FP16
3. **深层 out_proj 专属策略**：block3.out_proj 的激活值比 block0 大 2478 倍，可能需要层专属 scale 或 FP16
4. **QAT + per-channel scale**：在 W8A8_perchan 基础上做 QAT，可能让 ratio 从 1.74 降到 < 1.2
5. **混合精度**：in_proj 用 INT8（几乎无损），out_proj 用 FP16（避免大值量化），lm_head 用 FP16（避免 220x 放大）
