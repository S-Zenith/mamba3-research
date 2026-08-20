# Mamba3 组合训练实验报告

## 实验目标

结合 bigstate（d_model=256, d_state=4）和 longtrain（5000步）配置，测试 W8A8 混合精度（in_proj 保留 16bit），并对比 SiLU 和 ReLU 门控激活。

## 复数状态空间

详见 `docs/mamba3_complex_state_space.md`。核心：Mamba3 将复数特征值 `λ = a + bi` 拆成衰减（实部 a，用 `exp` 实现）和旋转（虚部 b，用 `cos/sin` 实现），旋转部分即 RoPE。三角函数不可省略，ASIC 上建议用 LUT。

## 模型配置

```text
preset: bigstate-long
d_model=256, n_layer=4, headdim=64, d_state=4
runtime_state_per_block = 4 * 64 * 4 = 1024
block_static_parameters ≈ 203K (< 1M)
steps = 5000, batch_size=8, seq_len=128, lr=3e-4
```

## 结果对比

| 指标 | SiLU gate | ReLU gate |
|---|---|---|
| FP32 val_loss | **8.37** | 9.45 |
| FP32 test_loss | 8.31 | 9.39 |
| W8A8 val_loss | 17.03 | **12.81** |
| W8A8 val_ratio | 5775x | **28.9x** |
| W8A8_inproj16 val_loss | 16.73 | 12.76 |
| W8A8_inproj16 val_ratio | 4284x | 27.5x |
| W4A16 val_loss | **9.40** | 10.31 |
| W4A16 val_ratio | 2.81x | **2.37x** |

## 关键发现

### 1. bigstate-long 是目前最好的 FP32 配置

FP32 val_loss=8.37（SiLU），优于 longtrain 的 9.89 和 bigstate 的 13.57。说明 d_state=4 比 d_state=2 更有表达力，即使 d_model 更小。

### 2. SiLU FP32 更好，但 ReLU 对 W8A8 量化远更鲁棒

- SiLU FP32 val_loss=8.37 < ReLU 的 9.45（SiLU 更好 11%）
- 但 SiLU W8A8 val_loss=17.03 >> ReLU 的 12.81（ReLU 更好 25%）
- SiLU W8A8 ratio=5775x，ReLU W8A8 ratio=28.9x（**ReLU 量化鲁棒性好 200 倍**）

原因：SiLU 在 0 附近有平滑曲线，量化后斜率信息丢失严重；ReLU 是分段线性，量化后斜率只有 0 和 1，天然对 INT8 友好。

### 3. in_proj 保留 16bit 帮助有限

W8A8_inproj16 vs W8A8：
- SiLU: 16.73 vs 17.03（仅改善 1.8%）
- ReLU: 12.76 vs 12.81（仅改善 0.4%）

说明激活量化误差不是集中在 in_proj，而是分散在所有 Linear 层。仅保留 in_proj 高精度不够，需要全链路混合精度或更精细的 scale 管理。

### 4. W4A16 对两种激活都稳定

- SiLU W4A16 ratio=2.81，ReLU W4A16 ratio=2.37
- ReLU 的 ratio 更低是因为 FP32 基线更高（9.45 vs 8.37），但绝对 val_loss 接近（10.31 vs 9.40）
- 权重量化到 4bit 对两种激活都影响可控

## 与之前实验的纵向对比

| 配置 | FP32 val_loss | W8A8 val_loss | W4A16 val_loss |
|---|---|---|---|
| baseline (1500步) | 22.97 | ~100 | 25.96 |
| longtrain (5000步) | 9.89 | ~100 | 12.26 |
| bigstate-long (5000步, SiLU) | **8.37** | 17.03 | **9.40** |
| bigstate-long (5000步, ReLU) | 9.45 | **12.81** | 10.31 |

bigstate-long + SiLU 把 W8A8 val_loss 从 ~100 降到 17.03，W4A16 从 12.26 降到 9.40。bigstate-long + ReLU 进一步把 W8A8 降到 12.81。

## SwanLab 对比

所有 run 在 https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant

**在同一张图上对比不同训练的方法**：
1. 进入项目页面 https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant
2. 在 Runs 列表中勾选要对比的 run（如 `combined-silu` 和 `combined-relu`）
3. 下方的 Charts 区域会自动将勾选的 run 叠加在同一张图上
4. 可以对比 `train/loss`、`val/loss`、`val/ppl` 等指标曲线

## ASIC 设计启示

1. **ReLU gate 是 ASIC 最优选择**：FP32 仅差 11%，但 W8A8 量化鲁棒性好 200 倍，且硬件实现只需比较器（无乘法）。
2. **d_state=4 比 d_state=2 更好**：在相同 state budget（1024 elements）下，更多 state 维度比更大 d_model 更有效。
3. **in_proj 16bit 不够**：需要全链路混合精度。Quamba 的 per-channel/per-group activation scale 可能比简单保留某层高精度更有效。
4. **W4A16 是可行方案**：两种激活的 W4A16 ratio 都在 2-3x，QAT 后可能进一步降低。
5. **下一步建议**：
   - 在 bigstate-long + ReLU 上做 W8A8 QAT（当前 W8A8 ratio=28.9，QAT 有可能降到 2-3x）
   - 尝试 per-channel activation scale（Quamba 风格）替代 per-tensor
   - 对 scan 路径的中间值单独校准 scale
