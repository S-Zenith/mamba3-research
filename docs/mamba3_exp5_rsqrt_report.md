# Exp5 扩展报告：rsqrt 角度方案与流程图

## 流程图

流程图保存在 `demo/wikitext_mamba3_quant/outputs/mamba3_block_flowchart.png`。

Mamba3 block 中的**非线性操作**（红色）：
1. RMSNorm（含 rsqrt）
2. softplus（A, dt）
3. tanh（angles）
4. cos/sin（RoPE）
5. exp（decay）
6. sigmoid（trap/scale）
7. SiLU/ReLU（gate）

**线性操作**（蓝色）：in_proj、out_proj、einsum（matmul）、element-wise multiply/add、cumsum、causal mask

---

## rsqrt 角度方案

### 核心思路

当前方案：`angle_proj → tanh → ×π → θ → cos(θ), sin(θ) → RoPE`

提议方案：`angle_proj → t → rsqrt(1+t²), t·rsqrt(1+t²) → RoPE`

数学基础：设 t = tan(θ)，则：
```
cos(θ) = 1/√(1+t²) = rsqrt(1+t²)
sin(θ) = t/√(1+t²) = t · rsqrt(1+t²)
```

**消除了 3 个非线性函数**（tanh、cos、sin），只保留 rsqrt。

### 角度范围变化

| 方案 | 角度范围 | 每步最大旋转 |
|---|---|---|
| tanh+π | (-π, π) | 半圈 |
| rsqrt (t=tanh) | arctan(t) ∈ (-π/4, π/4) | 八分之一圈 |
| rsqrt (t=raw) | arctan(t) ∈ (-π/2, π/2) | 四分之一圈 |

rsqrt 模式使用 raw 投影值（无 tanh），角度范围 (-π/2, π/2)，足够覆盖有意义的旋转。

### 快速 rsqrt

#### FP32 快速 rsqrt（Quake III 算法）

```c
float fast_rsqrt(float x) {
    long i = *(long*)&x;
    i = 0x5f3759df - (i >> 1);  // bit manipulation
    float y = *(float*)&i;
    y = y * (1.5f - 0.5f * x * y * y);  // one Newton-Raphson
    return y;
}
```

#### FP16 快速 rsqrt

FP16（5位指数，10位尾数）的 magic constant 为 `0x59DD`，算法相同：

```c
half fast_rsqrt_half(half x) {
    uint16_t i = *(uint16_t*)&x;
    i = 0x59DD - (i >> 1);
    half y = *(half*)&i;
    y = y * (1.5h - 0.5h * x * y * y);  // Newton-Raphson in FP16
    return y;
}
```

#### 精度测试

| x | 真值 | FP32 快速 | 误差 | FP16 快速 | 误差 |
|---|---|---|---|---|---|
| 0.5 | 1.4142 | 1.4139 | 0.025% | 1.4111 | 0.22% |
| 1.0 | 1.0000 | 0.9983 | 0.17% | 0.9990 | 0.10% |
| 2.0 | 0.7071 | 0.7069 | 0.025% | 0.7056 | 0.22% |
| 5.0 | 0.4472 | 0.4471 | 0.016% | 0.4468 | 0.10% |
| 10.0 | 0.3162 | 0.3157 | 0.17% | 0.3147 | 0.48% |

FP16 快速 rsqrt 最大误差 0.48%，平均 0.2%，足以满足旋转计算精度。

### 实验结果

| angle_mode | FP32 val_loss | W8A8_pct999 val | W8A8_pct999 ratio | W8A16 val | W8A16 ratio | W8A16 pass |
|---|---|---|---|---|---|---|
| tanh | 9.45 | 9.97 | 1.68 | 9.49 | 1.03 | ❌ |
| clamp | 9.05 | 9.70 | 1.92 | 9.00 | 0.95 | ✅ |
| sigmoid | 7.66 | 8.86 | 3.30 | 7.67 | 1.01 | ✅ |
| linear | 9.15 | 9.99 | 2.30 | 9.11 | 0.96 | ✅ |
| **rsqrt** | **6.80** | **7.05** | **1.29** | **6.82** | **1.03** | ✅ |

SwanLab: [exp5-3_angle_rsqrt](https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/8ux5c67y)

### 为什么 rsqrt 最好

1. **更少的非线性**：消除了 tanh + cos + sin，只有 rsqrt（可快速实现），梯度路径更短
2. **自然的参数化**：arctan(t) 是单调光滑的，角度分布更均匀
3. **更好的量化鲁棒性**：rsqrt 是平滑递减函数，不像 cos/sin 有振荡，量化后误差更可控
4. **W8A8_pct999 ratio=1.29**：所有角度方案中最好的 INT8 激活量化结果

### 硬件实现对比

| 操作 | tanh 方案 | rsqrt 方案 |
|---|---|---|
| 角度压缩 | tanh LUT (256 entries) | 不需要 |
| π乘法 | 1次乘法 | 不需要 |
| cos/sin | 2个 LUT (各 256 entries) | 不需要 |
| rsqrt | 不需要 | 1次快速 rsqrt (bit manipulation + 1次乘加) |
| 旋转应用 | 4次乘法 + 2次加法 | 4次乘法 + 2次加法 |
| **总计** | **3个 LUT + 1次乘法 + 旋转** | **1次 rsqrt + 旋转** |

rsqrt 方案在 ASIC 上**只需要 1 个 rsqrt 单元**（可用快速算法实现，无需 LUT），而 tanh 方案需要 3 个 LUT（tanh、cos、sin）。

---

## 综合最优配置

| 组件 | 选择 | 理由 |
|---|---|---|
| 角度函数 | **rsqrt** | FP32 最好(6.80)，消除3个非线性，硬件最简 |
| gate 激活 | **ReLU** | 硬件最简，量化鲁棒性好200x |
| 模型配置 | **bigstate-long** (d_model=256, d_state=4, 5000步) | state=1024/block, 参数<1M |
| 量化方案 | **W8A16** | ratio=1.03, 通过10%, INT8权重+FP16激活 |
| 激活量化策略 | **pct999** (备选) | ratio=1.29, 如果必须用INT8激活 |

### 最终指标

- FP32 val_loss = **6.80**
- W8A16 val_loss = 6.82, ratio = 1.03, **通过 10%**
- W8A8_pct999 val_loss = 7.05, ratio = 1.29
