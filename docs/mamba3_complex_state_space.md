# Mamba3 复数状态空间与三角函数的关系

## 核心结论

Mamba3 的状态空间模型（SSM）使用**复数对角状态转移**，但在实现中**从不显式使用复数运算**。它将复数特征值 `λ = a + bi` 拆成两个实数量，分别实现**衰减**（实部 `a`）和**旋转**（虚部 `b`），旋转部分天然需要 `cos/sin`。

## 数学背景

### 对角 SSM 递推

SSM 的核心递推是：

```
h_t = Ā · h_{t-1} + B̄ · x_t
```

其中 `Ā = exp(A·dt)` 是对角状态转移矩阵。

### 实数特征值（Mamba1/Mamba2）

当 `A` 是实负数时（Mamba1/Mamba2 中 `A = -exp(A_log)`）：

```
Ā = exp(A·dt) ∈ (0, 1)
```

这是**纯衰减**，没有振荡。状态单调趋零。

### 复数特征值（Mamba3）

当 `A` 是复数 `a + bi` 时，复特征值成共轭对 `a ± bi`：

```
exp((a+bi)·dt) = exp(a·dt) · exp(i·b·dt)
              = exp(a·dt) · (cos(b·dt) + i·sin(b·dt))
```

写成实 2×2 矩阵形式（对共轭对）：

```
exp(a·dt) · [[cos(b·dt), -sin(b·dt)],
             [sin(b·dt),  cos(b·dt)]]
```

这就是**衰减 × 旋转矩阵**。三角函数 `cos/sin` 自然出现，因为复指数 `exp(iθ)` 的实部和虚部就是 `cos θ` 和 `sin θ`。

## Mamba3 的实现方式

Mamba3 不使用复数算术，而是**把复数转移拆成两个独立的实数量**：

### 1. 衰减部分（实部 a）

```python
A = -F.softplus(dd_A)     # A < 0，实部
dt = F.softplus(dd_dt + dt_bias)
adt = A * dt              # a·dt
# 累积：
cs = torch.cumsum(adt, dim=-1)        # Σ a·dt
decay = torch.exp(cs[:,:,None] - cs[:,None,:])  # exp(Σ(a·dt_i - a·dt_j))
```

衰减是标量乘法，状态在每步乘以 `exp(a·dt) < 1`，单调衰减。

### 2. 旋转部分（虚部 b）

```python
# 角度投影：
angles = torch.tanh(angles_proj) * π   # 将投影值压缩到 (-π, π)

# 递推累积角度（等价于 Σ b·dt）：
increments = tanh(angles) * dt * π
θ_t = cumsum(increments) mod 2π       # 旋转角度状态

# 应用旋转到 Q/K 的坐标对：
cos_θ = cos(θ_t)
sin_θ = sin(θ_t)
# 对 (x0, x1) 对：
x0' = x0 * cos_θ - x1 * sin_θ
x1' = x0 * sin_θ + x1 * cos_θ
```

这就是 **RoPE（旋转位置编码）**，但关键区别是：Transformer 中 `θ` 是位置的固定函数，而 Mamba3 中 `θ` 是**输入依赖、递推累积**的状态量。

### 3. 两部分的关系

复数转移 `exp((a+bi)·dt)` 被实现为：
- 先用衰减 `exp(a·dt)` 缩放状态（通过 `cumsum` + `exp`）
- 再用旋转矩阵 `[[cos θ, -sin θ], [sin θ, cos θ]]` 旋转 Q/K（通过 RoPE）

两部分作用于不同的维度：衰减作用于整个 `d_state`，旋转只作用于 `d_state // 2` 对（即 `num_rope_angles = d_state // 2`）。

## 源码位置

| 组件 | 文件 | 关键行 |
|---|---|---|
| 角度递推（Triton kernel） | `ops/triton/mamba3/angle_dt.py` | 94-108: `tanh(angle)*π*dt`, `cumsum`, `mod 2π` |
| cos/sin 旋转（Triton kernel） | `ops/triton/mamba3/mamba3_siso_fwd.py` | 309-331: `cos_approx`, `sin_approx`, 2×2 旋转 |
| 共轭处理 | `ops/triton/mamba3/mamba3_mimo_rotary_step.py` | 86: `if CONJUGATE: sin = -sin` |
| PTX 三角近似 | `ops/triton/mamba3/utils.py` | 13-50: `cos.approx.f32`, `sin.approx.f32` 内联汇编 |
| 衰减累积 | `mamba3_siso_fwd.py` | 367-388: `cumsum(A·dt)`, `exp2` |
| PyTorch 复现 | `demo/tiny_causal_lm/tiny_mamba3_lm.py` | 185-203: `softplus→A`, `cumsum→decay`, `tanh→angles`, `cos/sin→RoPE` |
| 官方模块 | `modules/mamba3.py` | 76-83: `rope_fraction`, `num_rope_angles = d_state//2` |

## 与 Quamba（Mamba1/Mamba2）的对比

Quamba 的 `csrc/selective_scan/quant_sscan_common.h` 第 106-111 行直接用 C++ 复数类型：

```cpp
__device__ complex_t cexpf(complex_t z) {
    float t = expf(z.real_);      // 衰减
    sincosf(z.imag_, &s, &c);     // 旋转
    return complex_t(c * t, s * t);  // exp(a)·(cos b + i·sin b)
}
```

这是**显式复数路径**：`A` 是复数，`exp(A·dt)` 用 `expf + sincosf` 计算。

Mamba3 的创新在于**绕过复数算术**：把 `a` 和 `b` 拆成两个独立的实投影，分别用 `exp`（衰减）和 `cos/sin`（旋转）实现。这让状态转移可以用纯实数 kernel 高效计算，同时保留了复数特征值的振荡能力。

## 为什么 Phase 3 的 RoPE 复数近似失败

Phase 3 的 `rope-complex` 变体用一阶 Taylor 近似替代精确三角函数：

```
cos(θ) ≈ 1 - θ²/2
sin(θ) ≈ θ
```

这等价于 `exp(iθ) ≈ 1 + iθ - θ²/2`。当 `θ` 较大（接近 `π`）时，近似误差显著，导致旋转精度下降，val_loss 从 22.97 升到 26.58。这说明 Mamba3 的旋转角度在实践中可以达到较大值，不能简单线性化。

## ASIC 设计启示

1. **cos/sin 不可省略**：旋转是复数状态转移的核心，需要精确的三角函数。ASIC 上建议用 LUT（查找表）实现，而不是多项式近似。
2. **衰减和旋转可并行**：两部分作用于不同维度，硬件上可以并行计算。
3. **角度累积需要 mod 2π**：递推累积 `θ_t = Σ tanh(angle)·dt·π` 后需要取模，防止数值溢出。ASIC 上需要模运算单元或周期性重置。
4. **共轭对处理**：复特征值成共轭对，硬件上可以用一组 cos/sin 和取反的 sin 同时处理两个共轭分量。
