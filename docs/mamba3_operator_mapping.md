# Mamba3 算子、源码步骤和计算量对照

## 这份文档说明什么

这份文档把当前 benchmark 中 Mamba3 的 profiler 结果和 `mamba-ssm` 源码对应起来，回答三个问题：

- 为什么 `aten::linear`、`aten::matmul`、`aten::mm` 占了较多时间。
- 这些矩阵乘法分别对应 Mamba3 算法中的哪些步骤。
- Mamba3 中状态更新相关步骤对应哪些 profiler 算子。

当前结论基于本项目的 quick 配置：

```text
batch = 1
seq_len = 128
d_model = 256
expand = 2
d_inner = 512
headdim = 64
nheads = 8
d_state = 64
num_bc_heads = 1
mimo_rank = 1
num_rope_angles = 16
dtype = torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
torch.get_float32_matmul_precision() = highest
```

所以当前矩阵乘法按 FP32 语义执行，并且没有启用 TF32。

## 源码位置

Mamba3 forward 的源码在虚拟环境中：

```text
.venv/lib/python3.10/site-packages/mamba_ssm/modules/mamba3.py
```

关键 forward 结构如下：

```python
def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
    # 1. 输入投影
    zxBCdtAtrap = self.in_proj(u)

    # 2. 拆分投影结果
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(...)

    # 3. 形状变换
    z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    # 4. 计算状态更新需要的 A、DT、ADT
    _A = -F.softplus(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-self.A_floor)
    DT = F.softplus(dd_dt + self.dt_bias)
    ADT = _A * DT
    DT = rearrange(DT, "b l n -> b n l")
    ADT = rearrange(ADT, "b l n -> b n l")

    # 5. 准备角度参数
    angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

    # 6. B/C 归一化
    B = self.B_norm(B)
    C = self.C_norm(C)

    # 7. Mamba3 状态空间核心 kernel
    y = mamba3_siso_combined(...)

    # 8. 输出投影
    out = self.out_proj(y.to(x.dtype))
    return out
```

## 为什么矩阵乘法占比较高

当前 Mamba3 profile 中，矩阵相关算子聚合结果大致如下：

```text
operator       calls  cpu_time_total_us
aten::linear       2              3389
aten::matmul       2              3349
aten::mm           2              3281
```

这 2 次 `aten::linear` 分别对应源码中的：

- `self.in_proj(u)`
- `self.out_proj(y.to(x.dtype))`

`aten::linear` 是 PyTorch 的线性层接口，内部会触发矩阵乘法。profile 中同时看到 `aten::linear`、`aten::matmul`、`aten::mm`，不是三组完全独立的矩阵乘法，而是同一条调用链的不同层级：

- `aten::linear`：高层线性层接口。
- `aten::matmul`：通用矩阵乘法接口。
- `aten::mm`：二维矩阵乘法底层算子。

因此看 breakdown 时，不要把这三个时间简单相加当作三倍计算量。它们有调用层级关系。

## 第一次矩阵乘法：输入投影 in_proj

源码：

```python
zxBCdtAtrap = self.in_proj(u)
```

作用：

把输入 `u` 从模型维度 `d_model=256` 一次性投影出 Mamba3 后续需要的多组内部变量：

```text
z, x, B, C, dd_dt, dd_A, trap, angles
```

输入形状：

```text
u: [batch, seq_len, d_model]
 = [1, 128, 256]
```

权重形状：

```text
in_proj.weight: [1192, 256]
```

为什么输出维度是 1192：

```text
d_in_proj =
2 * d_inner
+ 2 * d_state * num_bc_heads * mimo_rank
+ 3 * nheads
+ num_rope_angles

= 2 * 512
+ 2 * 64 * 1 * 1
+ 3 * 8
+ 16

= 1024 + 128 + 24 + 16
= 1192
```

实际矩阵乘法可以理解成：

```text
[128, 256] @ [256, 1192] -> [128, 1192]
```

乘加规模：

```text
M = 128
K = 256
N = 1192

multiply-add pairs = M * K * N
                   = 128 * 256 * 1192
                   = 39,059,456

如果按 1 次乘法 + 1 次加法 = 2 FLOPs：
FLOPs ≈ 2 * 39,059,456
      = 78,118,912 FLOPs
      ≈ 78.1 MFLOPs
```

这也是为什么输入投影在当前 small demo 中显得很重：它不是只投影到一个中间张量，而是一次性生成多个后续状态空间计算需要的变量。

## 第二次矩阵乘法：输出投影 out_proj

源码：

```python
out = self.out_proj(y.to(x.dtype))
```

作用：

把 Mamba3 核心 kernel 的内部输出从 `d_inner=512` 投影回模型维度 `d_model=256`。

输入形状：

```text
y: [batch, seq_len, d_inner]
 = [1, 128, 512]
```

权重形状：

```text
out_proj.weight: [256, 512]
```

实际矩阵乘法可以理解成：

```text
[128, 512] @ [512, 256] -> [128, 256]
```

乘加规模：

```text
M = 128
K = 512
N = 256

multiply-add pairs = M * K * N
                   = 128 * 512 * 256
                   = 16,777,216

FLOPs ≈ 2 * 16,777,216
      = 33,554,432 FLOPs
      ≈ 33.6 MFLOPs
```

## 两次投影的总计算量

```text
in_proj  ≈ 78.1 MFLOPs
out_proj ≈ 33.6 MFLOPs

合计 ≈ 111.7 MFLOPs
```

这解释了为什么 `aten::linear`、`aten::matmul`、`aten::mm` 在当前 profile 中很显眼。

## 状态更新相关步骤对应哪些算子

Mamba3 的状态更新不是主要通过 `aten::mm` 表现出来，而是由一系列参数准备算子和一个自定义核心 kernel 组成。

### 1. 构造状态衰减参数 A

源码：

```python
_A = -F.softplus(dd_A.to(torch.float32))
_A = torch.clamp(_A, max=-self.A_floor)
```

profile 里对应：

```text
aten::to
aten::_to_copy
aten::copy_
aten::softplus
aten::neg
aten::clamp
```

含义：

- `dd_A.to(torch.float32)`：转成 float32，满足后续数值计算要求。
- `softplus`：把数值平滑地变成正数。
- `neg`：取负，构造负的状态衰减参数。
- `clamp`：限制数值范围，避免状态更新不稳定。

### 2. 构造时间步长 DT

源码：

```python
DT = F.softplus(dd_dt + self.dt_bias)
```

profile 里对应：

```text
aten::add
aten::softplus
```

含义：

- `add`：加上可学习的 `dt_bias`。
- `softplus`：保证时间步长 `DT` 为正。

### 3. 构造 ADT

源码：

```python
ADT = _A * DT
```

profile 里对应：

```text
aten::mul
```

含义：

把状态衰减参数 `_A` 和时间步长 `DT` 组合起来，得到进入状态扫描 kernel 的离散化更新因子。

### 4. 调整 DT 和 ADT 的维度布局

源码：

```python
DT = rearrange(DT, "b l n -> b n l")
ADT = rearrange(ADT, "b l n -> b n l")
```

profile 里对应：

```text
aten::permute
aten::as_strided
aten::reshape
aten::_reshape_alias
```

含义：

调整张量维度顺序，让后续 Mamba3 kernel 按需要的布局访问数据。

### 5. 准备角度参数 angles

源码：

```python
angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
```

profile 里对应：

```text
aten::unsqueeze
aten::expand
aten::to
aten::_to_copy
aten::copy_
```

含义：

- `unsqueeze`：增加一个维度。
- `expand`：扩展到每个 head。
- `to(float32)`：转换成 float32，因为 Mamba3 SISO/MIMO kernel 需要 float32 角度参数。

### 6. B 和 C 的归一化

源码：

```python
B = self.B_norm(B)
C = self.C_norm(C)
```

profile 里对应：

```text
LayerNormFn
aten::empty_like
aten::empty_strided
aten::empty
```

含义：

对状态空间里的 B/C 参数做 RMSNorm/RMSNormGated，稳定进入状态更新 kernel 的 Q/K 参数。

### 7. 真正的 Mamba3 状态空间扫描/更新

当前使用 SISO 路径，因为构造时 `is_mimo=False`。

源码：

```python
y = mamba3_siso_combined(
    Q=C.squeeze(2),
    K=B.squeeze(2),
    V=x,
    ADT=ADT,
    DT=DT,
    Trap=trap,
    Q_bias=self.C_bias.squeeze(1),
    K_bias=self.B_bias.squeeze(1),
    Angles=angles,
    D=self.D,
    Z=z if not self.is_outproj_norm else None,
    chunk_size=self.chunk_size,
    Input_States=None,
    return_final_states=ssm_state is not None,
    cu_seqlens=cu_seqlens,
)
```

profile 里对应：

```text
_Mamba3Function
cudaLaunchKernel
```

含义：

这是 Mamba3 的核心状态空间递推/扫描计算。它接收：

```text
Q = C.squeeze(2)
K = B.squeeze(2)
V = x
ADT
DT
Trap = trap
Angles = angles
D = self.D
Z = z
```

由于这是 Triton/CUDA 自定义 kernel，PyTorch profiler 不会把它内部再拆成普通 `aten::*` 算子。当前 profiler 只能看到 `_Mamba3Function` 和它触发的 `cudaLaunchKernel`。

## 为什么不能直接说矩阵乘法真实 GPU 时间一定大于状态更新

当前报告的运行时间来源是：

```text
runtime_source = cpu_fallback
```

这表示 PyTorch profiler 没有给出有效的 CUDA kernel 时间，所以报告使用 CPU profiler event 时间做近似 breakdown。

因此可以确认：

- CPU profiler 视角下，`in_proj/out_proj` 的调用跨度较大。
- 两次投影确实有约 111.7 MFLOPs 的计算量。
- `_Mamba3Function` 是 Mamba3 状态更新核心。

但不能严格推出：

- 真实 GPU kernel 时间里矩阵乘法一定比 Mamba3 状态更新更慢。
- Mamba3 自定义 kernel 内部每个子步骤的真实 CUDA 时间占比。

要得到真实 GPU kernel timeline，需要 Nsight Systems 或 Nsight Compute。

## 一句话总结

Mamba3 中 `aten::linear/matmul/mm` 主要来自输入投影和输出投影：

```text
self.in_proj(u)        ≈ 78.1 MFLOPs
self.out_proj(y)       ≈ 33.6 MFLOPs
合计                   ≈ 111.7 MFLOPs
```

Mamba3 的状态更新准备步骤对应 `softplus/add/mul/to/reshape/LayerNormFn` 等算子；真正的状态空间扫描/更新核心对应 `_Mamba3Function` 和它触发的 `cudaLaunchKernel`。
