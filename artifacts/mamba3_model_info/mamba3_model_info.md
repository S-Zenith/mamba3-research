# Mamba3 Demo 模型信息

## 说明

这个文件来自源码实例化的 Mamba3 block，不是论文训练后的预训练模型。
保存的 `.pt` 文件是随机初始化参数，只用于查看结构和参数形状。

## 配置

```text
d_model = 256
d_state = 64
expand = 2
headdim = 64
chunk_size = 64
d_inner = 512
nheads = 8
num_bc_heads = 1
mimo_rank = 1
num_rope_angles = 16
```

## 参数形状

| name | shape | dtype | numel |
|---|---:|---|---:|
| `dt_bias` | `(8,)` | `torch.float32` | 8 |
| `B_bias` | `(8, 1, 64)` | `torch.float32` | 512 |
| `C_bias` | `(8, 1, 64)` | `torch.float32` | 512 |
| `D` | `(8,)` | `torch.float32` | 8 |
| `in_proj.weight` | `(1192, 256)` | `torch.float32` | 305152 |
| `B_norm.weight` | `(64,)` | `torch.float32` | 64 |
| `C_norm.weight` | `(64,)` | `torch.float32` | 64 |
| `out_proj.weight` | `(256, 512)` | `torch.float32` | 131072 |

## 参数总量

```text
437,392 parameters
```

## 输出文件

- `artifacts/mamba3_model_info/mamba3_demo_state_dict.pt`