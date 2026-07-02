# Mamba End-to-End Task Model Research

本目录整理官方 Mamba 1/2/3 和相关变体是否提供完整预训练模型、端到端任务脚本，以及我们如何用当前已复现的 PyTorch Mamba3 SISO/MIMO block 搭一个可训练的端到端模型。

## 结论摘要

1. 官方 `state-spaces/mamba` 源码库提供完整语言模型封装、Hugging Face 预训练权重和生成/评测脚本，覆盖 Mamba-1 和 Mamba-2；Mamba-3 目前官方 README 给出 block 级接口，但未看到同等规模的 Mamba-3 完整预训练 LM 权重列表。
2. 官方端到端方向最清晰的是 causal language modeling：`Embedding -> N x Mixer Block -> final norm -> tied LM head`，并可用 `lm-evaluation-harness` 做 zero-shot 任务评测。
3. 第三方变体里，视觉/视频任务更完整：VMamba、Vim、VideoMamba 都提供完整训练/评估脚本和权重，参数量从 7M 到 122M 以上不等。
4. 若我们后续要训练，最适合从“小型语言模型”或“小型序列分类/时间序列预测”开始，因为我们的 PyTorch Mamba3 block 已经是 1D 序列 block，迁移到 LM/序列任务最直接。
5. 建议先用 SISO PyTorch Mamba3 block 搭 2M-15M 参数级模型验证训练闭环，再扩展到 MIMO。MIMO 复现更接近 Mamba-3 官方配置，但计算图和中间张量更多，训练显存/速度压力更大。

## 官方 Mamba 仓库调研

来源：`https://raw.githubusercontent.com/state-spaces/mamba/main/README.md` 和本地安装包源码。

### Mamba-1 / Mamba-2

官方提供：

- block 实现：`mamba_ssm.modules.mamba_simple.Mamba`、`mamba_ssm.modules.mamba2.Mamba2`
- 完整 LM 实现：`mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel`
- Hugging Face 预训练权重：`state-spaces/mamba-*`、`state-spaces/mamba2-*`
- generation benchmark：`benchmarks/benchmark_generation_mamba_simple.py`
- zero-shot 评测：通过 `lm-evaluation-harness` 的 `mamba_ssm` backend

官方 README 列出的预训练模型规模：

| 系列 | 模型 | 层数 | d_model | 训练数据 | 说明 |
|---|---:|---:|---:|---|---|
| Mamba-1 | 130M | 24 | 768 | Pile 300B tokens | base LM |
| Mamba-1 | 370M | 48 | 1024 | Pile 300B tokens | base LM |
| Mamba-1 | 790M | 48 | 1536 | Pile 300B tokens | base LM |
| Mamba-1 | 1.4B | 48 | 2048 | Pile 300B tokens | base LM |
| Mamba-1 | 2.8B | 64 | 2560 | Pile 300B tokens | base LM |
| Mamba-2 | 130M | 24 | 768 | Pile 300B tokens | base LM |
| Mamba-2 | 370M | 48 | 1024 | Pile 300B tokens | base LM |
| Mamba-2 | 780M | 48 | 1536 | Pile 300B tokens | base LM |
| Mamba-2 | 1.3B | 48 | 2048 | Pile 300B tokens | base LM |
| Mamba-2 | 2.7B | 64 | 2560 | Pile 300B tokens | base LM |

官方说明：Mamba 的 block 数通常是同等 Transformer 层数的 2 倍，因为它用两个 Mamba block 对应 Transformer 的 attention block + MLP block。

### Mamba-3

官方 README 给出 Mamba-3 block 用法：

```python
Mamba3(
    d_model=768,
    d_state=128,
    headdim=64,
    is_mimo=True,
    mimo_rank=4,
    chunk_size=16,
    dtype=torch.bfloat16,
)
```

官方说明该 block 约使用 `6 * d_model^2` 参数。以 `d_model=768` 估算，每层约 3.5M 参数，24 层约 85M block 参数，加词嵌入和 LM head 后会接近 120M-160M 量级，适合作为 Mamba-1/2 130M 的 Mamba-3 对应起点。

目前调研结果：官方 README 未列出 Mamba-3 的完整 Hugging Face 预训练 LM 权重。Mamba-3 更像是 block/kernel 已合入主仓库，但完整训练模型生态尚不如 Mamba-1/2 明确。

## 本地官方模型连接方式

本地源码关键路径：

- `.venv/lib/python3.10/site-packages/mamba_ssm/models/mixer_seq_simple.py`
- `.venv/lib/python3.10/site-packages/mamba_ssm/models/config_mamba.py`
- `.venv/lib/python3.10/site-packages/mamba_ssm/modules/block.py`

官方完整 LM 结构：

```text
input_ids
  -> token embedding
  -> for each layer:
       residual = hidden + residual
       hidden = Norm(residual)
       hidden = Mamba/Mamba2 mixer(hidden)
       optional MLP path, if d_intermediate > 0
  -> final norm
  -> tied LM head
  -> logits [B, S, vocab]
```

默认 `MambaConfig`：

```text
d_model = 2560
d_intermediate = 0
n_layer = 64
vocab_size = 50277
rms_norm = True
residual_in_fp32 = True
fused_add_norm = True
tie_embeddings = True
```

对于我们自己的 PyTorch Mamba3 复现，建议采用同样的 pre-norm residual stacking，但禁用 fused add norm，避免依赖官方 kernel：

```text
hidden = embedding(input_ids)
for block in blocks:
    residual = hidden
    hidden = rms_norm(hidden)
    hidden = pytorch_mamba3_block(hidden)
    hidden = hidden + residual
hidden = rms_norm(hidden)
logits = tied_lm_head(hidden)
```

## 第三方端到端 Mamba 变体

### VMamba

仓库：`https://github.com/MzeroMiko/VMamba`

任务覆盖：

- ImageNet-1K 分类
- COCO detection，Mask R-CNN/FPN
- ADE20K segmentation，UperNet

结构：

- 视觉 backbone
- patch embedding / stage hierarchy
- 多个 VSS block
- SS2D 通过四个扫描方向把 2D 图像转换为选择性扫描序列

参数规模和任务：

| 模型 | 分类参数 | 检测参数 | 分割参数 | 典型任务 |
|---|---:|---:|---:|---|
| VMamba-T | 30M | 50M | 62M | ImageNet / COCO / ADE20K |
| VMamba-S | 50M | 70M | 82M | ImageNet / COCO / ADE20K |
| VMamba-B | 89M | 108M | 122M | ImageNet / COCO / ADE20K |

可借鉴点：stage-wise backbone、patch merging、多尺度任务头。若我们用 1D Mamba3 block，可先把图像展平成 patch 序列；若要复刻 VMamba 的核心优势，需要实现 2D 多方向 scan，不是当前 PyTorch Mamba3 block 的最小改动。

### Vim / Vision Mamba

仓库：`https://github.com/hustvl/Vim`

任务覆盖：

- ImageNet 分类训练/评估
- COCO detection / ADE20K segmentation 作为 backbone 扩展

结构：

- ViT 风格 patch embedding
- 加绝对位置 embedding
- class token 可放中间
- bidirectional Mamba block
- mean/final pooling 分类头

参数规模：

| 模型 | 参数 | ImageNet top-1 |
|---|---:|---:|
| Vim-tiny | 7M | 76.1 / 78.3 |
| Vim-small | 26M | 80.5 / 81.6 |
| Vim-base | 98M | 81.9 |

可借鉴点：这是最适合我们快速仿真的视觉结构。我们可以直接用 patch embedding + position embedding + Mamba3 block stack + pooling/head，不必先实现复杂的 2D scan。

### VideoMamba

仓库：`https://github.com/OpenGVLab/VideoMamba`

任务覆盖：

- image classification
- short-term video understanding
- long-term video understanding
- masked video modeling
- video-text retrieval

结构：

- image/video patch embedding
- 时空 token 序列
- Mamba/Vim-style sequence backbone
- 分类、检索或 masked modeling head

可借鉴点：如果后续任务是长序列/视频，Mamba 的线性复杂度很有价值。但训练和数据管线复杂度明显高于语言 toy LM 或图像 patch 分类。

## 对我们最有用的三个端到端任务方向

### 方向 A：小型 Causal LM，最推荐第一步

目标：验证我们 PyTorch Mamba3 block 能参与端到端训练。

结构：

```text
input_ids [B,S]
  -> token embedding [B,S,D]
  -> N x PyTorchMamba3Block
  -> final RMSNorm
  -> tied LM head [B,S,V]
  -> next-token cross entropy
```

建议配置：

| 级别 | vocab | d_model | n_layer | d_state | headdim | expand | 估计参数 | 用途 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tiny | 256 | 128 | 4 | 32 | 32 | 2 | ~0.5M | 字符级 LM / 合成数据 |
| small | 4096 | 256 | 8 | 64 | 64 | 2 | ~6M-8M | byte/BPE toy LM |
| medium | 8192 | 512 | 12 | 64/128 | 64 | 2 | ~35M-45M | 小语料 LM |

为什么推荐：数据最简单，损失函数明确，不需要图像增强或检测/分割框架。官方 Mamba LM 也是这个结构，便于对齐。

### 方向 B：Patch 序列图像分类，借鉴 Vim

结构：

```text
image [B,3,H,W]
  -> Conv2d patch embedding, patch_size=16
  -> flatten to [B,N,D]
  -> optional class token / position embedding
  -> N x PyTorchMamba3Block
  -> mean pool or cls token
  -> Linear classifier
```

建议配置：

| 级别 | image | patch | d_model | n_layer | 参数 | 任务 |
|---|---:|---:|---:|---:|---:|---|
| toy | 32x32 | 4 | 128 | 4 | ~0.6M | CIFAR-10 smoke |
| tiny | 224x224 | 16 | 192 | 12 | ~4M-8M | ImageNet subset |
| small | 224x224 | 16 | 384 | 24 | ~20M-35M | 完整 ImageNet 试验 |

注意：这不是完整 Vim，因为没有 bidirectional scan 和 class-token-middle trick；但足够验证 Mamba3 block stacking 和训练闭环。

### 方向 C：时间序列预测/分类，工程上最轻

结构：

```text
continuous sequence [B,S,C]
  -> Linear(C,D)
  -> N x PyTorchMamba3Block
  -> last token / mean pool
  -> regression or classification head
```

建议配置：

| 级别 | d_model | n_layer | seq_len | 参数 | 任务 |
|---|---:|---:|---:|---:|---|
| tiny | 64 | 2 | 128-512 | <0.2M | 合成 copy/add/forecast |
| small | 128 | 4 | 512-2048 | ~0.5M | UCR/ETT 子集 |
| medium | 256 | 8 | 2048+ | ~5M | 长序列预测 |

## 用当前 PyTorch Mamba3 block 的实现建议

当前可复用代码：

- SISO 复现：`artifacts/mamba3_siso_pytorch_repro/`
- MIMO 复现：`artifacts/mamba3_mimo_pytorch_repro/`

建议优先抽象一个训练可用 block：

```text
class PyTorchMamba3Block(nn.Module):
    in_proj
    pure_pytorch_mamba3_kernel
    out_proj
```

然后外层做残差和归一化：

```text
class Mamba3ResidualBlock(nn.Module):
    norm = RMSNorm(d_model)
    mixer = PyTorchMamba3Block(...)
    forward(x): return x + mixer(norm(x))
```

SISO vs MIMO 选择：

- SISO：最适合先训练，参数少、实现简单、没有 rank mixing，debug 更容易。
- MIMO：更贴近 Mamba-3 README 推荐配置，block 参数更多，表达力更强，但训练成本和 shape 复杂度更高。

## 初始实现路线建议

1. 把 SISO 或 MIMO 复现脚本中的 block 逻辑抽成 importable `nn.Module`，不依赖官方 Mamba3 forward。
2. 先实现 `TinyMamba3LM`：字符级 LM，`vocab=256,d_model=128,n_layer=4,d_state=32,headdim=32,seq_len=128`。
3. 用随机/小文本跑 forward + cross entropy + backward，确认梯度通路完整。
4. 再增加 patch 分类模型，复用同一 stack。
5. 最后再考虑 MIMO 和更大的配置。

## 风险和限制

- 当前 PyTorch Mamba3 复现为 correctness/debug 取向，不是高性能训练 kernel；多层训练会明显慢于官方 fused kernel。
- MIMO 中间张量多，显存压力大；建议用小 batch、小 seq_len 起步。
- 官方 Mamba-3 还没有像 Mamba-1/2 那样明确列出完整预训练模型生态；若目标是对齐预训练 LM，Mamba-1/2 资料更完整。
- 视觉端到端模型资料丰富，但很多性能来自 2D scan、stage hierarchy、训练 recipe，不只是替换一个 block。

## 推荐下一步

下一步建议新建 `artifacts/mamba3_tiny_lm_pytorch/`：

- 抽象 `PyTorchMamba3SISOBlock`
- 搭 `TinyMamba3LM`
- 实现 synthetic next-token 或 tiny text dataloader
- 跑 1-5 step training smoke test
- 统计参数量和显存

这样能最快回答“我们复现的 PyTorch Mamba3 是否真的可训练”。
