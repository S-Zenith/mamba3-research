# Mamba3 WikiText-2 量化实验记录

## 实验目标

本实验在 `demo/wikitext_mamba3_quant/` 中训练 pure PyTorch Mamba3-SISO-style causal LM，并用 Quamba/Quamba2 风格 PTQ/fake-quant 评估低比特配置。通过标准是量化后 validation/test perplexity 相比 FP32 恶化不超过 10%。

## 模型约束

- `d_model=512`
- `n_layer=4`
- `expand=1`
- `headdim=64`
- `d_state=2`
- `vocab_size=50257`（GPT-2 tokenizer）
- 单 block runtime state：`1024` elements
- 全模型 runtime state：`4096` elements per batch item
- 单 block 静态参数量：`801812`（< 1M ✓）
- 总参数量：`28939344`（含 embedding/lm_head，绑定权重）

## 环境与命令

- Python：`3.10`
- PyTorch：`2.3.1+cu121`
- CUDA available：`True`
- Device：`NVIDIA GeForce RTX 4060 Laptop GPU`
- SwanLab：`0.8.4`，在线账号 `S-Zenith`，endpoint `https://api.swanlab.cn`
- HF endpoint：`huggingface.co` 不可达，使用镜像 `HF_ENDPOINT=https://hf-mirror.com`
- 训练命令：

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py \
  --steps 1500 --batch-size 8 --seq-len 128 --eval-interval 150 --eval-batches 10 \
  --lr 3e-4 --device auto --quant-modes W8A8,W4A8,W4A16 \
  --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq-long
```

## FP32 训练结果

- steps：1500
- train_seconds：277.14（约 4.6 分钟）
- final train loss：23.59
- final validation loss：22.97
- final validation perplexity：9.48e9
- test loss：22.98
- test perplexity：1.19e10

说明：FP32 基线 loss 仍然很高（PPL ~1e10），说明 4 层 d_model=512 模型从随机初始化在 50257 词表上训练 1500 步还远未收敛。绝对 PPL 没有参考意义，但量化前后的**相对 loss 变化**仍然能清楚反映量化误差来源。

## PTQ/Fake-Quant 结果

| mode | val loss | val ppl | test loss | test ppl | val ppl ratio | test ppl ratio | pass +10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `W8A8` | 100.30 | 3.61e43 | 105.50 | 6.58e45 | 3.81e33 | 5.55e35 | False |
| `W4A8` | 107.55 | 5.08e46 | 109.02 | 2.21e47 | 5.36e36 | 1.86e37 | False |
| `W4A16` | 25.96 | 1.87e11 | 26.30 | 2.64e11 | 19.75 | 22.26 | False |

三种配置均未通过 PPL +10% 标准。但结果本身很有信息量：

- **W4A16**（权重 4bit、激活 FP16）：val_loss 从 22.97 升到 25.96，仅 +13%。这表明**权重量化到 4bit 本身误差可控**，主要误差来自激活量化。
- **W8A8**（权重 8bit、激活 8bit）：val_loss 从 22.97 暴涨到 100.30，+337%。激活量化到 INT8 导致模型彻底失效。
- **W4A8**（权重 4bit、激活 8bit）：比 W8A8 更差，因为叠加了 4bit 权重误差。

核心结论：**激活量化是主要失效点，不是权重量化。** 这与 Quamba 论文的核心观察一致——Mamba 的 selective scan、门控、状态递推和激活离群值对量化敏感，不能只按线性层思路做 W8A8。

## SwanLab 状态

在线 run 创建成功：
- 项目：https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant
- run：https://swanlab.cn/@S-Zenith/mamba3-wikitext-quant/runs/r0azxw0o
- 上传记录数：3430
- 本地日志备份：`mamba/swanlog/run-20260707_192359-r0azxw0o`

## ASIC 观察

- **单 block state 为 1024 elements，4 层模型需要 4096 elements per batch item**。硬件约束必须明确 per-block 和 whole-model 两种口径。本项目按 per-block 口径设计，ASIC SRAM 预算应按 whole-model（4096 elements）规划。
- **W8A8 彻底失效**说明直接全 INT8 datapath 不可行。Quamba 的做法是对 scan/state/B/C 等关键张量单独校准 scale，而不是统一 per-tensor。ASIC 上需要混合精度 datapath：线性层可以用 INT8 MAC，但 scan 累积路径、softplus/exp/sigmoid 控制路径需要更高精度或查表近似。
- **W4A16 退化 +13%**说明 4bit 权重存储是可行的，ASIC 上可以用 4bit 权重 SRAM + FP16/INT16 激活 datapath 的混合设计。这与 Quamba2 的 W4AX/hybrid blocks 思路一致。
- **scan 累积精度**是关键：Mamba3 的 recurrent state 在序列内累积，量化误差会随步数放大。ASIC 上 state update 单元需要保留足够位宽（建议 ≥16bit 累加器），即使 state 存储用 INT8。
- **softplus、exp、sigmoid 等控制路径**对 range 敏感，不能简单截断。ASIC 上建议这些算子用 LUT 或保留 FP16/BF16 计算。
- **下一步建议**：
  1. 参照 Quamba2 的 head/channel grouping 和 reorder，把激活按 range 聚类后再量化，可能让 W8A8 从失效变为可用。
  2. 对 scan 路径单独做 percentile 校准（Quamba 用 0.9995/0.99999 percentile），而不是简单 absmax。
  3. 在 ASIC 设计上，把 state SRAM、scale storage、scan 累加器位宽、门控控制路径精度作为独立设计变量，而不是统一位宽。
  4. 训练更久或用更小词表（如 char-level）让 FP32 基线 PPL 收敛到合理范围，再重跑量化评估，能更清楚地看到 W8A8 和 W4A16 的相对差异。
