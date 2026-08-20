# WikiText-2 Mamba3 量化实验设计

## 目标

在 `demo/` 下新增一个独立 Mamba3 语言建模实验，用 WikiText-2 训练一个接近 ASIC 约束上限的小型 Mamba3-SISO-style 模型，并基于 Quamba/Quamba2 的后训练量化思路评估 Mamba3 是否可以量化到 8bit 整型或更低，同时保持验证和测试困惑度相对 FP32 基线恶化不超过 10%。

本工作还需要在 `mamba/docs` 中沉淀两类文档：Quamba/Quamba2 低比特量化思路总结，以及本次 Mamba3 实验的实际做法、指标和面向 ASIC 优化的观察。

## 范围

新增目录：`demo/wikitext_mamba3_quant/`。

核心产物：

- `train_wikitext_mamba3_quant.py`：训练、校准、PTQ/fake-quant 评估、SwanLab 在线记录的主脚本。
- `README.md`：运行方式、依赖、模型约束和输出说明。
- `outputs/`：训练日志、指标 JSON、checkpoint、曲线图和量化报告。
- `docs/quamba_low_bit_quantization_notes.md`：Quamba/Quamba2 思路总结。
- `docs/mamba3_wikitext_quant_experiment.md`：本次实验记录和 ASIC 观察。

当前 `mamba` 目录不是 git 仓库，因此设计文档和后续代码变更无法在该目录内提交 git commit。

## 模型设计

新模型继续使用本仓库现有 pure PyTorch Mamba3-SISO-style block 的结构，而不是调用官方 fused kernel。配置固定为：

```text
d_model = 512
n_layer = 4
expand = 1
headdim = 64
d_state = 2
chunk_size = 64
dropout = 0.05
```

约束口径按用户确认的 ASIC SRAM 相关运行状态大小执行，且该 `runtime_state` 是单个 block 自有 recurrent state，不是所有 block 共享的全模型 state：

```text
runtime_state_per_block = nheads * headdim * d_state
                        = (d_model * expand / headdim) * headdim * d_state
                        = 512 * 1 * 2
                        = 1024
```

4 层模型在自回归推理时的 recurrent state 总量为 `4 * 1024 = 4096` elements per batch item。单个 block 的静态参数量预计约 0.8M，低于 1M 且接近约束上限。全模型参数量会明显大于 block 参数量，因为使用 GPT-2 tokenizer 词表并绑定 embedding 与 LM head；该部分不计入单 block 静态参数约束。

## 数据与训练

数据集使用 Hugging Face `datasets` 加载 `wikitext-2-raw-v1`。Tokenizer 使用 `transformers` 的 `gpt2` tokenizer，按 causal language modeling 方式切分固定长度序列。

训练只训练 FP32 基线 checkpoint。脚本记录：

- train loss 曲线。
- validation loss 和 validation perplexity。
- test loss 和 test perplexity。
- 总参数量、单 block 静态参数量、运行 state 大小。
- 训练命令、随机种子、设备和依赖状态。

训练默认配置应能在当前 RTX 4060 Laptop GPU 上完成小规模可复现实验；步数、batch size、sequence length 和 eval interval 都通过命令行参数暴露，便于必要时缩短或加长实验。

## 量化设计

量化阶段采用 PTQ 评估，不做 QAT。流程参考 Quamba/Quamba2：

1. 用校准 batch 收集 activation、scan 中间值和 recurrent state 的范围。
2. 对 linear 权重使用对称量化；W8 使用 per-output-channel scale，W4 使用沿输入维的 group-wise scale。
3. 对激活和关键 scan 中间值插入 per-tensor symmetric fake quant/dequant 边界，模拟低精度存储和算子输出。
4. 对 recurrent state 单独配置量化策略，优先评估 INT8 state；W4A16 作为更稳定的低权重量化对照。
5. 在 validation 和 test split 上评估 FP32、W8A8、W4A8、W4A16。

通过标准：量化配置的 validation/test perplexity 相比 FP32 基线恶化不超过 10%。如果 W4A8 不满足标准，报告需明确指出失效位置和可能原因，而不是只给最终指标。

## SwanLab

在 `mamba/.venv` 中安装并配置 SwanLab，按用户要求使用在线记录。脚本需要将 FP32 训练曲线、验证指标、测试指标、量化配置和最终量化指标写入同一 SwanLab project/run。

如果当前机器没有 SwanLab 登录态或缺少 token，实施时脚本应给出明确错误或提示，不把本地离线日志伪装成在线结果。

## 文档输出

`docs/quamba_low_bit_quantization_notes.md` 应总结：

- Quamba 的 Mamba PTQ 核心问题：selective scan、门控、状态递推和激活离群值会放大量化误差。
- Quamba 的主要做法：校准激活范围、Hadamard/rotation 平滑、Norm/Linear 融合、W8A8/W4A8/W4A16、状态和 scan 路径专门量化。
- Quamba2 的扩展：Mamba2、大模型、多精度配置、head/channel grouping、reorder、hybrid W4AX、GPTQ 和真实部署 profiling。
- 对 Mamba3/ASIC 的启发：state SRAM、scale 粒度、scan 累积精度、门控/指数/softplus 控制路径不能简单全低比特。

`docs/mamba3_wikitext_quant_experiment.md` 应记录：

- 实际环境、依赖安装和 SwanLab 状态。
- 模型配置与约束检查。
- 训练数据、命令、loss/perplexity。
- PTQ 校准和各 bit-width 指标。
- 是否达到 PPL +10% 标准。
- 面向 ASIC 的观察和下一步建议。

## 验证

实施完成前需要运行：

- 脚本的快速 smoke test，确认数据管线、模型 forward、loss 和约束检查可运行。
- 至少一次 FP32 训练和 validation/test 评估。
- 至少完成 W8A8 和 W4A8 的 PTQ/fake-quant 评估；W4A16 作为低权重、高激活精度对照，若因运行时间或依赖失败必须在报告中说明。
- SwanLab 在线 run 创建和指标上报检查。

若某项因为网络、登录态、依赖或运行时间失败，最终实验报告必须写明失败原因和已完成的替代验证。

## 非目标

本阶段不实现真实 INT8/INT4 CUDA kernel，不改 Quamba 源码，不复现 Quamba 论文级大模型结果，不做 ASIC RTL 或 HLS 实现，不要求低比特结果 bitwise 等价真实硬件。
