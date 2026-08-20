# Mamba3 三方向探索设计

## 目标

在已有的 `demo/wikitext_mamba3_quant/` 基础上，分三阶段探索 Mamba3 的量化性能和 ASIC 友好性：消融+QAT、中间变量误差分析、非线性函数近似。

## Phase 1：消融 + QAT

### 消融配置

5 个 FP32 配置，全部满足单 block runtime state <= 1024 和单 block 静态参数 <= 1M：

| preset | d_model | n_layer | headdim | nheads | d_state | state/block | block_params | steps |
|---|---|---|---|---|---|---|---|---|
| baseline | 512 | 4 | 64 | 8 | 2 | 1024 | ~802K | 1500 |
| deep | 512 | 8 | 64 | 8 | 2 | 1024 | ~802K | 1500 |
| widehead | 512 | 4 | 128 | 4 | 2 | 1024 | ~801K | 1500 |
| bigstate | 256 | 4 | 64 | 4 | 4 | 1024 | ~203K | 1500 |
| longtrain | 512 | 4 | 64 | 8 | 2 | 1024 | ~802K | 5000 |

每个配置用相同数据（WikiText-2 + GPT-2 tokenizer）、相同 batch_size=8、seq_len=128、lr=3e-4 训练。SwanLab run 名为 `ablation-<preset>`。

### QAT

从 val_loss 最低的消融配置的 FP32 checkpoint 出发，做 2 个 QAT：

- `qat-w8a8`：前向插入 W8A8 fake-quant，反向用 straight-through estimator，微调 1000 步
- `qat-w4a16`：前向插入 W4A16 fake-quant（仅权重量化），反向 STE，微调 1000 步

QAT 后在 val/test 上评估，与 PTQ 结果对比。

### 实现方式

在 `train_wikitext_mamba3_quant.py` 中增加：
- `--preset` 参数选择消融配置
- `--mode` 参数支持 `fp32`（默认）和 `qat`
- `--qat-mode` 参数支持 `W8A8` 和 `W4A16`
- `--qat-steps` 参数默认 1000
- QAT 用 `FakeQuantLinear` 但 `requires_grad=True`，权重和 scale 可学习
- STE 通过自定义 autograd Function 或 `x.detach() - x.detach().round() + x` 实现

## Phase 2：中间变量误差分析

### 方法

在 best FP32 模型上增加 instrumentation 模式：
1. 用同一 batch（固定随机种子）跑 FP32 前向和 W8A8/W4A8/W4A16 前向
2. 用 forward hook 收集每个 intermediate tensor
3. 对每个 tensor 计算：abs_mean、abs_max、MSE、max_abs_error、relative_error（MSE / FP32 abs_mean²）

### 收集的中间变量

PureMamba3SISOBlock forward 中的关键节点：
- `in_proj_out`：in_proj 输出
- `z, x, b_raw, c_raw, dd_dt, dd_a, trap, angles`：split 后
- `q, k`：B_norm/C_norm 后
- `q_rope, k_rope`：RoPE 后
- `a`：softplus(dd_a) 后
- `dt`：softplus(dd_dt + bias) 后
- `adt`：a * dt 后
- `decay`：exp(cumsum) 后
- `scale`：dt * sigmoid(trap) 后
- `weights`：qk * decay * scale 后
- `mixed`：einsum(history, v) + diag + D*v 后
- `gate`：silu(z) 后
- `y`：cat(mixed * gate) 后
- `out_proj_out`：out_proj 输出

### 输出

`outputs/intermediate_error_analysis.json` + `docs/mamba3_exploration_report.md` 中的分析章节。

## Phase 3：非线性函数近似

### 4 个结构变体

在 best FP32 配置上做 4 个变体，每个从头训练 1500 步：

1. **silu-relu**：gate 的 `F.silu(z)` → `F.relu(z)`
2. **softplus-relu**：A 的 `F.softplus(dd_a)` → `F.relu(dd_a).clamp(min=1e-4)`，dt 的 `F.softplus(dd_dt + bias)` → `F.relu(dd_dt + bias).clamp(min=1e-4)`
3. **rope-complex**：去掉 `cos/sin`，用一阶近似 `cos≈1-θ²/2, sin≈θ` 做复数乘法
4. **exp-poly**：decay 的 `torch.exp(x)` → 分段近似：`|x|<1` 用 `1+x+x²/2`，`x>=1` 用 `exp(1)*(1+(x-1))`，`x<-1` 用 `exp(-1)*(1-(x+1))`，更远的 clamp

### 评估

每个变体训练后做 FP32 + W8A8 PTQ 评估，与 baseline 对比。SwanLab run 名为 `nonlinear-<variant>`。

### 实现方式

在 `Mamba3LMConfig` 中增加 `activation`、`rope_mode`、`exp_mode` 字段，`PureMamba3SISOBlock` 根据这些字段选择非线性函数实现。

## SwanLab

所有 run 记录到同一项目 `mamba3-wikitext-quant`。Phase 1 的 7 个 run、Phase 3 的 4 个 run 共 11 个 run，每个 run 记录 train/loss、val/loss、val/ppl、test/loss、test/ppl。

## 文档输出

`docs/mamba3_exploration_report.md` 包含：
- Phase 1 消融对比表和 QAT 结果
- Phase 2 中间变量误差分析表和关键发现
- Phase 3 非线性近似对比表和 ASIC 建议
- 综合结论：哪种配置+量化+非线性组合最适合 ASIC

## 验证

- 每次代码改动后运行 `pytest tests/test_wikitext_mamba3_quant.py -q` 确保不破坏现有测试
- 每个训练 run 完成后检查 `metrics.json` 和 SwanLab 在线记录
- Phase 2 分析必须基于实际 forward hook 数据，不是推测

## 非目标

- 不实现真实 INT8/INT4 CUDA kernel
- 不做 ASIC RTL/HLS 实现
- 不改变 WikiText-2 数据集或 tokenizer
- Phase 3 的非线性近似只做一阶近似，不做 exhaustive 搜索
