# Mamba3 MIMO PyTorch Block 复现设计

## 目标

在 `artifacts/mamba3_mimo_pytorch_repro/` 中实现一个脚本级 Mamba3 MIMO block 前向复现。脚本需要用官方 MIMO 路径生成 reference，用纯 PyTorch 按源码语义复现主要计算流程，并保存中间结果、最终输出和数值对照报告。

目标不是性能复现，而是验证算法流程、tensor 布局、rank mixing、chunk/state 递推和输出聚合是否与官方 kernel 语义一致。

## 范围

本阶段处理固定 demo 配置：`d_model=256`、`d_state=64`、`expand=2`、`headdim=64`、`chunk_size=64`、`is_mimo=True`。输入使用固定随机种子生成，默认形状为 `batch=1`、`seq_len=128`、`dtype=torch.float32`。

脚本会构造官方 `Mamba3` block 并使用可运行的官方 MIMO kernel 作为 reference。PyTorch 复现不调用官方 MIMO fused forward kernel，只复用官方模块权重和公开的基础函数语义。

## 产物

新增目录：`artifacts/mamba3_mimo_pytorch_repro/`。

核心文件：

- `mamba3_mimo_pytorch_repro.py`：可直接运行的 MIMO 复现和比对脚本。
- `intermediates.pt`：保存输入、配置、官方输出、PyTorch 输出和关键中间结果。
- `comparison_report.md`：保存误差统计和实现说明。
- `README.md`：说明运行方式，以及 SISO 和 MIMO 在 PyTorch 复现时的主要差异。

## 官方 Reference 路径

脚本会实例化 `Mamba3(is_mimo=True)`，使用固定随机输入执行官方 forward，得到最终输出 reference。若当前环境缺少可用 GPU、TileLang 或官方 MIMO kernel 运行失败，脚本会抛出明确错误，不伪造结果。

## PyTorch 复现路径

PyTorch 路径会按官方 MIMO forward 的结构展开：

1. 执行 `in_proj(u)`。
2. 按 MIMO 配置拆分 `z`、`x`、`B`、`C`、`dd_dt`、`dd_A`、`trap`、`angles`，以及 MIMO rank/head 相关权重。
3. 将 `x/z` reshape 为 `[B, S, H, P]`，将 `B/C` reshape 为 `[B, S, R, G, N]`。
4. 计算 `A = -softplus(dd_A)`、`DT = softplus(dd_dt + dt_bias)`、`ADT = A * DT`，并转为 `[B, H, S]` 布局。
5. 复现 angle cumsum 和 RoPE，对带 bias 的 `Q/K` 生效。
6. 计算 trap scale：`gamma = DT[t] * sigmoid(trap[t])`，`shifted_gamma = DT[t+1] * sigmoid(-trap[t+1])`，`scale = gamma + shifted_gamma`。
7. 按 chunk 复现状态递推：跨 chunk state contribution、chunk 内 causal contribution、chunk 尾 state 更新。
8. 使用 `MIMO_V` 将 `V` 扩展到 rank 维，并在输出端用 `MIMO_Out` 聚合 rank 维。
9. 加入 `D` skip、`Z` 和 `MIMO_Z` gate，最后执行 `out_proj` 得到 block 输出。

## 中间结果

脚本至少保存以下中间结果：

- 输入和配置：`u`、seed、shape 配置。
- 投影和拆分：`in_proj_out`、`z_flat`、`x_flat`、`B_flat`、`C_flat`、`dd_dt`、`dd_A`、`trap_flat`、`angles_flat`。
- kernel 输入：`Q_pre_bias`、`K_pre_bias`、`Q_biased`、`K_biased`、`V`、`Z`、`MIMO_V`、`MIMO_Out`、`MIMO_Z`。
- 离散化：`A`、`DT`、`ADT`、`angles_cumsum`、`gamma`、`shifted_gamma`、`trap_scale`。
- MIMO 过程：`psi_v`、`q_rot`、`k_rot`、chunk 内 `qk`、chunk 前后 state、pre-gate 输出。
- 输出：`torch_y`、`torch_out`、`official_out`。

## 数值比对

报告会比较最终 `torch_out` 和 `official_out` 的 `max_abs`、`mean_abs`、`max_rel`。对于官方 wrapper 不暴露的内部张量，只保存 PyTorch 侧结果；如果官方 MIMO API 能稳定返回 final state、final K 或 final angle，则纳入额外比对。

允许存在小幅数值差异，因为官方 TileLang kernel 可能启用 fast math、内部使用 `bfloat16`/混合精度、并在 RoPE 与激活函数上使用近似实现。报告会记录这些限制，不把性能优化或逐 bit 一致作为目标。

## SISO 与 MIMO 差异说明

README 和报告会明确说明以下差异：

- SISO 的 `mimo_rank=1`，`Q/K` 可视作单一路径；MIMO 必须保留 rank 维 `[B, S, R, G, N]`。
- MIMO 需要处理 `Q_bias/K_bias`、`MIMO_V`、`MIMO_Out`、`MIMO_Z`，这些参数控制输入 rank 展开、输出 rank 聚合和 gate 调制。
- SISO chunk 内相关性主要在时间维 `[chunk, chunk]` 上计算；MIMO 会在 `[chunk * rank, chunk * rank]` 展开空间中计算。
- MIMO 的 `V` 先经 `MIMO_V` 按 rank 扩展，再通过 `MIMO_Out` 聚合回每个 head 的输出；SISO 没有显式 rank mixing。
- MIMO 的 RoPE 仍作用在 Q/K 的 state/rotary 维，但需要覆盖 rank 展开后的 Q/K。

## 错误处理

脚本启动时检查 `mamba_ssm`、TileLang kernel、state dict 或随机权重初始化路径是否可用。缺失时给出明确异常。若官方 kernel 不能在当前设备运行，脚本会说明无法生成 reference。

## 验证

先运行脚本自身生成 `comparison_report.md`。若环境支持，再运行与新增脚本相关的轻量验证命令，确认脚本可导入或可执行。

## 非目标

不复现 backward，不做训练，不做量化实验，不做性能优化，不要求 bitwise 一致，也不改动官方 site-packages 源码。
