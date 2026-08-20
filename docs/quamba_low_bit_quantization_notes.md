# Quamba 与 Quamba2 低比特量化思路

## 背景

Quamba 和 Quamba2 都针对 Mamba/Selective State Space Models 做后训练量化（PTQ）。核心判断是：Mamba 的误差敏感点不只在线性层，selective scan 的状态递推、门控、指数衰减和激活离群值都会放大量化误差。如果只把 Mamba block 当成普通 Transformer MLP 来量化，精度往往在长序列和状态更新路径上崩掉。

本笔记基于 `quamba_iclr2025/source` 和 `quamba2_icml2025/source` 中的实现，提炼对 Mamba3 和 ASIC 量化设计有用的思路。

## Quamba

Quamba 的重点是给 Mamba1 建立一套可部署的 PTQ recipe。它先用校准数据统计激活范围，再把 RMSNorm/Linear 等边界整理成更适合量化的形式。权重量化支持 W8/W4，激活量化主要围绕 A8，状态和 selective scan 路径有独立 scale，而不是把整块当成普通 Transformer MLP 量化。

关键模块（来自 `quamba_iclr2025/source/quamba`）：

- `observer.py`：`PerTensorMinmaxObserver`、`PerTensorPercentileObserver`、`PerSSDGroupObserver`、`CrossHeadMinmaxObserver`。校准时按算子收集 input/output 的 min/max 或 percentile，区别对待 scan 输入、B/C、ssm_state 等关键张量。
- `qMambaLayer.py`：`MambaSimple` 把原始 fused block 拆成可观察的 in_proj、conv1d、x_proj、dt_proj、selective_scan、out_proj。这是 Quamba 能逐点插 scale 的前提。
- `qSelectiveScan.py` / `qChunkScan.py`：把 selective scan 内部的 `u, dt, A, B, C, ssm_state, D, z` 全部以 int8 + scale 的形式送进专用 kernel。这说明 scan 路径不能只用线性层的 W8A8 假设。
- `qLinearLayer.py`：`W8A8B8O8Linear`、`W4A8B16O16Linear`、`W4A16B16O16Linear` 等给出不同 bit-width 组合，权重用 per-channel/per-group scale，激活用 per-tensor scale。
- `hadamard_utils.py`：用 Hadamard 变换平滑激活分布，减少离群值对 INT8/INT4 的压力，并把变换矩阵融合进相邻 Linear 权重。
- `modelutils_mamba.py`：`run_quamba_calibration` 注册 hook、跑校准集、收集 scale；`quantize_fp16_model` 按 W8A8/W4A8/W4A16 替换模块；`apply_gptq` 用 wikitext2 做 GPTQ 权重量化。

关键启发：

- 线性层权重可以低到 8bit 或 4bit，但 selective scan 的输入、状态和输出需要单独校准。
- 状态递推中的 scale 粒度会影响误差累积，per-tensor scale 往往只是基线。
- Hadamard/rotation 和 norm/linear 融合用于平滑激活分布，减少离群值对 INT8/INT4 的压力。
- `delta_softplus`、`exp(A)`、`sigmoid` 等控制路径在量化时不能简单截断，需要在校准范围和 scale 选择上留余量。

## Quamba2

Quamba2 把框架扩展到 Mamba2 和更大模型，支持 W8A8、W4A8、W4A16、W4AX。它加入 head/channel grouping、reorder、GPTQ、hybrid blocks，并提供真实部署的 latency/memory profiling。

关键模块：

- `qChunkScan.py`：`Quamba2ChunkScan` 支持 `nhead_groups` 和 `ndim_groups`，把 state scale 做成 `[ngroups, nhead_groups, ndim_groups, d_state]`，而不是单一 per-tensor scale。这是 Mamba2 在 chunk scan 下控制误差累积的关键。
- `reorder_utils.py`：`group_wise_sort_indices` 用 AgglomerativeClustering 和 KMeans 把 head/channel 按激活幅度聚类重排，让同一组内的 range 接近，从而 group-wise scale 更紧。
- `modelutils_mamba.py`：`run_quamba2_calibration` 为 `x_conv_out` 用 `CrossHeadMinmaxObserver`，为 B/C 用 `PerSSDGroupObserver`，为 ssm_state 用 `CachedStatesCrossHeadMinmaxObserver`。`quantize_fp16_model_act_hybrid` 支持按层混合 W4A8/W4A16，敏感层保留更高激活精度。
- `qMamba2.py` / `qBlock.py`：`HybridQuambaBlock` 在运行时根据配置切换 mixer/norm 的 bit-width，便于在精度和成本之间做 trade-off。

关键启发：

- W4A8 对部分层不稳定时，可以用 W4AX/hybrid blocks 保留敏感层的更高激活精度。
- Mamba2 的 chunk scan/state passing 需要专门 kernel 和 scale 管理，不能直接套 Mamba1 的 per-tensor scale。
- 对部署来说，量化格式和 scale 布局必须服务 kernel 数据流；论文级 PTQ 不是只看 checkpoint size。
- GPTQ 在 Mamba 上仍然有用，但校准集和 group_size 选择会显著影响最终 PPL。

## 对 Mamba3 和 ASIC 的启发

本项目的 Mamba3 实验先做 fake-quant PTQ，而不是直接写 INT8/INT4 kernel。这样可以先定位：

- 单 block recurrent state SRAM 预算是否可控。本项目按用户要求把单 block runtime state 限制在 1024 elements，4 层模型总共 4096 elements per batch item，硬件约束必须明确 per-block 和 whole-model 两种口径。
- W8A8 是否在 validation/test perplexity 上接近 FP32。如果通过 10% PPL 标准，可作为第一版 INT8 datapath 候选。
- W4A8 失效时误差更可能来自权重、激活还是 scan/state 路径。Quamba 的经验表明 scan/state 路径往往是首要嫌疑。
- ASIC 上 softplus、sigmoid、exp、RoPE、state update 是否需要保留更高精度或查表近似。这些控制路径对 range 和数值稳定性敏感，不能简单全低比特。

后续 ASIC 优化应优先把 state layout、scale storage、scan 累积精度和门控控制路径定义清楚，再决定 INT8/INT4 MAC 阵列和片上 SRAM 分块。Quamba2 的 head/channel grouping 和 hybrid blocks 思路可以直接映射到 ASIC 的 tile 分组和混合精度单元设计。
