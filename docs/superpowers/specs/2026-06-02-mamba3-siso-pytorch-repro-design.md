# Mamba3 SISO PyTorch Block 复现设计

## 目标

在新建子目录中实现一个纯 PyTorch 版 Mamba3 SISO block 前向计算流程，不调用官方 Mamba3 fused kernel。脚本需要保存官方结果、PyTorch 复现结果、主要中间张量和数值比对摘要，目标是让 PyTorch 复现与官方包在数值上尽量一致。

## 范围

本阶段只处理当前 demo 配置对应的 SISO 路径：`is_mimo=False`、`mimo_rank=1`。MIMO TileLang 路径留到 SISO 跑通后再做。

脚本将使用当前项目已有的随机初始化权重：`artifacts/mamba3_model_info/mamba3_demo_state_dict.pt`。输入使用固定随机种子生成，默认形状与现有文档一致：`batch=1`、`seq_len=128`、`d_model=256`、`dtype=torch.float32`。

## 架构

新增目录：`artifacts/mamba3_siso_pytorch_repro/`。

核心文件：

- `mamba3_siso_pytorch_repro.py`：可直接运行的复现和比对脚本。
- `intermediates.pt`：保存输入、权重相关快照、官方输出、PyTorch 输出和中间结果。
- `comparison_report.md`：保存每个可比对张量的最大绝对误差、平均绝对误差和相对误差。

## 计算流程

脚本会实例化官方 `Mamba3` block，并加载 demo state dict。然后执行两条路径：

1. 官方 reference 路径：调用官方 block forward，仅用于生成最终输出 reference。
2. PyTorch 复现路径：按官方 `mamba3.py` 和 `mamba3_siso_fwd.py` 的语义逐步计算。

PyTorch 路径包括：

- `in_proj(u)`。
- 按官方顺序拆分 `z, x, B, C, dd_dt, dd_A, trap, angles`。
- reshape 到 SISO kernel 需要的布局。
- 计算 `_A`、`DT`、`ADT`。
- 对 `B`、`C` 执行等价 RMSNorm。
- 对 `Q=C`、`K=B` 加 bias 并应用 RoPE。
- 计算 `gamma = DT * sigmoid(trap)`、`shifted_gamma = DT[t+1] * (1 - sigmoid(trap[t+1]))`、`scale = gamma + shifted_gamma`。
- 按 chunk 复现官方 SISO forward 的跨 chunk state contribution、chunk 内 strictly causal contribution、`D` skip、`QK` diagonal contribution 和 `z` SiLU gate。
- `out_proj` 得到最终输出。

## 中间结果

保存的中间结果至少包括：

- 输入和配置：`u`、seed、shape 配置。
- 投影结果：`zxBCdtAtrap`、拆分后的 `z/x/B/C/dd_dt/dd_A/trap/angles`。
- 参数准备：`A`、`DT`、`ADT`、`gamma`、`shifted_gamma`、`scale`。
- kernel 准备：`Q_pre_bias`、`K_pre_bias`、`Q_rot`、`K_rot`、`QK_store`。
- 状态扫描：每个 chunk 的 `da_cs`、chunk 前状态、chunk 后状态、pre-gate 输出。
- 输出：`torch_y`、`torch_out`、`official_out`。

## 数值比对

最终输出与官方输出做 `max_abs`、`mean_abs`、`max_rel` 比对。对于官方 kernel 不暴露的内部张量，只保存 PyTorch 侧中间结果；若官方 wrapper 可通过 `store_states_adt_outv=True` 暴露某些 SISO 中间张量，则额外纳入比对。

数值一致标准先以 `float32` 小规模输入为目标。由于官方 Triton kernel 可能使用近似 `sin/cos/sigmoid/silu` 或内部 dtype 转换，允许先记录误差；如果误差偏大，再逐项定位 RoPE、trap scale、chunk state update 或 gate 的差异。

## 错误处理

脚本启动时检查 `mamba_ssm` 是否可导入、state dict 是否存在、官方 Mamba3 是否可实例化。缺失时给出明确错误信息，不伪造结果。

## 测试与验证

先运行脚本自身的比对流程。若项目测试可用，再运行已有 pytest，确保新增文件不破坏现有代码。

## 非目标

本阶段不实现 MIMO TileLang 路径，不训练模型，不做算子融合，不做性能优化，不复现 backward。
