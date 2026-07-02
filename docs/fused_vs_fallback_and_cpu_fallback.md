# Fused Kernel、Fallback 路径和 cpu_fallback 说明

## 这份文档解释什么

本项目里你会看到这些说法：

- `Mamba block；使用非 fused fallback 路径`
- `Mamba2 block；使用非 fused fallback 路径`
- `runtime_source=cpu_fallback`

它们容易混淆，但含义不同。

简单说：

- **fused / non-fused fallback** 说的是模型前向计算时用了哪种实现路径。
- **cpu_fallback** 说的是 profiler 报告里运行时间 breakdown 用了哪种计时来源。

二者不是一回事。

## 什么是 fused kernel

`fused` 可以理解成“融合算子”。

普通神经网络计算往往由很多小步骤组成，例如：

1. 线性投影
2. 卷积
3. 激活函数
4. 状态空间扫描
5. 门控
6. 输出投影

如果每一步都单独调用一个 PyTorch 或 CUDA 算子，GPU 需要反复启动 kernel，反复读写中间张量。

`fused kernel` 的思想是：把多个连续步骤合并到一个或少数几个自定义 CUDA/Triton kernel 里执行。

这样做的好处是：

- 减少 CUDA kernel 启动次数。
- 减少中间张量写回显存再读出的次数。
- 更好地利用 GPU cache、寄存器和共享内存。
- 对长序列、大 batch、大模型时通常更快。

代价是：

- 对 CUDA、PyTorch、Triton、显卡架构版本更敏感。
- 编译和安装更麻烦。
- profiler 里看到的算子可能变少，但单个 fused kernel 内部做了很多事情，不一定容易拆开理解。
- 如果依赖版本不完全匹配，可能导入或运行失败。

## 什么是 non-fused fallback

`fallback` 是“备用路径”。

`non-fused fallback` 指的是：不用高度融合的自定义 kernel，而是退回到更基础、更通用的 PyTorch/Triton 实现。

例如 Mamba/Mamba2 里本来可以走 fused fast path，但本项目为了兼容当前环境，显式设置了：

```python
Mamba(..., use_fast_path=False)
Mamba2(..., use_mem_eff_path=False)
```

这样会让官方模块使用更普通的实现路径。

它的好处是：

- 更容易跑通。
- 对环境版本要求低一些。
- profiler 里能看到更多中间算子，适合学习每一步大概做什么。
- 出错时更容易定位问题。

它的缺点是：

- 通常比 fused kernel 慢。
- 中间张量更多，可能有更多显存分配。
- 不能代表论文中极致优化 kernel 的最好性能。

## Mamba 使用 fused 和 fallback 的区别

Mamba1 的核心计算包括：

- 输入投影，把输入 hidden state 映射到内部维度。
- 因果卷积，用短卷积混合局部上下文。
- `SelectiveScanFn`，也就是选择性状态空间扫描。
- 门控和输出投影。

在 fused fast path 中，Mamba 会尽量调用更融合的实现，例如把卷积、投影、扫描等操作组合得更紧密。

在 fallback path 中，这些步骤更多地表现为普通算子，例如：

- `aten::matmul`
- `aten::mm`
- `aten::linear`
- `CausalConv1dFn`
- `SelectiveScanFn`
- `aten::empty`
- `aten::empty_strided`

所以本项目报告里看到这些算子，是因为我们更偏向“可解释、可跑通”的路径，而不是最强性能路径。

## Mamba2 使用 fused 和 fallback 的区别

Mamba2 的设计和 Mamba1 不完全一样。它引入了更适合分块并行的结构，核心里会出现类似：

- 输入投影。
- 分块状态空间扫描。
- `MambaChunkScanCombinedFn`。
- RMSNorm 或门控归一化。
- 输出投影。

Mamba2 的 fused / memory-efficient path 会尽量把更多计算合并起来，减少中间结果和显存访问。

本项目中使用：

```python
Mamba2(..., use_mem_eff_path=False)
```

表示关闭这条 memory-efficient fused 路径，改走更基础的 fallback 逻辑。

这样做的主要原因是当前环境里多个依赖版本之间比较敏感：

- `torch 2.3.1+cu121`
- `triton 3.7.0`
- `mamba-ssm 2.3.2.post1`
- `causal-conv1d 1.4.0`
- CUDA toolkit 12.1

为了保证 Mamba、Mamba2、Mamba3 都能进入同一套 profiling 流程，当前选择了更稳的 fallback 路径。

## 为什么 fallback 结果不能直接和论文最快结果比较

论文里的性能结果通常依赖：

- 官方推荐版本组合。
- 特定 CUDA/Triton kernel。
- 大 batch 或长序列输入。
- H100/A100 等数据中心 GPU。
- 更完整的 benchmark 脚本和严格同步计时。

本项目当前目标是帮助你理解：

- 哪些算子出现了。
- 哪些算子更耗时。
- 哪些算子分配了显存。
- Mamba/Mamba2/Mamba3 在同一台机器上的小规模行为。

所以它是教学和本机 profiling demo，不是论文极限性能复现。

## 什么是 cpu_fallback

`cpu_fallback` 出现在 `reports/runtime_breakdown.csv` 的 `runtime_source` 列里。

它和模型是否使用 fused kernel 没有直接关系。

它的意思是：

> PyTorch profiler 没有给出有效的 CUDA 算子时间，所以报告退回使用 CPU 侧 profiler 时间来做运行时间 breakdown。

当前环境里，`operator_profile.csv` 的 `cuda_time_total_us` 全部是 0。这说明 PyTorch profiler 没有记录到每个算子的 CUDA 时间。

为了仍然能回答“哪些算子看起来更耗时”，脚本会自动选择：

```text
如果 cuda_time_total_us 总和 > 0：使用 CUDA 时间
否则：使用 cpu_time_total_us，并标记 runtime_source=cpu_fallback
```

## cpu_fallback 统计的是什么时间

`cpu_fallback` 使用的是 `cpu_time_total_us`。

它大致表示 CPU 侧在调度、调用、等待某个 PyTorch 算子时 profiler 看到的总时间。

它可能包含：

- Python/PyTorch 调用开销。
- CUDA kernel 启动开销。
- 某些 CPU 侧准备工作。
- profiler 引入的同步或记录开销。

它不等于真正的 GPU kernel 执行时间。

## cpu_fallback 可以怎么看

可以用它粗略理解：

- 哪类算子在当前前向过程中更常出现。
- 哪些 PyTorch 调用链比较重。
- 不同 block 的大致算子结构差异。

不要用它严格判断：

- GPU kernel 真实耗时。
- 显存带宽瓶颈。
- fused kernel 内部每一步的真实占比。
- 与论文表格中的 GPU throughput 直接比较。

如果要更精确地看 GPU 侧 kernel 时间，需要 Nsight Systems 或 Nsight Compute，例如：

- `nsys profile ...`
- `ncu ...`

这些工具能看到更底层的 CUDA kernel 时间、内存吞吐、SM 利用率等。

## 当前报告应该如何阅读

建议这样读：

1. 先看 `reports/summary.csv`。
   - 看每个 block 是否 `ok`。
   - 看整体平均前向时间和峰值显存。

2. 再看 `reports/runtime_breakdown.csv`。
   - 如果 `runtime_source=cpu_fallback`，把它理解为 CPU profiler 视角的算子耗时分布。
   - 看 `percent`，了解每个算子占该 block 总 profiler 时间的大概比例。

3. 再看 `reports/memory_breakdown.csv`。
   - 看 `memory_mb` 和 `percent`，了解哪些算子申请了较多中间显存。
   - `aten::empty`、`aten::empty_strided` 这类通常不是数学计算，而是中间张量分配。

4. 最后看图。
   - `runtime_breakdown_stacked.png`：运行时间来源的堆叠图。
   - `memory_breakdown_stacked.png`：显存分配来源的堆叠图。

## 一句话总结

`fused` 是更快但更依赖环境的高度融合 GPU 实现；`fallback` 是更稳、更容易解释但通常更慢的备用实现。

`cpu_fallback` 不是模型实现路径，而是 profiler 计时来源的说明：当前没有拿到 CUDA 算子时间，所以用 CPU profiler 时间来做 breakdown。
