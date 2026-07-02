# 项目阶段总结

## 为什么做这件事

本项目的目标是基于 `mamba.pdf`、`mamba2.pdf`、`mamba3.pdf`，在本机搭建一个可运行的小规模 Mamba 性能分析 demo。重点不是复现论文完整训练结果，而是理解 Mamba、Mamba2、Mamba3 block 在同一输入规模下的运行时间、显存占用和主要算子构成。

## 已完成工作

- 创建了 Python 虚拟环境 `.venv`。
- 安装并调通了 `mamba-ssm`，包括 Mamba、Mamba2、Mamba3 模块。
- 编写了 benchmark 脚本：`benchmarks/benchmark_mamba_blocks.py`。
- 编写了 profiler 和报告工具：`benchmarks/profile_utils.py`。
- 对 `torch_baseline`、Mamba、Mamba2、Mamba3 做了 quick 配置下的前向 profiling。
- 生成了运行时间、显存、算子 breakdown 和 Mamba3 timeline 图表。
- 编写了多份中文说明文档，解释 fused/fallback、cpu_fallback、Mamba3 算子对应关系等。

## 使用的方法和工具

- 使用 PyTorch 构造输入和运行前向。
- 使用 `mamba-ssm` 官方模块构造 Mamba/Mamba2/Mamba3 block。
- 使用 PyTorch profiler 获取算子级时间、显存和 event timeline。
- 使用 pandas 整理 CSV 表格。
- 使用 matplotlib 生成柱状图、堆叠图和 Mamba3 类甘特图。
- 使用源码阅读把 profile 中的算子映射回 Mamba3 forward 步骤。

## 主要产物

- `reports/summary.csv`：整体运行时间和峰值显存。
- `reports/operator_profile.csv`：原始算子级 profile。
- `reports/runtime_breakdown.csv`：运行时间 breakdown，含中文算子解释。
- `reports/memory_breakdown.csv`：显存占用 breakdown，含中文算子解释。
- `reports/runtime_breakdown_stacked.png`：运行时间堆叠图。
- `reports/memory_breakdown_stacked.png`：显存占用堆叠图。
- `reports/mamba3_timeline.csv`：Mamba3 单次前向 event 时间线。
- `reports/mamba3_timeline_gantt.png`：Mamba3 横向类甘特图。

## 当前关键结论

- Mamba、Mamba2、Mamba3 都已进入同一套 profiling 流程。
- Mamba/Mamba2 当前使用非 fused fallback 路径，目的是稳定运行和方便解释，不代表论文最快 kernel 性能。
- 当前 PyTorch profiler 没有给出有效 CUDA kernel 时间，所以运行时间 breakdown 使用 `cpu_fallback`，即 CPU profiler event 时间。
- Mamba3 中 `aten::linear/matmul/mm` 主要对应 `in_proj` 和 `out_proj` 两次投影。
- 当前 quick 配置下，Mamba3 两次投影合计约 `111.7 MFLOPs`。
- Mamba3 状态更新核心对应 `_Mamba3Function` 和它触发的 `cudaLaunchKernel`；状态更新前的参数准备对应 `softplus/add/mul/to/reshape/LayerNormFn` 等算子。

## 关于 Mamba3 模型文件

可以从源码实例化 Mamba3 block，并导出结构和随机初始化参数。注意这不是论文训练好的权重，只是 demo 配置下的模型结构和 state dict。

运行：

```bash
source .venv/bin/activate
CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-} \
python scripts/export_mamba3_model_info.py
```

输出：

- `artifacts/mamba3_model_info/mamba3_model_info.md`
- `artifacts/mamba3_model_info/mamba3_demo_state_dict.pt`

## 当前限制

- 结果是小规模 demo，不是论文完整复现。
- `cpu_fallback` 不能等同于真实 CUDA kernel 时间。
- 若需要真实 GPU kernel timeline，应使用 Nsight Systems 或 Nsight Compute。
- 导出的 `.pt` 是随机初始化权重，不是官方预训练模型。

## 推荐阅读顺序

1. `README.md`
2. `reports/README_report.md`
3. `docs/fused_vs_fallback_and_cpu_fallback.md`
4. `docs/mamba3_operator_mapping.md`
5. `summary/README.md`
