# Mamba 小规模性能复现 Demo 设计

日期：2026-05-21

## 目标

本项目目标是在当前目录中搭建一个可运行的小规模 Mamba 性能复现 demo。它不尝试复现论文中的完整训练精度或大规模吞吐结果，而是帮助用户理解 Mamba 系列 block 在小输入规模下的运行时间、显存使用和主要算子开销。

## 输入资料

工作区已有三篇论文 PDF：

- `mamba.pdf`
- `mamba2.pdf`
- `mamba3.pdf`

当前模型不能直接通过文件读取工具理解 PDF 内容。后续实现会使用命令行或 Python PDF 工具提取文本。如果环境缺少 `pdftotext`，会使用 Python 依赖如 `pypdf` 或 `pdfplumber`。

## 范围

本 demo 包含：

- 创建本地 Python 虚拟环境 `.venv`。
- 安装 `mamba-ssm`、PyTorch、性能分析和绘图依赖。
- 编写 benchmark 脚本，运行小规模前向推理。
- 对不同 Mamba block 或可用替代实现进行对比。
- 使用 PyTorch profiler 输出每个算子的 CPU 时间、CUDA 时间、调用次数和显存变化。
- 使用 `torch.cuda.max_memory_allocated()` 记录整体峰值显存。
- 生成 CSV、Markdown 和 PNG 图表形式的性能报告。
- 编写中文 `README.md`，用基础友好的方式解释目录结构、运行方法和结果含义。

本 demo 不包含：

- 完整论文训练流程。
- 大数据集下载和训练。
- 对论文表格中所有指标的严格复现。
- Nsight Compute 级别的硬件访存统计，除非后续发现本机已安装对应工具。

## 架构

建议目录结构如下：

```text
.
├── README.md
├── docs/
│   ├── paper_notes.md
│   └── superpowers/specs/2026-05-21-mamba-profiling-demo-design.md
├── scripts/
│   └── setup_env.sh
├── benchmarks/
│   ├── benchmark_mamba_blocks.py
│   └── profile_utils.py
└── reports/
    ├── summary.csv
    ├── operator_profile.csv
    ├── runtime_bar.png
    ├── memory_bar.png
    └── README_report.md
```

## 组件设计

### 环境脚本

`scripts/setup_env.sh` 负责创建 `.venv` 并安装依赖。脚本会尽量保持简单，方便用户手动检查每一步。安装 `mamba-ssm` 时可能受 CUDA、PyTorch 和 Triton 版本影响，因此 README 会说明常见失败原因。

### Benchmark 主脚本

`benchmarks/benchmark_mamba_blocks.py` 负责检测 CUDA、构造小规模输入、导入可用 Mamba 模块、执行 warmup 和正式计时、调用 profiling 工具保存每个算子的性能明细，并生成汇总数据和图表。

如果 Mamba-3 没有可用官方实现，脚本会明确标注该项不可用，不伪造官方结果。

### Profiling 工具

`benchmarks/profile_utils.py` 负责封装 PyTorch profiler、将 profiler 结果转成表格、保存 CSV 和 Markdown 报告、生成运行时间和显存柱状图。

代码注释会使用中文，解释 profiler、warmup、CUDA 同步、显存统计等基础概念。

### 报告

报告输出包含：

- `summary.csv`：每种 block 和输入配置的总运行时间、峰值显存。
- `operator_profile.csv`：每个算子的 CPU/CUDA 时间、调用次数和显存变化。
- `runtime_bar.png`：不同 block 的运行时间对比图。
- `memory_bar.png`：不同 block 的峰值显存对比图。
- `README_report.md`：中文解释如何读这些表格和图。

## 数据流

1. 用户运行环境安装脚本。
2. 用户激活 `.venv`。
3. 用户运行 benchmark 脚本。
4. 脚本加载可用 Mamba block。
5. 脚本生成随机输入张量。
6. 脚本执行 warmup，避免第一次 CUDA 初始化影响计时。
7. 脚本执行 profiler，采集每个算子性能。
8. 脚本写出报告和图表。

## 错误处理

- 如果没有 CUDA，脚本会退回 CPU 或提示 GPU benchmark 不可用。
- 如果 `mamba-ssm` 安装失败，README 会说明排查方式。
- 如果某个 block 无法导入，脚本会跳过该 block，并在报告中写明原因。
- 如果绘图依赖不可用，脚本仍保存 CSV，不因为图表失败而丢失主要结果。

## 测试与验证

实现后至少运行以下验证：

- `python --version` 确认虚拟环境 Python 可用。
- `python -c "import torch; print(torch.cuda.is_available())"` 确认 PyTorch 和 CUDA 状态。
- `python -c "import mamba_ssm"` 确认 `mamba-ssm` 是否安装成功。
- 运行一次 benchmark，确认 `reports/` 下产生 CSV、Markdown 和 PNG 文件。

## 已知限制

当前机器检测到 NVIDIA GPU，但未检测到 `ncu` 或 `nvprof`。因此本项目默认使用 PyTorch profiler 统计算子级运行时间和显存变化，不提供 Nsight Compute 级硬件内存带宽分析。

本项目将尽量使用官方 `mamba-ssm`。如果某个论文版本没有公开可安装实现，报告必须明确说明不可用，不能把教学替代实现包装成官方复现。
