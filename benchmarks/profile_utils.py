from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch


OPERATOR_EXPLANATIONS_ZH = {
    "aten::matmul": ("矩阵计算", "矩阵乘法", "通用矩阵乘法接口。在线性层、投影层、注意力或状态空间计算里，经常用它把输入向量乘以权重矩阵。耗时高通常说明大部分计算量来自 dense 矩阵运算。"),
    "aten::mm": ("矩阵计算", "二维矩阵乘法", "专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。"),
    "aten::linear": ("神经网络层", "线性层", "PyTorch 的全连接层接口，本质是输入乘以权重矩阵，再可选加 bias。Mamba/Mamba2/Mamba3 里很多输入投影、输出投影都来自这个算子。"),
    "aten::addmm": ("矩阵计算", "加偏置矩阵乘法", "执行 bias + input @ weight 这类融合形式，常见于 nn.Linear。它同时包含矩阵乘法和加法，所以既会占用计算时间，也会申请输出张量显存。"),
    "aten::bmm": ("矩阵计算", "批量矩阵乘法", "一次处理一批小矩阵乘法。序列模型里如果把 batch、head 或 chunk 拆开计算，可能会出现这个算子。"),
    "aten::layer_norm": ("归一化", "层归一化入口", "LayerNorm 的高层接口，用来把每个 token 的特征归一化，让数值更稳定。它通常会继续调用 native_layer_norm。"),
    "aten::native_layer_norm": ("归一化", "层归一化底层实现", "LayerNorm 的底层实现，计算均值、方差并缩放输入。baseline 里它占比高，是因为 baseline block 很小，归一化开销相对明显。"),
    "aten::silu": ("激活函数", "SiLU 激活", "SiLU，也叫 Swish，形式大致是 x * sigmoid(x)。Mamba 系列里常用它作为门控或卷积后的非线性激活。"),
    "aten::softplus": ("激活/参数约束", "Softplus", "把数值平滑地变成正数，常用于保证时间步长 dt 等参数为正。Mamba3 里会用它处理 dt 或 A 相关参数。"),
    "aten::sigmoid": ("激活函数", "Sigmoid", "把数值压到 0 到 1 之间，常用于门控。出现它通常说明模型在计算某种比例、开关或概率型权重。"),
    "aten::exp": ("数学函数", "指数函数", "计算 e 的 x 次方。Mamba 里状态空间参数 A 有时会通过指数形式参数化，用来保证数值范围符合模型设计。"),
    "aten::neg": ("数学函数", "取负号", "把张量变成相反数。Mamba 状态空间里常会构造负的 A 参数，使状态衰减而不是发散。"),
    "aten::add": ("张量运算", "逐元素加法", "两个张量逐元素相加。常见于加 bias、残差连接、参数组合或中间结果累加。"),
    "aten::mul": ("张量运算", "逐元素乘法", "两个张量逐元素相乘。门控结构里很常见，例如一个分支产生内容，另一个分支产生门控权重。"),
    "aten::empty": ("显存分配", "申请未初始化张量", "只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。"),
    "aten::empty_strided": ("显存分配", "按步幅申请张量", "申请带有特定 stride 布局的张量。常由转置、reshape、contiguous 或某些 kernel 输出触发，主要反映中间结果的内存分配。"),
    "aten::to": ("数据转换", "类型或设备转换", "把张量转换 dtype、device 或 memory format。比如 float16/float32 之间转换，或为了 kernel 要求临时转换精度。"),
    "aten::permute": ("形状变换", "维度重排", "只改变张量维度顺序的视图操作，例如从 B,L,D 调整到 B,D,L。通常本身不大分配显存，但可能导致后续 contiguous 分配。"),
    "aten::transpose": ("形状变换", "转置", "交换两个维度。线性层或卷积前后经常需要转置以满足算子输入格式。"),
    "aten::reshape": ("形状变换", "重塑形状", "改变张量形状。多数情况下是视图操作，但如果内存不连续，也可能触发额外拷贝。"),
    "aten::view": ("形状变换", "视图变形", "不复制数据，只用新的形状解释同一块内存。通常耗时和显存都很小。"),
    "aten::as_strided": ("形状变换", "底层视图步幅", "PyTorch 用 stride 描述张量视图的底层操作。它通常由 view、transpose、slice 等高级操作间接触发。"),
    "cudaLaunchKernel": ("CUDA 调度", "启动 CUDA kernel", "CPU 向 GPU 发起一个 kernel 启动请求。它不是模型数学计算本身，但 kernel 很多时，调度开销会在 CPU profiler 中变明显。"),
    "cudaMemsetAsync": ("CUDA 内存操作", "异步清零/填充", "GPU 侧异步设置一段内存的值，常用于初始化输出缓冲区或临时状态。"),
    "cudaDeviceSynchronize": ("CUDA 同步", "等待 GPU 完成", "CPU 等待 GPU 上已提交的工作完成。profiling 或显式计时会引入同步，所以它可能出现在报告中。"),
    "SelectiveScanFn": ("Mamba 核心", "选择性扫描", "Mamba1 的核心状态空间扫描函数。它沿序列维度递推状态，是 Mamba 区别于普通 MLP/卷积的重要部分。"),
    "MambaChunkScanCombinedFn": ("Mamba2 核心", "分块扫描融合函数", "Mamba2 的核心分块扫描函数，把序列切成 chunk 来处理状态空间递推，目标是更适合 GPU 并行执行。"),
    "Other": ("汇总项", "其他算子", "为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。"),
}


@dataclass(frozen=True)
class BenchmarkCase:
    """一次实验的输入规模。"""

    name: str
    batch_size: int
    seq_len: int
    d_model: int


@dataclass(frozen=True)
class BenchmarkSummary:
    """一条汇总结果，对应 summary.csv 的一行。"""

    block_name: str
    case_name: str
    device: str
    batch_size: int
    seq_len: int
    d_model: int
    avg_time_ms: float
    peak_memory_mb: float
    status: str
    note: str


def synchronize_if_needed(device: torch.device) -> None:
    """CUDA 是异步执行的；同步后计时才更接近真实 GPU 用时。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_forward(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    """返回平均前向时间和峰值显存。

    warmup 用来避开第一次 CUDA 初始化、kernel 编译等额外开销。
    repeat 是正式测量次数，取平均值降低偶然波动。
    """

    device = x.device
    for _ in range(warmup):
        _ = forward_fn(x)
    synchronize_if_needed(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if device.type == "cuda":
        assert start is not None and end is not None
        start.record()
        for _ in range(repeat):
            _ = forward_fn(x)
        end.record()
        synchronize_if_needed(device)
        avg_time_ms = start.elapsed_time(end) / repeat
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    else:
        import time

        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = forward_fn(x)
        avg_time_ms = (time.perf_counter() - t0) * 1000 / repeat
        peak_memory_mb = 0.0

    return avg_time_ms, peak_memory_mb


def _cuda_time_total_us(item: torch.autograd.profiler_util.FunctionEventAvg) -> float:
    """兼容不同 PyTorch 版本里的 CUDA 时间字段名。"""

    if hasattr(item, "cuda_time_total"):
        return float(item.cuda_time_total)
    if hasattr(item, "device_time_total"):
        return float(item.device_time_total)
    return 0.0


def _cuda_memory_bytes(item: torch.autograd.profiler_util.FunctionEventAvg) -> int:
    """兼容不同 PyTorch 版本里的 CUDA 显存字段名。"""

    if hasattr(item, "self_cuda_memory_usage"):
        return int(item.self_cuda_memory_usage)
    if hasattr(item, "self_device_memory_usage"):
        return int(item.self_device_memory_usage)
    return 0


def profile_forward(
    block_name: str,
    case: BenchmarkCase,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> pd.DataFrame:
    """用 PyTorch profiler 采集算子级信息。"""

    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        _ = forward_fn(x)
        synchronize_if_needed(x.device)

    return _profiler_key_averages_to_dataframe(prof, block_name, case)


def profile_forward_with_timeline(
    block_name: str,
    case: BenchmarkCase,
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    max_events: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """采集算子聚合表和单次执行时间线。

    聚合表用于 breakdown；timeline 表用于画类似甘特图的横向时间轴。
    当前环境拿不到 CUDA kernel 时间戳时，timeline 展示 CPU profiler event 时间。
    """

    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        _ = forward_fn(x)
        synchronize_if_needed(x.device)

    summary_df = _profiler_key_averages_to_dataframe(prof, block_name, case)
    timeline_df = _profiler_events_to_timeline_dataframe(prof, block_name, case, max_events=max_events)
    return summary_df, timeline_df


def _profiler_key_averages_to_dataframe(
    prof: torch.profiler.profile,
    block_name: str,
    case: BenchmarkCase,
) -> pd.DataFrame:
    """把 key_averages 转成 DataFrame。"""

    rows = []
    for item in prof.key_averages():
        rows.append(
            {
                "block_name": block_name,
                "case_name": case.name,
                "operator": item.key,
                "calls": item.count,
                "cpu_time_total_us": item.cpu_time_total,
                "cuda_time_total_us": _cuda_time_total_us(item),
                "self_cpu_memory_usage_bytes": item.self_cpu_memory_usage,
                "self_cuda_memory_usage_bytes": _cuda_memory_bytes(item),
                "input_shapes": str(item.input_shapes),
            }
        )
    return pd.DataFrame(rows)


def _profiler_events_to_timeline_dataframe(
    prof: torch.profiler.profile,
    block_name: str,
    case: BenchmarkCase,
    max_events: int,
) -> pd.DataFrame:
    """把单次 event 转成时间线表。"""

    rows = []
    for event in prof.events():
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            continue
        duration_us = float(getattr(event, "cpu_time_total", 0.0))
        if duration_us <= 0:
            continue
        rows.append(
            {
                "block_name": block_name,
                "case_name": case.name,
                "operator": event.key,
                "start_us": float(time_range.start),
                "end_us": float(time_range.end),
                "duration_us": duration_us,
                "timeline_source": "cpu_profiler_event",
            }
        )
    timeline_df = pd.DataFrame(rows)
    if timeline_df.empty:
        return timeline_df
    timeline_df["start_us"] = timeline_df["start_us"] - timeline_df["start_us"].min()
    timeline_df["end_us"] = timeline_df["end_us"] - timeline_df["end_us"].min()
    timeline_df = timeline_df.sort_values(["start_us", "duration_us"], ascending=[True, False]).head(max_events)
    return add_operator_explanations(timeline_df.reset_index(drop=True))


def write_reports(
    report_dir: Path,
    summaries: Iterable[BenchmarkSummary],
    operator_frames: Iterable[pd.DataFrame],
) -> None:
    """保存 CSV、Markdown 和图表。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    frame_list = list(operator_frames)
    operator_df = pd.concat(frame_list, ignore_index=True) if frame_list else pd.DataFrame()

    summary_df.to_csv(report_dir / "summary.csv", index=False)
    operator_df.to_csv(report_dir / "operator_profile.csv", index=False)
    runtime_breakdown_df, memory_breakdown_df = build_operator_breakdowns(operator_df)
    runtime_breakdown_df.to_csv(report_dir / "runtime_breakdown.csv", index=False)
    memory_breakdown_df.to_csv(report_dir / "memory_breakdown.csv", index=False)
    write_markdown_report(report_dir / "README_report.md", summary_df, operator_df, runtime_breakdown_df, memory_breakdown_df)
    write_charts(report_dir, summary_df, runtime_breakdown_df, memory_breakdown_df)


def build_operator_breakdowns(operator_df: pd.DataFrame, top_n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 block 聚合算子耗时和显存，并保留每个 block 的 top_n 项。

    PyTorch profiler 会把同名算子的多次调用聚合到多行，这里再次按
    block/case/operator 汇总，方便画 stacked bar 图。超过 top_n 的算子合并为 Other。
    """

    if operator_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 某些 PyTorch/CUDA 组合下 profiler 的 CUDA time 会全部为 0。
    # 这种情况下退回 CPU profiler time，至少能展示 Python/PyTorch 调度视角下的算子延时分布。
    runtime_source = "cuda_time_total_us" if operator_df["cuda_time_total_us"].sum() > 0 else "cpu_time_total_us"
    runtime_df = _build_single_breakdown(operator_df, runtime_source, "runtime_ms", top_n, scale=1000.0)
    if not runtime_df.empty:
        runtime_df["runtime_source"] = "cuda" if runtime_source == "cuda_time_total_us" else "cpu_fallback"
    memory_df = _build_single_breakdown(operator_df, "self_cuda_memory_usage_bytes", "memory_mb", top_n, scale=1024.0 * 1024.0)
    return runtime_df, memory_df


def _build_single_breakdown(
    operator_df: pd.DataFrame,
    source_column: str,
    value_column: str,
    top_n: int,
    scale: float,
) -> pd.DataFrame:
    """构造单类 breakdown 表，例如耗时或显存。"""

    grouped = (
        operator_df.groupby(["block_name", "case_name", "operator"], as_index=False)[source_column]
        .sum()
        .rename(columns={source_column: value_column})
    )
    # 显存可能出现负数，代表释放。这里关注“谁占用/申请过显存”，所以只统计正数。
    grouped[value_column] = grouped[value_column].clip(lower=0) / scale
    grouped = grouped[grouped[value_column] > 0].copy()
    if grouped.empty:
        return grouped

    rows = []
    for (block_name, case_name), part in grouped.groupby(["block_name", "case_name"]):
        part = part.sort_values(value_column, ascending=False)
        total = float(part[value_column].sum())
        top = part.head(top_n).copy()
        other_value = float(part.iloc[top_n:][value_column].sum())
        if other_value > 0:
            top = pd.concat(
                [
                    top,
                    pd.DataFrame(
                        [
                            {
                                "block_name": block_name,
                                "case_name": case_name,
                                "operator": "Other",
                                value_column: other_value,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        top["total"] = total
        top["percent"] = top[value_column] / total * 100 if total else 0.0
        rows.append(top)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return add_operator_explanations(result)


def add_operator_explanations(df: pd.DataFrame) -> pd.DataFrame:
    """给 breakdown 表添加中文算子说明列。"""

    if df.empty:
        return df
    df = df.copy()
    explanations = df["operator"].map(explain_operator_zh)
    df["operator_category_zh"] = explanations.map(lambda item: item[0])
    df["operator_name_zh"] = explanations.map(lambda item: item[1])
    df["operator_explanation_zh"] = explanations.map(lambda item: item[2])
    return df


def explain_operator_zh(operator: str) -> tuple[str, str, str]:
    """返回算子的中文类别、中文名和详细解释。"""

    if operator in OPERATOR_EXPLANATIONS_ZH:
        return OPERATOR_EXPLANATIONS_ZH[operator]
    if "mamba3" in operator.lower():
        return ("Mamba3 核心", "Mamba3 自定义 kernel", "Mamba3 的 Triton/TileLang/CUTE 自定义算子，通常负责状态空间递推、旋转位置相关计算或 MIMO/SISO 扫描。")
    if "layer_norm" in operator or "rms_norm" in operator.lower() or "RMSNorm" in operator:
        return ("归一化", "归一化相关算子", "用于稳定神经网络数值分布的归一化操作，例如 LayerNorm 或 RMSNorm。")
    if "conv" in operator.lower():
        return ("卷积", "卷积相关算子", "卷积或因果卷积相关操作。Mamba/Mamba2 中短卷积用于混合局部上下文信息。")
    if "scan" in operator.lower():
        return ("状态空间扫描", "扫描/递推相关算子", "沿序列维度进行状态递推或分块扫描，是 Mamba 类模型的核心计算之一。")
    return ("其他", "未专门标注的算子", "PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。")


def write_markdown_report(
    path: Path,
    summary_df: pd.DataFrame,
    operator_df: pd.DataFrame,
    runtime_breakdown_df: pd.DataFrame,
    memory_breakdown_df: pd.DataFrame,
) -> None:
    """生成中文报告说明，方便不熟悉 profiler 的读者阅读。"""

    lines = [
        "# Mamba 性能报告",
        "",
        "## 汇总结果",
        "",
        summary_df.to_markdown(index=False) if not summary_df.empty else "没有成功的 benchmark 结果。",
        "",
        "## 算子级结果说明",
        "",
        "`operator_profile.csv` 中每一行是一个 PyTorch profiler 看到的算子。",
        "`cuda_time_total_us` 表示该算子累计 CUDA 时间，单位是微秒。",
        "`self_cuda_memory_usage_bytes` 表示该算子自身造成的 CUDA 显存变化，单位是字节。",
        "`runtime_breakdown.csv` 是按 block 聚合后的运行时间 breakdown，单位是毫秒。",
        "如果 `runtime_source=cpu_fallback`，表示当前 PyTorch profiler 没有给出 CUDA 时间，图中使用 CPU profiler 时间近似说明各算子的延时占比。",
        "`memory_breakdown.csv` 是按 block 聚合后的显存占用 breakdown，单位是 MB。",
        "",
    ]
    if not operator_df.empty:
        top_ops = operator_df.sort_values("cuda_time_total_us", ascending=False).head(20)
        lines.extend(["## CUDA 时间最高的前 20 个算子", "", top_ops.to_markdown(index=False), ""])
    if not runtime_breakdown_df.empty:
        lines.extend(["## 运行时间 breakdown", "", runtime_breakdown_df.to_markdown(index=False), ""])
    if not memory_breakdown_df.empty:
        lines.extend(["## 显存占用 breakdown", "", memory_breakdown_df.to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_charts(
    report_dir: Path,
    summary_df: pd.DataFrame,
    runtime_breakdown_df: pd.DataFrame,
    memory_breakdown_df: pd.DataFrame,
) -> None:
    """生成运行时间和显存柱状图。"""

    if summary_df.empty:
        return

    import matplotlib.pyplot as plt

    ok_df = summary_df[summary_df["status"] == "ok"].copy()
    if ok_df.empty:
        return

    ok_df["label"] = ok_df["block_name"] + "\n" + ok_df["case_name"]

    plt.figure(figsize=(10, 5))
    plt.bar(ok_df["label"], ok_df["avg_time_ms"])
    # 图表使用英文轴标签，避免某些 Linux 环境缺少中文字体导致乱码。
    plt.ylabel("Average forward time / ms")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "runtime_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(ok_df["label"], ok_df["peak_memory_mb"])
    plt.ylabel("Peak memory / MB")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "memory_bar.png", dpi=160)
    plt.close()

    write_stacked_breakdown_chart(report_dir / "runtime_breakdown_stacked.png", runtime_breakdown_df, "runtime_ms", "Runtime breakdown / ms")
    write_stacked_breakdown_chart(report_dir / "memory_breakdown_stacked.png", memory_breakdown_df, "memory_mb", "Memory allocation breakdown / MB")


def write_stacked_breakdown_chart(path: Path, breakdown_df: pd.DataFrame, value_column: str, ylabel: str) -> None:
    """把每个 block 的 top operators 画成 stacked bar。"""

    if breakdown_df.empty:
        return

    import matplotlib.pyplot as plt

    plot_df = breakdown_df.copy()
    plot_df["label"] = plot_df["block_name"] + "\n" + plot_df["case_name"]
    pivot = plot_df.pivot_table(index="label", columns="operator", values=value_column, aggfunc="sum", fill_value=0)
    # 让总体贡献最大的算子排在前面，图例更容易阅读。
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(13, 6), width=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.legend(title="Operator", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_timeline_chart(timeline_df: pd.DataFrame, path: Path, title: str) -> None:
    """画横向甘特图，展示算子在单次前向中的开始时间和持续时间。"""

    if timeline_df.empty:
        return

    import matplotlib.pyplot as plt

    plot_df = timeline_df.copy().sort_values("start_us")
    if "operator_category_zh" not in plot_df.columns:
        plot_df = add_operator_explanations(plot_df)

    # 同一个 operator 合并到同一行，避免图太长；行顺序按首次出现时间排序。
    first_start = plot_df.groupby("operator")["start_us"].min().sort_values()
    operators = first_start.index.tolist()
    y_map = {operator: idx for idx, operator in enumerate(operators)}
    category_color_map = build_timeline_color_map(plot_df["operator_category_zh"].dropna().unique())

    fig_height = max(6, min(18, len(operators) * 0.34))
    plt.figure(figsize=(14, fig_height))
    for _, row in plot_df.iterrows():
        category = row.get("operator_category_zh", "其他")
        plt.barh(
            y_map[row["operator"]],
            row["duration_us"] / 1000.0,
            left=row["start_us"] / 1000.0,
            height=0.72,
            color=category_color_map.get(category, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.2,
        )

    plt.yticks(range(len(operators)), operators, fontsize=8)
    plt.xlabel("Relative time / ms")
    plt.title(title)
    plt.gca().invert_yaxis()
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in category_color_map.values()]
    legend_labels = [timeline_category_label_en(category) for category in category_color_map.keys()]
    plt.legend(legend_handles, legend_labels, title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_timeline_color_map(categories: list[str]) -> dict[str, str]:
    """给算子类别分配稳定颜色。"""

    preferred = {
        "矩阵计算": "#1f77b4",
        "神经网络层": "#17becf",
        "Mamba3 核心": "#d62728",
        "Mamba2 核心": "#9467bd",
        "Mamba 核心": "#8c564b",
        "状态空间扫描": "#e377c2",
        "CUDA 调度": "#ff7f0e",
        "CUDA 内存操作": "#bcbd22",
        "CUDA 同步": "#7f7f7f",
        "显存分配": "#2ca02c",
        "形状变换": "#aec7e8",
        "张量运算": "#98df8a",
        "激活函数": "#ff9896",
        "激活/参数约束": "#c5b0d5",
        "数学函数": "#c49c94",
        "卷积": "#f7b6d2",
        "归一化": "#dbdb8d",
        "数据转换": "#9edae5",
        "其他": "#7f7f7f",
    }
    return {category: preferred.get(category, "#7f7f7f") for category in categories}


def timeline_category_label_en(category: str) -> str:
    """图例使用英文，避免系统缺中文字体。"""

    labels = {
        "矩阵计算": "Matrix ops",
        "神经网络层": "NN layers",
        "Mamba3 核心": "Mamba3 core",
        "Mamba2 核心": "Mamba2 core",
        "Mamba 核心": "Mamba core",
        "状态空间扫描": "State scan",
        "CUDA 调度": "CUDA launch",
        "CUDA 内存操作": "CUDA memory",
        "CUDA 同步": "CUDA sync",
        "显存分配": "Allocation",
        "形状变换": "Shape ops",
        "张量运算": "Tensor ops",
        "激活函数": "Activation",
        "激活/参数约束": "Activation/params",
        "数学函数": "Math ops",
        "卷积": "Convolution",
        "归一化": "Normalization",
        "数据转换": "Cast/transfer",
        "其他": "Other",
    }
    return labels.get(category, "Other")
