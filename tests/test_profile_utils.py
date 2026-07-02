from pathlib import Path

import pandas as pd

from benchmarks.profile_utils import BenchmarkSummary, write_reports, write_timeline_chart


def test_write_reports_creates_expected_files(tmp_path: Path) -> None:
    summary = BenchmarkSummary(
        block_name="torch_baseline",
        case_name="quick",
        device="cpu",
        batch_size=1,
        seq_len=8,
        d_model=16,
        avg_time_ms=1.5,
        peak_memory_mb=0.0,
        status="ok",
        note="测试数据",
    )
    operator_df = pd.DataFrame(
        [
            {
                "block_name": "torch_baseline",
                "case_name": "quick",
                "operator": "aten::linear",
                "calls": 1,
                "cpu_time_total_us": 10.0,
                "cuda_time_total_us": 30.0,
                "self_cpu_memory_usage_bytes": 0,
                "self_cuda_memory_usage_bytes": 1024,
                "input_shapes": "[]",
            },
            {
                "block_name": "torch_baseline",
                "case_name": "quick",
                "operator": "aten::silu",
                "calls": 1,
                "cpu_time_total_us": 5.0,
                "cuda_time_total_us": 10.0,
                "self_cpu_memory_usage_bytes": 0,
                "self_cuda_memory_usage_bytes": 512,
                "input_shapes": "[]",
            }
        ]
    )

    write_reports(tmp_path, [summary], [operator_df])

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "operator_profile.csv").exists()
    assert (tmp_path / "runtime_breakdown.csv").exists()
    assert (tmp_path / "memory_breakdown.csv").exists()
    assert (tmp_path / "runtime_breakdown_stacked.png").exists()
    assert (tmp_path / "memory_breakdown_stacked.png").exists()
    assert (tmp_path / "README_report.md").exists()
    runtime_df = pd.read_csv(tmp_path / "runtime_breakdown.csv")
    memory_df = pd.read_csv(tmp_path / "memory_breakdown.csv")
    for df in [runtime_df, memory_df]:
        assert "operator_category_zh" in df.columns
        assert "operator_name_zh" in df.columns
        assert "operator_explanation_zh" in df.columns
    assert "线性层" in runtime_df["operator_name_zh"].to_string()
    assert "Mamba 性能报告" in (tmp_path / "README_report.md").read_text(encoding="utf-8")
    assert "运行时间 breakdown" in (tmp_path / "README_report.md").read_text(encoding="utf-8")


def test_write_timeline_chart_creates_gantt_plot(tmp_path: Path) -> None:
    timeline_df = pd.DataFrame(
        [
            {"operator": "aten::linear", "start_us": 0.0, "end_us": 100.0, "duration_us": 100.0},
            {"operator": "aten::linear", "start_us": 130.0, "end_us": 160.0, "duration_us": 30.0},
            {"operator": "aten::mm", "start_us": 120.0, "end_us": 200.0, "duration_us": 80.0},
        ]
    )

    write_timeline_chart(timeline_df, tmp_path / "timeline.png", title="Mamba3 timeline")

    assert (tmp_path / "timeline.png").exists()
    assert (tmp_path / "timeline.png").stat().st_size > 0
