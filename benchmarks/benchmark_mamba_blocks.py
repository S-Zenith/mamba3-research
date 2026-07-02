from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

try:
    from profile_utils import BenchmarkCase, BenchmarkSummary, benchmark_forward, profile_forward, profile_forward_with_timeline, write_reports, write_timeline_chart
except ModuleNotFoundError:
    from benchmarks.profile_utils import BenchmarkCase, BenchmarkSummary, benchmark_forward, profile_forward, profile_forward_with_timeline, write_reports, write_timeline_chart


class TorchBaselineBlock(nn.Module):
    """教学用基线 block。

    它不是论文里的 Mamba，只用于说明普通 PyTorch 算子在 profiler 里长什么样。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_cases(quick: bool) -> list[BenchmarkCase]:
    """小规模输入，避免 8GB 显存机器压力过大。"""

    if quick:
        return [BenchmarkCase("quick_b1_l128_d256", batch_size=1, seq_len=128, d_model=256)]
    return [
        BenchmarkCase("small_b1_l128_d256", batch_size=1, seq_len=128, d_model=256),
        BenchmarkCase("medium_b2_l512_d512", batch_size=2, seq_len=512, d_model=512),
    ]


def try_build_mamba_block(d_model: int, device: torch.device) -> tuple[str, nn.Module | None, str]:
    """尝试构造官方 mamba-ssm 的 Mamba block。"""

    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except Exception as exc:
        return "mamba_ssm_mamba", None, f"无法导入官方 Mamba: {exc}"

    try:
        block = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False).to(device).eval()
        return "mamba_ssm_mamba", block, "官方 mamba-ssm Mamba block；使用非 fused fallback 路径"
    except Exception as exc:
        return "mamba_ssm_mamba", None, f"无法构造官方 Mamba: {exc}"


def try_build_mamba2_block(d_model: int, device: torch.device) -> tuple[str, nn.Module | None, str]:
    """尝试构造官方 mamba-ssm 的 Mamba2 block。"""

    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except Exception as exc:
        return "mamba_ssm_mamba2", None, f"无法导入官方 Mamba2: {exc}"

    try:
        block = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2, use_mem_eff_path=False).to(device).eval()
        return "mamba_ssm_mamba2", block, "官方 mamba-ssm Mamba2 block；使用非 fused fallback 路径"
    except Exception as exc:
        return "mamba_ssm_mamba2", None, f"无法构造官方 Mamba2: {exc}"


def try_build_mamba3_block(d_model: int, device: torch.device) -> tuple[str, nn.Module | None, str]:
    """尝试构造官方 mamba-ssm 的 Mamba3 block。

    Mamba3 要求 d_model * expand 能被 headdim 整除。这里使用较小的 headdim=64，
    适合本项目的小规模 demo，同时避免 8GB 显存机器压力过大。
    """

    try:
        from mamba_ssm.modules.mamba3 import Mamba3
    except Exception as exc:
        return "mamba_ssm_mamba3", None, f"无法导入官方 Mamba3: {exc}"

    try:
        block = Mamba3(d_model=d_model, d_state=64, expand=2, headdim=64, chunk_size=64).to(device).eval()
        return "mamba_ssm_mamba3", block, "官方 mamba-ssm Mamba3 block"
    except Exception as exc:
        return "mamba_ssm_mamba3", None, f"无法构造官方 Mamba3: {exc}"


def build_blocks(d_model: int, device: torch.device) -> list[tuple[str, nn.Module | None, str]]:
    """返回所有要测试的 block。

    Mamba-3 如果没有公开 Python 包，不在这里伪造官方实现。
    """

    baseline = TorchBaselineBlock(d_model).to(device).eval()
    blocks = [("torch_baseline", baseline, "教学基线，不是论文 Mamba")]
    blocks.append(try_build_mamba_block(d_model, device))
    blocks.append(try_build_mamba2_block(d_model, device))
    blocks.append(try_build_mamba3_block(d_model, device))
    return blocks


def run_one_block(
    block_name: str,
    block: nn.Module | None,
    note: str,
    case: BenchmarkCase,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[BenchmarkSummary, object | None, object | None]:
    """运行单个 block，并返回汇总结果和算子表。"""

    if block is None:
        return (
            BenchmarkSummary(block_name, case.name, str(device), case.batch_size, case.seq_len, case.d_model, 0.0, 0.0, "skipped", note),
            None,
            None,
        )

    x = torch.randn(case.batch_size, case.seq_len, case.d_model, device=device)

    def forward_fn(inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return block(inp)

    try:
        avg_time_ms, peak_memory_mb = benchmark_forward(forward_fn, x, warmup=warmup, repeat=repeat)
        if block_name == "mamba_ssm_mamba3":
            op_df, timeline_df = profile_forward_with_timeline(block_name, case, forward_fn, x)
        else:
            op_df = profile_forward(block_name, case, forward_fn, x)
            timeline_df = None
        summary = BenchmarkSummary(
            block_name,
            case.name,
            str(device),
            case.batch_size,
            case.seq_len,
            case.d_model,
            avg_time_ms,
            peak_memory_mb,
            "ok",
            note,
        )
        return summary, op_df, timeline_df
    except Exception as exc:
        summary = BenchmarkSummary(
            block_name,
            case.name,
            str(device),
            case.batch_size,
            case.seq_len,
            case.d_model,
            0.0,
            0.0,
            "failed",
            f"{note}; 运行失败: {exc}",
        )
        return summary, None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="小规模 Mamba block 性能 profiling demo")
    parser.add_argument("--quick", action="store_true", help="只运行一个很小的配置，适合快速验证")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU，即使有 CUDA")
    parser.add_argument("--warmup", type=int, default=5, help="正式计时前的预热次数")
    parser.add_argument("--repeat", type=int, default=20, help="正式计时重复次数")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="报告输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"使用设备: {device}")

    summaries = []
    operator_frames = []
    mamba3_timeline_frames = []
    for case in build_cases(args.quick):
        print(f"运行输入配置: {case}")
        for block_name, block, note in build_blocks(case.d_model, device):
            print(f"  测试 block: {block_name}")
            summary, op_df, timeline_df = run_one_block(block_name, block, note, case, device, args.warmup, args.repeat)
            summaries.append(summary)
            if op_df is not None:
                operator_frames.append(op_df)
            if timeline_df is not None:
                mamba3_timeline_frames.append(timeline_df)

    write_reports(args.report_dir, summaries, operator_frames)
    if mamba3_timeline_frames:
        import pandas as pd

        mamba3_timeline_df = pd.concat(mamba3_timeline_frames, ignore_index=True)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        mamba3_timeline_df.to_csv(args.report_dir / "mamba3_timeline.csv", index=False)
        write_timeline_chart(mamba3_timeline_df, args.report_dir / "mamba3_timeline_gantt.png", "Mamba3 operator timeline (CPU profiler events)")
    print(f"报告已写入: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
