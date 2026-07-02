# Mamba Profiling Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small-scale, runnable Mamba profiling demo that installs dependencies, benchmarks available Mamba blocks, records operator-level profiler data, generates charts, and explains the workflow in Chinese.

**Architecture:** Keep the project as a lightweight script-based benchmark. `benchmarks/profile_utils.py` owns profiling and report writing; `benchmarks/benchmark_mamba_blocks.py` owns model discovery, benchmark execution, and CLI flow; `scripts/setup_env.sh` owns environment setup; Markdown files explain the papers, limitations, and results.

**Tech Stack:** Python 3.10, PyTorch, `mamba-ssm`, pandas, matplotlib, pypdf, PyTorch profiler, Bash virtualenv setup.

---

## File Structure

- Create `requirements.txt`: Python dependencies for benchmark, plotting, and PDF extraction.
- Create `scripts/setup_env.sh`: Creates `.venv`, upgrades pip, installs dependencies, and prints next commands.
- Create `benchmarks/profile_utils.py`: Chinese-commented helper functions for timing, PyTorch profiler export, Markdown report writing, and chart generation.
- Create `benchmarks/benchmark_mamba_blocks.py`: Chinese-commented CLI benchmark entrypoint.
- Create `docs/paper_notes.md`: Chinese beginner-friendly summary of Mamba, Mamba-2, and Mamba-3 plus scope limitations.
- Create `README.md`: Chinese guide for setup, running, reading reports, and troubleshooting.
- Create `reports/.gitkeep`: Keeps report directory present before generated files exist.

## Task 1: Dependency And Environment Files

**Files:**
- Create: `requirements.txt`
- Create: `scripts/setup_env.sh`
- Create: `reports/.gitkeep`

- [ ] **Step 1: Add dependency list**

Create `requirements.txt` with:

```text
torch
torchvision
torchaudio
mamba-ssm
pandas
matplotlib
pypdf
rich
```

- [ ] **Step 2: Add environment setup script**

Create `scripts/setup_env.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "[1/4] 创建 Python 虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "[2/4] 激活虚拟环境"
source "$VENV_DIR/bin/activate"

echo "[3/4] 升级 pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[4/4] 安装项目依赖"
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "环境安装完成。下一步运行："
echo "source .venv/bin/activate"
echo "python benchmarks/benchmark_mamba_blocks.py --quick"
```

- [ ] **Step 3: Add reports directory marker**

Create empty file `reports/.gitkeep`.

- [ ] **Step 4: Verify setup script syntax**

Run: `bash -n scripts/setup_env.sh`

Expected: command exits with code 0 and prints no syntax errors.

- [ ] **Step 5: Skip commit in this workspace**

Expected: no commit because `/home/myclaw/Projects/AI/mamba` is not a git repository.

## Task 2: Profiling Utilities

**Files:**
- Create: `benchmarks/profile_utils.py`

- [ ] **Step 1: Create profiling helper module**

Create `benchmarks/profile_utils.py` with dataclasses and helper functions:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch


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

    rows = []
    for item in prof.key_averages():
        rows.append(
            {
                "block_name": block_name,
                "case_name": case.name,
                "operator": item.key,
                "calls": item.count,
                "cpu_time_total_us": item.cpu_time_total,
                "cuda_time_total_us": getattr(item, "cuda_time_total", 0.0),
                "self_cpu_memory_usage_bytes": item.self_cpu_memory_usage,
                "self_cuda_memory_usage_bytes": getattr(item, "self_cuda_memory_usage", 0),
                "input_shapes": str(item.input_shapes),
            }
        )
    return pd.DataFrame(rows)


def write_reports(
    report_dir: Path,
    summaries: Iterable[BenchmarkSummary],
    operator_frames: Iterable[pd.DataFrame],
) -> None:
    """保存 CSV、Markdown 和图表。"""

    report_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([s.__dict__ for s in summaries])
    operator_df = pd.concat(list(operator_frames), ignore_index=True) if operator_frames else pd.DataFrame()

    summary_df.to_csv(report_dir / "summary.csv", index=False)
    operator_df.to_csv(report_dir / "operator_profile.csv", index=False)
    write_markdown_report(report_dir / "README_report.md", summary_df, operator_df)
    write_charts(report_dir, summary_df)


def write_markdown_report(path: Path, summary_df: pd.DataFrame, operator_df: pd.DataFrame) -> None:
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
        "",
    ]
    if not operator_df.empty:
        top_ops = operator_df.sort_values("cuda_time_total_us", ascending=False).head(20)
        lines.extend(["## CUDA 时间最高的前 20 个算子", "", top_ops.to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_charts(report_dir: Path, summary_df: pd.DataFrame) -> None:
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
    plt.ylabel("平均前向时间 / ms")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "runtime_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(ok_df["label"], ok_df["peak_memory_mb"])
    plt.ylabel("峰值显存 / MB")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(report_dir / "memory_bar.png", dpi=160)
    plt.close()
```

- [ ] **Step 2: Verify module imports**

Run: `python3 -m py_compile benchmarks/profile_utils.py`

Expected: command exits with code 0.

## Task 3: Benchmark Entrypoint

**Files:**
- Create: `benchmarks/benchmark_mamba_blocks.py`

- [ ] **Step 1: Create benchmark script**

Create `benchmarks/benchmark_mamba_blocks.py` with:

```python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from profile_utils import BenchmarkCase, BenchmarkSummary, benchmark_forward, profile_forward, write_reports


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
        block = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to(device).eval()
        return "mamba_ssm_mamba", block, "官方 mamba-ssm Mamba block"
    except Exception as exc:
        return "mamba_ssm_mamba", None, f"无法构造官方 Mamba: {exc}"


def try_build_mamba2_block(d_model: int, device: torch.device) -> tuple[str, nn.Module | None, str]:
    """尝试构造官方 mamba-ssm 的 Mamba2 block。"""

    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except Exception as exc:
        return "mamba_ssm_mamba2", None, f"无法导入官方 Mamba2: {exc}"

    try:
        block = Mamba2(d_model=d_model, d_state=64, d_conv=4, expand=2).to(device).eval()
        return "mamba_ssm_mamba2", block, "官方 mamba-ssm Mamba2 block"
    except Exception as exc:
        return "mamba_ssm_mamba2", None, f"无法构造官方 Mamba2: {exc}"


def build_blocks(d_model: int, device: torch.device) -> list[tuple[str, nn.Module | None, str]]:
    """返回所有要测试的 block。

    Mamba-3 如果没有公开 Python 包，不在这里伪造官方实现。
    """

    baseline = TorchBaselineBlock(d_model).to(device).eval()
    blocks = [("torch_baseline", baseline, "教学基线，不是论文 Mamba")]
    blocks.append(try_build_mamba_block(d_model, device))
    blocks.append(try_build_mamba2_block(d_model, device))
    blocks.append(("mamba3_unavailable", None, "未发现可直接 pip 安装的官方 Mamba-3 block；报告中仅标注不可用"))
    return blocks


def run_one_block(
    block_name: str,
    block: nn.Module | None,
    note: str,
    case: BenchmarkCase,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> tuple[BenchmarkSummary, object | None]:
    """运行单个 block，并返回汇总结果和算子表。"""

    if block is None:
        return (
            BenchmarkSummary(block_name, case.name, str(device), case.batch_size, case.seq_len, case.d_model, 0.0, 0.0, "skipped", note),
            None,
        )

    x = torch.randn(case.batch_size, case.seq_len, case.d_model, device=device)

    def forward_fn(inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return block(inp)

    try:
        avg_time_ms, peak_memory_mb = benchmark_forward(forward_fn, x, warmup=warmup, repeat=repeat)
        op_df = profile_forward(block_name, case, forward_fn, x)
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
        return summary, op_df
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
        return summary, None


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
    for case in build_cases(args.quick):
        print(f"运行输入配置: {case}")
        for block_name, block, note in build_blocks(case.d_model, device):
            print(f"  测试 block: {block_name}")
            summary, op_df = run_one_block(block_name, block, note, case, device, args.warmup, args.repeat)
            summaries.append(summary)
            if op_df is not None:
                operator_frames.append(op_df)

    write_reports(args.report_dir, summaries, operator_frames)
    print(f"报告已写入: {args.report_dir.resolve()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -m py_compile benchmarks/benchmark_mamba_blocks.py`

Expected: command exits with code 0.

## Task 4: Documentation

**Files:**
- Create: `docs/paper_notes.md`
- Create: `README.md`

- [ ] **Step 1: Add paper notes**

Create `docs/paper_notes.md` with a concise Chinese explanation of the three papers, the demo scope, and the limitation that PDF text extraction may be partial.

- [ ] **Step 2: Add user README**

Create `README.md` with sections: 项目目标, 目录结构, 安装环境, 运行 benchmark, 查看报告, 如何理解算子级 profiling, 已知限制, 常见问题.

- [ ] **Step 3: Verify Markdown files exist**

Run: `python3 - <<'PY'
from pathlib import Path
for path in [Path('README.md'), Path('docs/paper_notes.md')]:
    assert path.exists(), path
    assert path.read_text(encoding='utf-8').strip(), path
print('docs ok')
PY`

Expected output includes: `docs ok`.

## Task 5: Environment Installation And Smoke Test

**Files:**
- No source edits expected unless installation exposes a bug in earlier tasks.

- [ ] **Step 1: Run setup script**

Run: `bash scripts/setup_env.sh`

Expected: `.venv` is created and dependencies install. If `mamba-ssm` installation fails, capture the error and document it in `README.md` under 常见问题.

- [ ] **Step 2: Verify Python and CUDA**

Run: `.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`

Expected: prints PyTorch version and CUDA availability.

- [ ] **Step 3: Verify mamba import**

Run: `.venv/bin/python -c "import mamba_ssm; print('mamba_ssm ok')"`

Expected: prints `mamba_ssm ok`. If it fails, README must describe that official Mamba blocks will be skipped.

## Task 6: Run Benchmark And Generate Reports

**Files:**
- Generated: `reports/summary.csv`
- Generated: `reports/operator_profile.csv`
- Generated: `reports/runtime_bar.png`
- Generated: `reports/memory_bar.png`
- Generated: `reports/README_report.md`

- [ ] **Step 1: Run quick benchmark**

Run: `.venv/bin/python benchmarks/benchmark_mamba_blocks.py --quick`

Expected: prints device, tested blocks, and report directory.

- [ ] **Step 2: Verify report files**

Run: `python3 - <<'PY'
from pathlib import Path
required = [
    'reports/summary.csv',
    'reports/operator_profile.csv',
    'reports/runtime_bar.png',
    'reports/memory_bar.png',
    'reports/README_report.md',
]
for name in required:
    path = Path(name)
    assert path.exists(), name
    assert path.stat().st_size > 0, name
print('reports ok')
PY`

Expected output includes: `reports ok`.

- [ ] **Step 3: Inspect summary statuses**

Run: `.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv('reports/summary.csv')
print(df[['block_name', 'status', 'note']])
PY`

Expected: `torch_baseline` is `ok`; official Mamba rows are either `ok`, `failed`, or `skipped` with explanatory notes.

## Task 7: Final Verification

**Files:**
- Modify docs only if verification finds missing instructions.

- [ ] **Step 1: Run Python compile check**

Run: `python3 -m py_compile benchmarks/profile_utils.py benchmarks/benchmark_mamba_blocks.py`

Expected: exits with code 0.

- [ ] **Step 2: Confirm generated report is readable**

Run: `python3 - <<'PY'
from pathlib import Path
text = Path('reports/README_report.md').read_text(encoding='utf-8')
assert 'Mamba 性能报告' in text
assert '汇总结果' in text
print('report markdown ok')
PY`

Expected output includes: `report markdown ok`.

- [ ] **Step 3: Summarize results for user**

Report which blocks ran successfully, which were skipped, where reports are saved, and any installation/profiling limitations.

## Self-Review

- Spec coverage: Environment setup, official package attempt, benchmark script, operator profiling, CSV/Markdown/PNG reports, Chinese README, and limitations are each covered by tasks.
- Placeholder scan: No TBD or implementation placeholders remain; documentation task names specify required sections even though prose will be written during execution.
- Type consistency: `BenchmarkCase`, `BenchmarkSummary`, `benchmark_forward`, `profile_forward`, and `write_reports` names match across tasks.
