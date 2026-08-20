# Mamba3 Research

This repository contains small, runnable Mamba/Mamba2/Mamba3 research demos and PyTorch reproductions. The goal is not to reproduce full paper-scale training, but to make the algorithm flow, profiling output, and Mamba3 SISO/MIMO details easy to inspect locally.

## What Is Included

- `benchmarks/`: small forward-pass benchmarks for a PyTorch baseline and available official Mamba blocks.
- `artifacts/mamba3_siso_pytorch_repro/`: pure PyTorch reproduction of the Mamba3 SISO flow, plus quantization experiments.
- `artifacts/mamba3_mimo_pytorch_repro/`: pure PyTorch reproduction of the Mamba3 MIMO flow and comparison helpers.
- `demo/tiny_causal_lm/`: a character-level causal language model using a pure PyTorch Mamba3-SISO-style block.
- `docs/`: notes about paper background, fused/fallback paths, profiler interpretation, and Mamba3 operator mapping.
- `reports/` and `summary/`: generated profiling summaries and Chinese project notes.
- `mamba.pdf`, `mamba2.pdf`, `mamba3.pdf`: local reference papers used while building the demos.

Large generated model checkpoints and ONNX exports are intentionally ignored by Git. Trained checkpoints and intermediate tensors are hosted on Hugging Face: [S-Zenith/mamba3-research](https://huggingface.co/S-Zenith/mamba3-research). Re-run the relevant scripts to regenerate other artifacts.

## Environment

Python 3.10 is recommended. The project pins `torch==2.3.1` because that version is more likely to work with the `mamba-ssm` wheels and CUDA 12.1-era environments used during development.

Create and install the base environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Some scripts compare against official Mamba/Mamba3 kernels. Install `mamba-ssm` separately after PyTorch is installed:

```bash
python -m pip install --no-build-isolation mamba-ssm
```

If `mamba-ssm` cannot be installed, the pure PyTorch demos and many tests still run. Official-kernel comparisons will be skipped or marked unavailable.

## Quick Start

Run the benchmark smoke test:

```bash
python benchmarks/benchmark_mamba_blocks.py --quick
```

Force CPU execution if CUDA or official kernels are unavailable:

```bash
python benchmarks/benchmark_mamba_blocks.py --quick --cpu
```

Run the Mamba3 SISO PyTorch reproduction:

```bash
python artifacts/mamba3_siso_pytorch_repro/mamba3_siso_pytorch_repro.py
```

Run the Mamba3 MIMO PyTorch reproduction:

```bash
python artifacts/mamba3_mimo_pytorch_repro/mamba3_mimo_pytorch_repro.py
```

Train the tiny causal language model:

```bash
python demo/tiny_causal_lm/tiny_mamba3_lm.py
```

Run the test suite:

```bash
pytest
```

## Outputs

Benchmark outputs are written to `reports/`:

- `summary.csv`: runtime, peak memory, and status by block.
- `operator_profile.csv`: PyTorch profiler operator-level details.
- `runtime_bar.png` and `memory_bar.png`: summary charts.
- `README_report.md`: Chinese profiling report.

Mamba3 reproduction outputs are written under their corresponding `artifacts/` subdirectories. Tiny LM outputs are written to `demo/tiny_causal_lm/outputs/`.

## Notes On Profiling

PyTorch profiler memory fields describe allocations from the profiler perspective; they are not hardware memory-bandwidth measurements. Use Nsight Systems or Nsight Compute for lower-level CUDA kernel analysis.

Current scripts favor stable fallback paths when fused official kernels are unavailable or environment-specific. These results are useful for understanding the flow and profiler structure, but they are not paper-performance numbers.

## Repository Hygiene

The repository intentionally ignores:

- `.venv/`, `venv/`, and other local virtual environments.
- `__pycache__/`, `.pytest_cache/`, and editor caches.
- Generated model/export files such as `*.pt`, `*.pth`, and `*.onnx`.

This keeps the GitHub repository focused on source code, documentation, small reports, and reproducible experiment scripts.

## Related Documentation

- `demo/tiny_causal_lm/README.md`: tiny causal LM details and presets.
- `artifacts/mamba3_mimo_pytorch_repro/README.md`: Mamba3 MIMO reproduction notes.
- `docs/paper_notes.md`: background paper notes.
- `docs/fused_vs_fallback_and_cpu_fallback.md`: fused/fallback and `cpu_fallback` explanation.
- `docs/mamba3_operator_mapping.md`: Mamba3 operator and matrix multiplication mapping.
- `summary/README.md`: Chinese phase summary, conclusions, and limitations.

## Limitations

- This is a small research/demo repository, not a full reproduction of Mamba paper training results.
- Official Mamba kernel availability depends on CUDA, compiler, Triton, TileLang, PyTorch, and `mamba-ssm` compatibility.
- Exported checkpoints, when regenerated, are demo or randomly initialized artifacts unless a script explicitly says otherwise.
