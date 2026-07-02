# Mamba3 MIMO PyTorch Repro

Run:

```bash
python artifacts/mamba3_mimo_pytorch_repro/mamba3_mimo_pytorch_repro.py
```

Outputs:

- `intermediates.pt`: CPU tensors for inputs, prepared kernel inputs, PyTorch intermediates, PyTorch output, and official output when the official MIMO kernel is available.
- `comparison_report.md`: final-output error statistics when the official kernel is available, or a clear unavailable-reference note otherwise.

The script requires CUDA and a working official TileLang MIMO kernel for the final reference comparison. The importable helper tests do not require the official MIMO kernel.

The current checked configuration uses `batch=2`, `ngroups=2`, `mimo_rank=4`, and `chunk_size=8`. This avoids two environment-specific TileLang limitations observed here: a stride check failure for the size-one `G=1` Q/K dimension, and a dynamic shared-memory limit failure at `chunk_size=16`.

## SISO vs MIMO PyTorch Reproduction Differences

SISO has `mimo_rank=1`, so `Q/K` can be treated as a single stream. MIMO keeps rank explicit as `[B, S, R, G, N]`.

MIMO adds `Q_bias/K_bias`, `MIMO_V`, `MIMO_Z`, and `MIMO_Out`. `MIMO_V` expands `V` into rank-specific values, `MIMO_Z` expands the gate, and `MIMO_Out` reduces rank-specific outputs back to `[B, S, H, P]`.

SISO chunk-local attention/state work is over time positions. MIMO flattens each chunk as `chunk_size * mimo_rank`, while causal masking still compares time indices and allows all rank interactions at earlier time steps.

Both paths use `ADT/DT/Trap` for decay and trapezoidal scaling, and both apply RoPE to Q/K. In MIMO, RoPE is applied to every rank copy of Q/K.
