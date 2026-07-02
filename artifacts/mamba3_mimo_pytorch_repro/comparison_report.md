# Mamba3 MIMO PyTorch Repro Comparison

- note: PyTorch path keeps the MIMO rank dimension explicit and does not call the official fused MIMO forward kernel.
- note: official TileLang uses mixed precision and fast math; small numerical differences are expected.
- note: SISO rank is effectively 1, while MIMO expands V/Z by rank and then reduces with MIMO_Out.

| tensor | shape | max_abs | mean_abs | max_rel |
|---|---:|---:|---:|---:|
| `final_output` | `(2, 128, 256)` | 1.64049268e-02 | 2.15846378e-03 | 2.31830801e+02 |
