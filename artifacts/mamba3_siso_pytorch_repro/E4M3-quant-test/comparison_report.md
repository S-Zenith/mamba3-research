# E4M3-quant-test

- experiment: `E4M3-quant-test`
- baseline: PyTorch SISO FP32 output, dtype `torch.float32`
- weights/activations/intermediates/states: `torch.float8_e4m3fn`
- arithmetic model: op-level fake quant; each major kernel op output is quantized back to E4M3

| tensor | shape | max_abs | mean_abs | max_rel | rel_l2 |
|---|---:|---:|---:|---:|---:|
| `final_output` | `(1, 128, 256)` | 5.45794249e-01 | 9.38775255e-02 | 3.17850303e+03 | 1.36135542e-01 |
| `kernel_y` | `(1, 128, 8, 64)` | 4.25000000e+00 | 1.05607832e-01 | 2.26709135e+04 | 1.30438955e-01 |
| `out_v` | `(1, 128, 8, 64)` | 5.06037903e+00 | 4.12938509e-01 | 6.23063233e+03 | 1.21138413e-01 |
| `Q_rot` | `(1, 128, 8, 64)` | 1.21093750e+00 | 9.27598178e-02 | 6.25000000e+10 | 1.06811934e-01 |
| `K_scaled` | `(1, 128, 8, 64)` | 1.22192383e-01 | 3.25217890e-03 | 3.90625000e+09 | 1.24582728e-01 |
