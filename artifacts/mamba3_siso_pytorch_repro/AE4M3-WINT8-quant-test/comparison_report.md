# AE4M3-WINT8-quant-test

- experiment: `AE4M3-WINT8-quant-test`
- baseline: PyTorch SISO FP32 output, dtype `torch.float32`
- activations/intermediates/states: `torch.float8_e4m3fn`
- weights: symmetric per-tensor INT8
- arithmetic model: op-level fake quant; each major kernel op output is quantized back to activation format

| tensor | shape | max_abs | mean_abs | max_rel | rel_l2 |
|---|---:|---:|---:|---:|---:|
| `final_output` | `(1, 128, 256)` | 5.86197853e-01 | 8.07727537e-02 | 6.54202111e+03 | 1.17858428e-01 |
| `kernel_y` | `(1, 128, 8, 64)` | 4.25000000e+00 | 9.15481285e-02 | 2.26729135e+04 | 1.14793300e-01 |
| `out_v` | `(1, 128, 8, 64)` | 5.36466217e+00 | 3.55689963e-01 | 9.96961173e+03 | 1.04983958e-01 |
| `Q_rot` | `(1, 128, 8, 64)` | 1.21093750e+00 | 8.84475645e-02 | 4.68214286e+03 | 1.04731862e-01 |
| `K_scaled` | `(1, 128, 8, 64)` | 1.55883789e-01 | 2.69428579e-03 | 3.90625000e+09 | 1.10369882e-01 |
