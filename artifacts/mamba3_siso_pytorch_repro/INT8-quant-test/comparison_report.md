# INT8-quant-test

- experiment: `INT8-quant-test`
- baseline: PyTorch SISO FP32 output, dtype `torch.float32`
- quantization: symmetric per-tensor int8
- arithmetic model: op-level fake quant; each major kernel op output is quantized back to INT8

| tensor | shape | max_abs | mean_abs | max_rel | rel_l2 |
|---|---:|---:|---:|---:|---:|
| `final_output` | `(1, 128, 256)` | 3.21251452e-01 | 5.60751993e-02 | 1.05405350e+03 | 7.91999388e-02 |
| `kernel_y` | `(1, 128, 8, 64)` | 1.68508101e+00 | 8.85438158e-02 | 2.15505045e+03 | 7.87656486e-02 |
| `out_v` | `(1, 128, 8, 64)` | 1.44520187e+00 | 2.16246260e-01 | 4.09178764e+03 | 5.67517024e-02 |
| `Q_rot` | `(1, 128, 8, 64)` | 3.12500000e-01 | 2.17647567e-02 | 3.80859375e+10 | 2.09903248e-02 |
| `K_scaled` | `(1, 128, 8, 64)` | 2.63671875e-02 | 1.69600841e-03 | 4.60815430e+09 | 4.93614693e-02 |
