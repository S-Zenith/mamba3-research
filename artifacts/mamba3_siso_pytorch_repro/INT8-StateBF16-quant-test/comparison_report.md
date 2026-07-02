# INT8-StateBF16-quant-test

- experiment: `INT8-StateBF16-quant-test`
- baseline: PyTorch SISO FP32 output, dtype `torch.float32`
- weights/non-state intermediates: symmetric per-tensor INT8
- recurrent states: BF16 storage
- arithmetic model: op-level fake quant; each major non-state kernel op output is quantized back to INT8

| tensor | shape | max_abs | mean_abs | max_rel | rel_l2 |
|---|---:|---:|---:|---:|---:|
| `final_output` | `(1, 128, 256)` | 3.22030663e-01 | 5.60407331e-02 | 2.10896569e+03 | 7.91212451e-02 |
| `kernel_y` | `(1, 128, 8, 64)` | 1.68514633e+00 | 8.85239123e-02 | 2.15503080e+03 | 7.87518996e-02 |
| `out_v` | `(1, 128, 8, 64)` | 1.44486618e+00 | 2.16209715e-01 | 4.09175040e+03 | 5.67455198e-02 |
| `Q_rot` | `(1, 128, 8, 64)` | 3.12500000e-01 | 2.17647567e-02 | 3.80859375e+10 | 2.09903248e-02 |
| `K_scaled` | `(1, 128, 8, 64)` | 2.63671875e-02 | 1.69600841e-03 | 4.60815430e+09 | 4.93614693e-02 |
