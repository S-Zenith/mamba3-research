# Mamba3 SISO PyTorch Repro Comparison

- device: `cuda`
- official path: `Mamba3.forward`
- PyTorch path: translated SISO forward without calling official fused kernel
- note: 官方 SISO kernel 使用 PTX `tanh.approx.f32/cos.approx.f32/sin.approx.f32`；纯 PyTorch 使用 `torch.tanh/cos/sin`，因此 RoPE 相关张量会有小幅数值差异。
- note: 本脚本没有调用官方 Mamba3 fused forward kernel 来生成 PyTorch 路径，只用官方 block 生成 reference 输出。

| tensor | shape | max_abs | mean_abs | max_rel |
|---|---:|---:|---:|---:|
| `final_output` | `(1, 128, 256)` | 1.30549669e-02 | 1.86940572e-03 | 1.30010782e+02 |
| `kernel_y` | `(1, 128, 8, 64)` | 1.25000000e-01 | 1.29144908e-03 | 2.35611511e+01 |
| `out_v` | `(1, 128, 8, 64)` | 1.15192413e-01 | 7.33394696e-03 | 2.35346859e+01 |
| `Q_rot` | `(1, 128, 8, 64)` | 3.12500000e-02 | 4.56018897e-04 | 1.16470588e+00 |
| `K_scaled` | `(1, 128, 8, 64)` | 3.90625000e-03 | 3.96613881e-05 | 1.33079268e+00 |
| `QK_store` | `(1, 8, 128)` | 2.59447098e-03 | 2.52138106e-04 | 5.41017921e-04 |
| `Scale` | `(1, 8, 128)` | 7.45058060e-09 | 5.39557732e-10 | 2.24536167e-07 |
| `Gamma` | `(1, 8, 128)` | 7.45058060e-09 | 1.84826376e-10 | 2.28456600e-07 |
