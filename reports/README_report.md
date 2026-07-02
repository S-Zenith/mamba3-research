# Mamba 性能报告

## 汇总结果

| block_name       | case_name          | device   |   batch_size |   seq_len |   d_model |   avg_time_ms |   peak_memory_mb | status   | note                                            |
|:-----------------|:-------------------|:---------|-------------:|----------:|----------:|--------------:|-----------------:|:---------|:------------------------------------------------|
| torch_baseline   | quick_b1_l128_d256 | cuda     |            1 |       128 |       256 |      0.105626 |          15.6182 | ok       | 教学基线，不是论文 Mamba                                 |
| mamba_ssm_mamba  | quick_b1_l128_d256 | cuda     |            1 |       128 |       256 |      0.517243 |          15.876  | ok       | 官方 mamba-ssm Mamba block；使用非 fused fallback 路径  |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | cuda     |            1 |       128 |       256 |      0.868458 |          15.8882 | ok       | 官方 mamba-ssm Mamba2 block；使用非 fused fallback 路径 |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | cuda     |            1 |       128 |       256 |      1.07096  |          15.8965 | ok       | 官方 mamba-ssm Mamba3 block                       |

## 算子级结果说明

`operator_profile.csv` 中每一行是一个 PyTorch profiler 看到的算子。
`cuda_time_total_us` 表示该算子累计 CUDA 时间，单位是微秒。
`self_cuda_memory_usage_bytes` 表示该算子自身造成的 CUDA 显存变化，单位是字节。
`runtime_breakdown.csv` 是按 block 聚合后的运行时间 breakdown，单位是毫秒。
如果 `runtime_source=cpu_fallback`，表示当前 PyTorch profiler 没有给出 CUDA 时间，图中使用 CPU profiler 时间近似说明各算子的延时占比。
`memory_breakdown.csv` 是按 block 聚合后的显存占用 breakdown，单位是 MB。

## CUDA 时间最高的前 20 个算子

| block_name      | case_name          | operator                                      |   calls |   cpu_time_total_us |   cuda_time_total_us |   self_cpu_memory_usage_bytes |   self_cuda_memory_usage_bytes | input_shapes   |
|:----------------|:-------------------|:----------------------------------------------|--------:|--------------------:|---------------------:|------------------------------:|-------------------------------:|:---------------|
| torch_baseline  | quick_b1_l128_d256 | aten::layer_norm                              |       1 |                2871 |                    0 |                             0 |                          -1024 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::native_layer_norm                       |       1 |                2843 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::empty                                   |       3 |                  57 |                    0 |                             0 |                         132096 |                |
| torch_baseline  | quick_b1_l128_d256 | cudaLaunchKernel                              |       5 |                 152 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::view                                    |       6 |                  36 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::linear                                  |       2 |                 310 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::reshape                                 |       2 |                  14 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::t                                       |       2 |                  34 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::transpose                               |       2 |                  17 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::as_strided                              |       2 |                   5 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::addmm                                   |       2 |                 228 |                    0 |                             0 |                         393216 |                |
| torch_baseline  | quick_b1_l128_d256 | cudaMemsetAsync                               |       1 |                  31 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | [memory]                                      |       3 |                   0 |                    0 |                             0 |                        -655360 |                |
| torch_baseline  | quick_b1_l128_d256 | aten::silu                                    |       1 |                  48 |                    0 |                             0 |                         262144 |                |
| torch_baseline  | quick_b1_l128_d256 | cudaOccupancyMaxActiveBlocksPerMultiprocessor |       1 |                   4 |                    0 |                             0 |                              0 |                |
| torch_baseline  | quick_b1_l128_d256 | cudaDeviceSynchronize                         |       2 |                  10 |                    0 |                             0 |                              0 |                |
| mamba_ssm_mamba | quick_b1_l128_d256 | aten::permute                                 |       7 |                  41 |                    0 |                             0 |                              0 |                |
| mamba_ssm_mamba | quick_b1_l128_d256 | aten::as_strided                              |      19 |                  14 |                    0 |                             0 |                              0 |                |
| mamba_ssm_mamba | quick_b1_l128_d256 | aten::reshape                                 |      10 |                  65 |                    0 |                             0 |                              0 |                |
| mamba_ssm_mamba | quick_b1_l128_d256 | aten::_reshape_alias                          |       5 |                   8 |                    0 |                             0 |                              0 |                |

## 运行时间 breakdown

| block_name       | case_name          | operator                 |   runtime_ms |   total |   percent | operator_category_zh   | operator_name_zh   | operator_explanation_zh                                                           | runtime_source   |
|:-----------------|:-------------------|:-------------------------|-------------:|--------:|----------:|:-----------------------|:-------------------|:----------------------------------------------------------------------------------|:-----------------|
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::matmul             |        4.377 |  10.568 | 41.4175   | 矩阵计算                   | 矩阵乘法               | 通用矩阵乘法接口。在线性层、投影层、注意力或状态空间计算里，经常用它把输入向量乘以权重矩阵。耗时高通常说明大部分计算量来自 dense 矩阵运算。         | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::mm                 |        4.341 |  10.568 | 41.0768   | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | cudaLaunchKernel         |        0.304 |  10.568 |  2.87661  | CUDA 调度                | 启动 CUDA kernel     | CPU 向 GPU 发起一个 kernel 启动请求。它不是模型数学计算本身，但 kernel 很多时，调度开销会在 CPU profiler 中变明显。     | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::linear             |        0.237 |  10.568 |  2.24262  | 神经网络层                  | 线性层                | PyTorch 的全连接层接口，本质是输入乘以权重矩阵，再可选加 bias。Mamba/Mamba2/Mamba3 里很多输入投影、输出投影都来自这个算子。    | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | SelectiveScanFn          |        0.182 |  10.568 |  1.72218  | Mamba 核心               | 选择性扫描              | Mamba1 的核心状态空间扫描函数。它沿序列维度递推状态，是 Mamba 区别于普通 MLP/卷积的重要部分。                          | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | CausalConv1dFn           |        0.146 |  10.568 |  1.38153  | 卷积                     | 卷积相关算子             | 卷积或因果卷积相关操作。Mamba/Mamba2 中短卷积用于混合局部上下文信息。                                         | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::contiguous         |        0.092 |  10.568 |  0.870553 | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::clone              |        0.088 |  10.568 |  0.832702 | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              | cpu_fallback     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | Other                    |        0.801 |  10.568 |  7.57949  | 汇总项                    | 其他算子               | 为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。                        | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::linear             |        4.561 |  15.067 | 30.2715   | 神经网络层                  | 线性层                | PyTorch 的全连接层接口，本质是输入乘以权重矩阵，再可选加 bias。Mamba/Mamba2/Mamba3 里很多输入投影、输出投影都来自这个算子。    | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::matmul             |        4.524 |  15.067 | 30.0259   | 矩阵计算                   | 矩阵乘法               | 通用矩阵乘法接口。在线性层、投影层、注意力或状态空间计算里，经常用它把输入向量乘以权重矩阵。耗时高通常说明大部分计算量来自 dense 矩阵运算。         | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::mm                 |        4.462 |  15.067 | 29.6144   | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | MambaChunkScanCombinedFn |        0.766 |  15.067 |  5.08396  | Mamba2 核心              | 分块扫描融合函数           | Mamba2 的核心分块扫描函数，把序列切成 chunk 来处理状态空间递推，目标是更适合 GPU 并行执行。                           | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | cudaLaunchKernel         |        0.142 |  15.067 |  0.942457 | CUDA 调度                | 启动 CUDA kernel     | CPU 向 GPU 发起一个 kernel 启动请求。它不是模型数学计算本身，但 kernel 很多时，调度开销会在 CPU profiler 中变明显。     | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | LayerNormFn              |        0.134 |  15.067 |  0.889361 | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | CausalConv1dFn           |        0.109 |  15.067 |  0.723435 | 卷积                     | 卷积相关算子             | 卷积或因果卷积相关操作。Mamba/Mamba2 中短卷积用于混合局部上下文信息。                                         | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::reshape            |        0.051 |  15.067 |  0.338488 | 形状变换                   | 重塑形状               | 改变张量形状。多数情况下是视图操作，但如果内存不连续，也可能触发额外拷贝。                                             | cpu_fallback     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | Other                    |        0.318 |  15.067 |  2.11057  | 汇总项                    | 其他算子               | 为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。                        | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::linear             |        3.389 |  13.101 | 25.8683   | 神经网络层                  | 线性层                | PyTorch 的全连接层接口，本质是输入乘以权重矩阵，再可选加 bias。Mamba/Mamba2/Mamba3 里很多输入投影、输出投影都来自这个算子。    | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::matmul             |        3.349 |  13.101 | 25.5629   | 矩阵计算                   | 矩阵乘法               | 通用矩阵乘法接口。在线性层、投影层、注意力或状态空间计算里，经常用它把输入向量乘以权重矩阵。耗时高通常说明大部分计算量来自 dense 矩阵运算。         | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::mm                 |        3.281 |  13.101 | 25.0439   | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | cudaLaunchKernel         |        0.783 |  13.101 |  5.97664  | CUDA 调度                | 启动 CUDA kernel     | CPU 向 GPU 发起一个 kernel 启动请求。它不是模型数学计算本身，但 kernel 很多时，调度开销会在 CPU profiler 中变明显。     | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | _Mamba3Function          |        0.581 |  13.101 |  4.43478  | Mamba3 核心              | Mamba3 自定义 kernel  | Mamba3 的 Triton/TileLang/CUTE 自定义算子，通常负责状态空间递推、旋转位置相关计算或 MIMO/SISO 扫描。            | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | LayerNormFn              |        0.381 |  13.101 |  2.90817  | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::to                 |        0.165 |  13.101 |  1.25945  | 数据转换                   | 类型或设备转换            | 把张量转换 dtype、device 或 memory format。比如 float16/float32 之间转换，或为了 kernel 要求临时转换精度。   | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::_to_copy           |        0.158 |  13.101 |  1.20601  | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              | cpu_fallback     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | Other                    |        1.014 |  13.101 |  7.73987  | 汇总项                    | 其他算子               | 为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。                        | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::layer_norm         |        2.871 |   6.66  | 43.1081   | 归一化                    | 层归一化入口             | LayerNorm 的高层接口，用来把每个 token 的特征归一化，让数值更稳定。它通常会继续调用 native_layer_norm。             | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::native_layer_norm  |        2.843 |   6.66  | 42.6877   | 归一化                    | 层归一化底层实现           | LayerNorm 的底层实现，计算均值、方差并缩放输入。baseline 里它占比高，是因为 baseline block 很小，归一化开销相对明显。      | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::linear             |        0.31  |   6.66  |  4.65465  | 神经网络层                  | 线性层                | PyTorch 的全连接层接口，本质是输入乘以权重矩阵，再可选加 bias。Mamba/Mamba2/Mamba3 里很多输入投影、输出投影都来自这个算子。    | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::addmm              |        0.228 |   6.66  |  3.42342  | 矩阵计算                   | 加偏置矩阵乘法            | 执行 bias + input @ weight 这类融合形式，常见于 nn.Linear。它同时包含矩阵乘法和加法，所以既会占用计算时间，也会申请输出张量显存。 | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | cudaLaunchKernel         |        0.152 |   6.66  |  2.28228  | CUDA 调度                | 启动 CUDA kernel     | CPU 向 GPU 发起一个 kernel 启动请求。它不是模型数学计算本身，但 kernel 很多时，调度开销会在 CPU profiler 中变明显。     | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::empty              |        0.057 |   6.66  |  0.855856 | 显存分配                   | 申请未初始化张量           | 只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。                     | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::silu               |        0.048 |   6.66  |  0.720721 | 激活函数                   | SiLU 激活            | SiLU，也叫 Swish，形式大致是 x * sigmoid(x)。Mamba 系列里常用它作为门控或卷积后的非线性激活。                    | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | aten::view               |        0.036 |   6.66  |  0.540541 | 形状变换                   | 视图变形               | 不复制数据，只用新的形状解释同一块内存。通常耗时和显存都很小。                                                   | cpu_fallback     |
| torch_baseline   | quick_b1_l128_d256 | Other                    |        0.115 |   6.66  |  1.72673  | 汇总项                    | 其他算子               | 为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。                        | cpu_fallback     |

## 显存占用 breakdown

| block_name       | case_name          | operator            |   memory_mb |    total |    percent | operator_category_zh   | operator_name_zh   | operator_explanation_zh                                                           |
|:-----------------|:-------------------|:--------------------|------------:|---------:|-----------:|:-----------------------|:-------------------|:----------------------------------------------------------------------------------|
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::mm            | 0.898438    | 1.78906  | 50.2183    | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::empty_strided | 0.75        | 1.78906  | 41.9214    | 显存分配                   | 按步幅申请张量            | 申请带有特定 stride 布局的张量。常由转置、reshape、contiguous 或某些 kernel 输出触发，主要反映中间结果的内存分配。        |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::empty         | 0.078125    | 1.78906  |  4.36681   | 显存分配                   | 申请未初始化张量           | 只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。                     |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::exp           | 0.03125     | 1.78906  |  1.74672   | 数学函数                   | 指数函数               | 计算 e 的 x 次方。Mamba 里状态空间参数 A 有时会通过指数形式参数化，用来保证数值范围符合模型设计。                          |
| mamba_ssm_mamba  | quick_b1_l128_d256 | aten::neg           | 0.03125     | 1.78906  |  1.74672   | 数学函数                   | 取负号                | 把张量变成相反数。Mamba 状态空间里常会构造负的 A 参数，使状态衰减而不是发散。                                       |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::empty         | 0.891113    | 2.146    | 41.5245    | 显存分配                   | 申请未初始化张量           | 只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。                     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::mm            | 0.691406    | 2.146    | 32.2184    | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::empty_strided | 0.5625      | 2.146    | 26.2116    | 显存分配                   | 按步幅申请张量            | 申请带有特定 stride 布局的张量。常由转置、reshape、contiguous 或某些 kernel 输出触发，主要反映中间结果的内存分配。        |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::exp           | 0.000488281 | 2.146    |  0.0227531 | 数学函数                   | 指数函数               | 计算 e 的 x 次方。Mamba 里状态空间参数 A 有时会通过指数形式参数化，用来保证数值范围符合模型设计。                          |
| mamba_ssm_mamba2 | quick_b1_l128_d256 | aten::neg           | 0.000488281 | 2.146    |  0.0227531 | 数学函数                   | 取负号                | 把张量变成相反数。Mamba 状态空间里常会构造负的 A 参数，使状态衰减而不是发散。                                       |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::mm            | 0.707031    | 2.18408  | 32.372     | 矩阵计算                   | 二维矩阵乘法             | 专门处理二维矩阵相乘的底层算子，常由 Linear、matmul 或权重投影触发。它通常是神经网络里最主要的浮点计算来源。                     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::empty_strided | 0.658203    | 2.18408  | 30.1364    | 显存分配                   | 按步幅申请张量            | 申请带有特定 stride 布局的张量。常由转置、reshape、contiguous 或某些 kernel 输出触发，主要反映中间结果的内存分配。        |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::empty         | 0.654785    | 2.18408  | 29.9799    | 显存分配                   | 申请未初始化张量           | 只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。                     |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::to            | 0.140625    | 2.18408  |  6.43863   | 数据转换                   | 类型或设备转换            | 把张量转换 dtype、device 或 memory format。比如 float16/float32 之间转换，或为了 kernel 要求临时转换精度。   |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::softplus      | 0.0078125   | 2.18408  |  0.357702  | 激活/参数约束                | Softplus           | 把数值平滑地变成正数，常用于保证时间步长 dt 等参数为正。Mamba3 里会用它处理 dt 或 A 相关参数。                          |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::add           | 0.00390625  | 2.18408  |  0.178851  | 张量运算                   | 逐元素加法              | 两个张量逐元素相加。常见于加 bias、残差连接、参数组合或中间结果累加。                                             |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::clamp         | 0.00390625  | 2.18408  |  0.178851  | 其他                     | 未专门标注的算子           | PyTorch profiler 记录到的其他底层操作。可以结合 operator 原名和 input_shapes 进一步定位其来源。              |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | aten::neg           | 0.00390625  | 2.18408  |  0.178851  | 数学函数                   | 取负号                | 把张量变成相反数。Mamba 状态空间里常会构造负的 A 参数，使状态衰减而不是发散。                                       |
| mamba_ssm_mamba3 | quick_b1_l128_d256 | Other               | 0.00390625  | 2.18408  |  0.178851  | 汇总项                    | 其他算子               | 为了让图表可读，排名靠后的许多小算子被合并到 Other。它代表剩余算子的总和，不是单个真实 PyTorch 算子。                        |
| torch_baseline   | quick_b1_l128_d256 | aten::addmm         | 0.375       | 0.750977 | 49.935     | 矩阵计算                   | 加偏置矩阵乘法            | 执行 bias + input @ weight 这类融合形式，常见于 nn.Linear。它同时包含矩阵乘法和加法，所以既会占用计算时间，也会申请输出张量显存。 |
| torch_baseline   | quick_b1_l128_d256 | aten::silu          | 0.25        | 0.750977 | 33.29      | 激活函数                   | SiLU 激活            | SiLU，也叫 Swish，形式大致是 x * sigmoid(x)。Mamba 系列里常用它作为门控或卷积后的非线性激活。                    |
| torch_baseline   | quick_b1_l128_d256 | aten::empty         | 0.125977    | 0.750977 | 16.775     | 显存分配                   | 申请未初始化张量           | 只申请一块张量内存，不填充值。它本身不是数学计算，但在显存 breakdown 中出现较多，说明该模块创建了不少中间张量。                     |
