# Tiny Mamba3 Causal LM 训练报告

## 任务说明
本实验把字符级语言建模作为端到端任务：给定前面的字符序列，预测下一个字符。模型使用我们自己写的纯 PyTorch Mamba3-SISO-style block，不调用官方 Mamba3 forward 或 fused kernel。

## 数据集
- 数据来源：`local:/home/myclaw/Projects/AI/mamba/demo/tiny_causal_lm/data/tiny_shakespeare.txt`
- 词表大小：65
- train/val/test 字符数：1003854 / 55769 / 55771

## 模型配置
- preset：`small`
- 参数量：455744
- 配置：`{"vocab_size": 65, "seq_len": 128, "d_model": 128, "n_layer": 4, "d_state": 32, "expand": 2, "headdim": 32, "chunk_size": 32, "dropout": 0.05}`

## 训练设置
- device：`cuda`
- steps：1000
- batch size：16
- eval interval：100

## 指标
- 初始 train loss：120.6717
- 最终 train loss：1.7093
- 最终 val loss：1.7385
- 最终 val perplexity：5.6891
- test loss：1.8773
- test perplexity：6.5358

## 曲线文件
- `loss_curve.png`：训练 loss 和验证 loss。
- `loss_curve_after50.png`：跳过前 50 step 后的 loss 曲线，便于观察后期趋势。
- `perplexity_curve.png`：验证集 perplexity。

## 生成效果
### 训练前
```text
mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
```

### 训练后
```text
morow more, have mean her,
And of prophy, and me name the command.

DUKE VINCENTIO:
My but where thine shall done of Norfolk son man;
If her truth, follower more this convers
fure thou further, own thee runced shall queen of the quen.
My dar
```

## 结论
模型在更大的文本数据上完成了 train/validation/test 流程，并输出了验证集和测试集困惑度。这个 demo 的目的不是追求语言模型 SOTA，而是验证复现的 PyTorch Mamba3 block 可以在标准 next-token prediction 任务中训练、验证和测试。
