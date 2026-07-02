# Mamba3 结构图查看说明

## 为什么 Netron 打不开 state_dict.pt 的结构

`mamba3_demo_state_dict.pt` 只是参数字典，里面只有参数名和张量值，不包含 forward 计算图。
Netron 可以显示部分权重文件的张量，但通常无法从 PyTorch state_dict 推断完整网络结构。

## 已导出的结构文件

- `artifacts/mamba3_model_info/mamba3_module_structure.dot`：Graphviz DOT 格式的模块层级图。

如果系统安装了 Graphviz，可以生成 PNG：

```bash
dot -Tpng artifacts/mamba3_model_info/mamba3_module_structure.dot -o artifacts/mamba3_model_info/mamba3_module_structure.png
```

这个图展示的是 PyTorch Module 层级，不是 Triton kernel 内部计算图。