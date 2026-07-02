from __future__ import annotations

from pathlib import Path


def dot_id(name: str) -> str:
    """Graphviz 节点名不能包含很多特殊字符，这里做一个简单替换。"""

    return "node_" + "_".join(part for part in name.replace(".", "_").replace("-", "_").split() if part)


def main() -> None:
    """导出 Mamba3 的模块层级结构图。

    Netron 看不了纯 state_dict 的计算结构；这个脚本改用 Graphviz dot 描述
    PyTorch Module 层级，适合先理解模型有哪些子模块和参数。
    """

    from mamba_ssm.modules.mamba3 import Mamba3

    output_dir = Path("artifacts/mamba3_model_info")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=64, chunk_size=64).eval()

    lines = [
        "digraph Mamba3 {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded];",
    ]
    root_name = "Mamba3"
    lines.append(f'  {dot_id(root_name)} [label="{root_name}"];')

    for name, module in model.named_modules():
        if name == "":
            continue
        parent = root_name if "." not in name else name.rsplit(".", 1)[0]
        label = f"{name}\n{module.__class__.__name__}"
        lines.append(f'  {dot_id(name)} [label="{label}"];')
        lines.append(f"  {dot_id(parent)} -> {dot_id(name)};")

    lines.append("}")
    dot_path = output_dir / "mamba3_module_structure.dot"
    dot_path.write_text("\n".join(lines), encoding="utf-8")

    md_lines = [
        "# Mamba3 结构图查看说明",
        "",
        "## 为什么 Netron 打不开 state_dict.pt 的结构",
        "",
        "`mamba3_demo_state_dict.pt` 只是参数字典，里面只有参数名和张量值，不包含 forward 计算图。",
        "Netron 可以显示部分权重文件的张量，但通常无法从 PyTorch state_dict 推断完整网络结构。",
        "",
        "## 已导出的结构文件",
        "",
        f"- `{dot_path}`：Graphviz DOT 格式的模块层级图。",
        "",
        "如果系统安装了 Graphviz，可以生成 PNG：",
        "",
        "```bash",
        "dot -Tpng artifacts/mamba3_model_info/mamba3_module_structure.dot -o artifacts/mamba3_model_info/mamba3_module_structure.png",
        "```",
        "",
        "这个图展示的是 PyTorch Module 层级，不是 Triton kernel 内部计算图。",
    ]
    (output_dir / "how_to_view_mamba3_structure.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {dot_path}")
    print(f"wrote {output_dir / 'how_to_view_mamba3_structure.md'}")


if __name__ == "__main__":
    main()
