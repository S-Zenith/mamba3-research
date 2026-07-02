from __future__ import annotations

from pathlib import Path

import torch


def main() -> None:
    """导出当前 demo 使用的 Mamba3 模型结构和参数。

    这里不是下载论文训练好的权重，而是根据源码实例化一个 Mamba3 block，
    然后保存它的结构摘要和随机初始化参数，方便查看模型包含哪些张量。
    """

    from mamba_ssm.modules.mamba3 import Mamba3

    output_dir = Path("artifacts/mamba3_model_info")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=64, chunk_size=64).eval()
    state_dict_path = output_dir / "mamba3_demo_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)

    lines = [
        "# Mamba3 Demo 模型信息",
        "",
        "## 说明",
        "",
        "这个文件来自源码实例化的 Mamba3 block，不是论文训练后的预训练模型。",
        "保存的 `.pt` 文件是随机初始化参数，只用于查看结构和参数形状。",
        "",
        "## 配置",
        "",
        "```text",
        "d_model = 256",
        "d_state = 64",
        "expand = 2",
        "headdim = 64",
        "chunk_size = 64",
        f"d_inner = {model.d_inner}",
        f"nheads = {model.nheads}",
        f"num_bc_heads = {model.num_bc_heads}",
        f"mimo_rank = {model.mimo_rank}",
        f"num_rope_angles = {model.num_rope_angles}",
        "```",
        "",
        "## 参数形状",
        "",
        "| name | shape | dtype | numel |",
        "|---|---:|---|---:|",
    ]
    total_params = 0
    for name, param in model.state_dict().items():
        numel = param.numel()
        total_params += numel
        lines.append(f"| `{name}` | `{tuple(param.shape)}` | `{param.dtype}` | {numel} |")

    lines.extend(
        [
            "",
            "## 参数总量",
            "",
            f"```text\n{total_params:,} parameters\n```",
            "",
            "## 输出文件",
            "",
            f"- `{state_dict_path}`",
        ]
    )
    (output_dir / "mamba3_model_info.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_dir / 'mamba3_model_info.md'}")
    print(f"wrote {state_dict_path}")


if __name__ == "__main__":
    main()
