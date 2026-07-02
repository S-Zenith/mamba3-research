from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "demo" / "tiny_causal_lm" / "outputs" / "tiny_mamba3_lm_highlevel.onnx"


def make_node(op_type: str, inputs: list[str], outputs: list[str], name: str, **attrs):
    return helper.make_node(op_type, inputs, outputs, name=name, domain="mamba.demo", **attrs)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    vocab_size = 65
    seq_len = 128
    d_model = 128
    n_layer = 4
    d_state = 32
    expand = 2
    headdim = 32
    chunk_size = 32

    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, seq_len])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, seq_len, vocab_size])]

    nodes = [
        make_node(
            "TokenEmbedding",
            ["input_ids"],
            ["hidden_0"],
            "token_embedding",
            vocab_size=vocab_size,
            d_model=d_model,
        )
    ]
    for idx in range(n_layer):
        nodes.append(
            make_node(
                "Mamba3ResidualBlock",
                [f"hidden_{idx}"],
                [f"hidden_{idx + 1}"],
                f"mamba3_residual_block_{idx}",
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                mode="pure_pytorch_siso_style",
                contains="RMSNorm + in_proj + Q/K/V split + RoPE + causal SSM mixing + gate + out_proj + residual",
            )
        )
    nodes.extend(
        [
            make_node("RMSNorm", [f"hidden_{n_layer}"], ["normalized"], "final_rms_norm", d_model=d_model),
            make_node(
                "TiedLMHead",
                ["normalized"],
                ["logits"],
                "tied_lm_head",
                vocab_size=vocab_size,
                d_model=d_model,
                tied_to="token_embedding.weight",
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "TinyMamba3LMHighLevel",
        inputs,
        outputs,
        value_info=[
            helper.make_tensor_value_info(f"hidden_{i}", TensorProto.FLOAT, [1, seq_len, d_model])
            for i in range(n_layer + 1)
        ]
        + [helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [1, seq_len, d_model])],
    )
    model = helper.make_model(
        graph,
        producer_name="mamba-demo-highlevel-export",
        opset_imports=[helper.make_operatorsetid("", 17), helper.make_operatorsetid("mamba.demo", 1)],
    )
    model.doc_string = (
        "High-level architecture-only ONNX for Netron inspection. "
        "Custom nodes intentionally summarize the pure PyTorch Mamba3-SISO-style internals. "
        "Use tiny_mamba3_lm.onnx for the fully expanded executable trace."
    )
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
