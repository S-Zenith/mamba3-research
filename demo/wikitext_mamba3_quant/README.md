# WikiText-2 Mamba3 Quantization Demo

This demo trains a pure PyTorch Mamba3-SISO-style causal LM on WikiText-2 and evaluates PTQ/fake-quant variants for ASIC-oriented low-bit analysis.

## Constraint

Default preset:

```text
d_model=512
n_layer=4
expand=1
headdim=64
d_state=2
runtime_state_per_block=1024 elements
runtime_state_total=4096 elements for 4 blocks
block_static_parameters<1M
```

## Smoke Test

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --smoke
```

## Full Experiment

If `huggingface.co` is unreachable, set the mirror endpoint:

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py \
  --steps 1500 --batch-size 8 --seq-len 128 --eval-interval 150 --eval-batches 10 \
  --lr 3e-4 --device auto --quant-modes W8A8,W4A8,W4A16 \
  --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq-long
```

## Generated Files

- `fp32_checkpoint.pt`
- `fp32_metrics.json`
- `quant_metrics.json`
- `metrics.json`
- `loss_curve.png`
- `data/wikitext2_tokens.pt` (cached tokenized WikiText-2)

See `../../docs/mamba3_wikitext_quant_experiment.md` for measured results and ASIC observations.

## Outputs

Runtime outputs are written to `demo/wikitext_mamba3_quant/outputs/`.
