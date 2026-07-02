# Tiny Causal LM with Pure PyTorch Mamba3 Flow

This demo trains a character-level causal language model under `demo/tiny_causal_lm/`.

It uses a pure PyTorch Mamba3-SISO-style algorithm flow:

```text
token ids
  -> token embedding
  -> 2 x pre-norm residual Mamba3-SISO-style blocks
  -> final RMSNorm
  -> tied LM head
  -> next-token cross entropy
```

The model does not call official Mamba3 forward functions or fused official kernels. The goal is a small training loop that we can later extend and modify freely.

## Run

```bash
.venv/bin/python demo/tiny_causal_lm/tiny_mamba3_lm.py
```

Run the expanded preset with validation/test evaluation:

```bash
.venv/bin/python demo/tiny_causal_lm/tiny_mamba3_lm.py --preset small --steps 500 --batch-size 16 --eval-interval 100
```

The latest expanded run used 1000 steps:

```bash
.venv/bin/python demo/tiny_causal_lm/tiny_mamba3_lm.py --preset small --steps 1000 --batch-size 16 --eval-interval 100 --eval-batches 10
```

## Outputs

Generated files are written to `demo/tiny_causal_lm/outputs/`:

- `loss_curve.png`: training loss curve.
- `loss_curve_after50_log.png`: log-scale loss curve after step 50, easier to inspect later training dynamics.
- `perplexity_curve.png`: validation perplexity curve.
- `train_log.json`: config, parameter count, and per-step losses.
- `metrics.json`: train/validation/test losses and perplexities.
- `generations.txt`: text generated before and after training.
- `tiny_mamba3_lm.pt`: checkpoint with model state and character vocabulary.
- `report.md`: detailed Chinese training report.

## Presets

Default `tiny` preset:

```text
seq_len=64
d_model=64
n_layer=2
expand=2
headdim=32
chunk_size=16
```

Expanded `small` preset:

```text
seq_len=128
d_model=128
n_layer=4
d_state=32
expand=2
headdim=32
chunk_size=32
```

The script downloads Tiny Shakespeare to `data/tiny_shakespeare.txt` when possible, then creates deterministic 90/5/5 train/validation/test splits. If download fails, it uses an expanded built-in text fallback and records that in the report.
