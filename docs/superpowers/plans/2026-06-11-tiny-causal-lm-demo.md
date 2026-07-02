# Tiny Causal LM Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train a minimal character-level causal LM using the reproduced pure PyTorch Mamba3 SISO algorithm flow.

**Architecture:** Create a self-contained demo under `demo/tiny_causal_lm/`. The model uses token embeddings, stacked pre-norm residual Mamba3 SISO blocks implemented in pure PyTorch, final RMSNorm, and a tied LM head. Training uses a small built-in text corpus, saves a loss curve, training log, checkpoint, and before/after generation samples.

**Tech Stack:** Python, PyTorch, matplotlib, pytest.

---

## Files

- Create: `demo/tiny_causal_lm/tiny_mamba3_lm.py`
- Create: `demo/tiny_causal_lm/README.md`
- Create: `tests/test_tiny_causal_lm.py`
- Generated: `demo/tiny_causal_lm/outputs/loss_curve.png`
- Generated: `demo/tiny_causal_lm/outputs/train_log.json`
- Generated: `demo/tiny_causal_lm/outputs/generations.txt`
- Generated: `demo/tiny_causal_lm/outputs/tiny_mamba3_lm.pt`

## Tasks

### Task 1: Tests First

- [ ] Create `tests/test_tiny_causal_lm.py` with tests for tokenizer, forward shape, one training step, and generation.
- [ ] Run `.venv/bin/python -m pytest tests/test_tiny_causal_lm.py -q` and confirm it fails because the demo module does not exist.

### Task 2: Model and Data

- [ ] Create `demo/tiny_causal_lm/tiny_mamba3_lm.py` with `CharDataset`, `RMSNorm`, `PureMamba3SISOBlock`, `Mamba3ResidualBlock`, and `TinyMamba3LM`.
- [ ] Implement a differentiable pure PyTorch SISO kernel. Keep it small and training-oriented; do not call official Mamba3 forward or official fused kernels.
- [ ] Run tests and confirm tokenizer, forward, backward, and generation pass.

### Task 3: Training and Artifacts

- [ ] Add `train_demo()` to run 200-300 steps on built-in text.
- [ ] Save `loss_curve.png`, `train_log.json`, `generations.txt`, and checkpoint.
- [ ] Run the script and confirm artifacts are generated.

### Task 4: Documentation and Verification

- [ ] Add README with run command, model config, outputs, and interpretation.
- [ ] Run pytest.
- [ ] Run demo script.
- [ ] Confirm loss decreases and generation text changes after training.

## Self-Review

- Covers requested path: `demo/tiny_causal_lm/`.
- Uses a tiny causal LM and built-in dataset.
- Uses our PyTorch Mamba3-style flow, not official forward/kernel.
- Saves loss curve and effect demonstration.
- No git commit should be created unless explicitly requested.
