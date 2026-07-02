# Expanded Causal LM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the tiny causal LM demo into a larger train/validation/test workflow with perplexity evaluation and a detailed Chinese report.

**Architecture:** Keep all work under `demo/tiny_causal_lm/`. Extend the existing pure PyTorch Mamba3-SISO-style model with presets, dataset loading/downloading, deterministic train/val/test splits, evaluation loops, loss/perplexity curves, generations, metrics, and a Chinese report. Continue avoiding official Mamba3 forward and fused kernels.

**Tech Stack:** Python, PyTorch, matplotlib, pytest, urllib standard library.

---

## Files

- Modify: `demo/tiny_causal_lm/tiny_mamba3_lm.py`
- Modify: `demo/tiny_causal_lm/README.md`
- Modify: `tests/test_tiny_causal_lm.py`
- Generated: `demo/tiny_causal_lm/data/tiny_shakespeare.txt`
- Generated: `demo/tiny_causal_lm/outputs/loss_curve.png`
- Generated: `demo/tiny_causal_lm/outputs/perplexity_curve.png`
- Generated: `demo/tiny_causal_lm/outputs/metrics.json`
- Generated: `demo/tiny_causal_lm/outputs/generations.txt`
- Generated: `demo/tiny_causal_lm/outputs/report.md`

## Tasks

### Task 1: Dataset and Splits

- [ ] Add tests for deterministic `split_text`, split lengths, and no empty train/val/test splits.
- [ ] Implement `load_or_download_text`, `split_text`, and split-aware `CharDataset` use.

### Task 2: Training/Evaluation Metrics

- [ ] Add tests for `perplexity_from_loss` and `evaluate_loss` returning finite values.
- [ ] Implement validation and test evaluation, `metrics.json`, and `perplexity_curve.png`.

### Task 3: Larger Presets and CLI

- [ ] Add `tiny` and `small` presets.
- [ ] Add CLI args: `--preset`, `--steps`, `--batch-size`, `--eval-interval`, `--device`.
- [ ] Keep default runtime reasonable for this environment.

### Task 4: Train and Report

- [ ] Run tests.
- [ ] Run `small` training with normal train/val/test flow.
- [ ] Save `report.md` in Chinese with dataset, config, parameter count, train/val/test loss, perplexity, curve paths, and generated examples.

## Self-Review

- Covers bigger model, larger dataset, train/validation/test, perplexity, loss curves, and detailed Chinese report.
- Keeps pure PyTorch Mamba3-style algorithm flow.
- Does not require external package additions.
- Does not create git commits.
