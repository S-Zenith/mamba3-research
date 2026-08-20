# WikiText-2 Mamba3 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a WikiText-2 Mamba3 quantization demo with a per-block runtime state limit of 1024 elements, PTQ/fake-quant evaluation, online SwanLab logging, and documentation for ASIC-oriented observations.

**Architecture:** Add a self-contained demo under `demo/wikitext_mamba3_quant/` that reuses the repo's pure PyTorch Mamba3-SISO-style flow but keeps the experiment isolated from `demo/tiny_causal_lm/`. Train one FP32 checkpoint on WikiText-2 with a GPT-2 tokenizer, then run calibration-driven PTQ/fake-quant evaluation for W8A8, W4A8, and W4A16 without introducing real CUDA quant kernels. Write final reports under `docs/`.

**Tech Stack:** Python 3.10, PyTorch 2.3.1, Hugging Face `datasets`, Hugging Face `transformers`, matplotlib, pytest, SwanLab online logging.

**Repository Note:** `/home/myclaw/Projects/AI/mamba` is not a git repository. Replace commit steps with a changed-file summary; do not run `git commit` in this directory.

---

## File Structure

- Create `demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`: CLI entrypoint, data loading, model, training, evaluation, PTQ/fake-quant, SwanLab logging, report writing.
- Create `demo/wikitext_mamba3_quant/README.md`: run commands, model constraints, outputs, SwanLab setup notes.
- Create `tests/test_wikitext_mamba3_quant.py`: fast unit tests for config constraints, tokenizer-free batching, model forward, and quantizers.
- Modify `requirements.txt`: add `datasets`, `transformers`, and `swanlab`.
- Create `docs/quamba_low_bit_quantization_notes.md`: Quamba/Quamba2 summary focused on PTQ ideas relevant to Mamba3 and ASIC.
- Create `docs/mamba3_wikitext_quant_experiment.md`: final experiment log with commands, metrics, SwanLab status, and ASIC observations.
- Output directory created by runtime: `demo/wikitext_mamba3_quant/outputs/`.

## Task 1: Add Dependencies And Demo Skeleton

**Files:**
- Modify: `/home/myclaw/Projects/AI/mamba/requirements.txt`
- Create: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`
- Create: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/README.md`
- Create: `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py`

- [ ] **Step 1: Write failing tests for config constraints**

Create `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py` with this initial content:

```python
from pathlib import Path
import importlib.util

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "wikitext_mamba3_quant" / "train_wikitext_mamba3_quant.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("wikitext_mamba3_quant", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_asic_preset_constraints():
    demo = load_demo_module()
    config = demo.Mamba3LMConfig(vocab_size=128)
    assert config.nheads == 8
    assert config.runtime_state_per_block == 1024
    assert config.runtime_state_total == 4096
    assert config.runtime_state_per_block <= 1024
    assert demo.estimate_block_static_parameters(config) <= 1_000_000


def test_model_forward_shape_small_vocab():
    demo = load_demo_module()
    torch.manual_seed(7)
    config = demo.Mamba3LMConfig(vocab_size=128, d_model=64, n_layer=2, expand=1, headdim=32, d_state=2, seq_len=16)
    model = demo.WikiTextMamba3LM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    loss = demo.compute_lm_loss(logits, input_ids)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests and verify they fail because the script does not exist**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py -q
```

Expected: FAIL with `FileNotFoundError` for `demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`.

- [ ] **Step 3: Add dependency lines**

Append these lines to `/home/myclaw/Projects/AI/mamba/requirements.txt` if they are not already present:

```text
datasets
transformers
swanlab
```

- [ ] **Step 4: Create the demo script skeleton with model constraints**

Create `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py` containing the imports, constants, config, parameter estimator, RMSNorm, RoPE helper, Mamba3 block, residual block, LM model, and loss function. Use the same pure PyTorch algorithm shape as `demo/tiny_causal_lm/tiny_mamba3_lm.py`, with this public API:

```python
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo" / "wikitext_mamba3_quant"
OUTPUT_DIR = DEMO_DIR / "outputs"
SEED = 20260707


@dataclass
class Mamba3LMConfig:
    vocab_size: int
    seq_len: int = 128
    d_model: int = 512
    n_layer: int = 4
    d_state: int = 2
    expand: int = 1
    headdim: int = 64
    chunk_size: int = 64
    dropout: float = 0.05

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def nheads(self) -> int:
        if self.d_inner % self.headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim")
        return self.d_inner // self.headdim

    @property
    def runtime_state_per_block(self) -> int:
        return self.nheads * self.headdim * self.d_state

    @property
    def runtime_state_total(self) -> int:
        return self.n_layer * self.runtime_state_per_block


def estimate_block_static_parameters(config: Mamba3LMConfig) -> int:
    num_rope_angles = max(1, config.d_state // 2)
    in_proj_dim = 2 * config.d_inner + 2 * config.d_state + 3 * config.nheads + num_rope_angles
    in_proj = config.d_model * in_proj_dim
    out_proj = config.d_inner * config.d_model
    norms_and_scan = config.d_model + 2 * config.d_state + 2 * config.nheads
    return in_proj + out_proj + norms_and_scan


def validate_config(config: Mamba3LMConfig) -> None:
    if config.runtime_state_per_block > 1024:
        raise ValueError(f"runtime_state_per_block={config.runtime_state_per_block} exceeds 1024")
    block_params = estimate_block_static_parameters(config)
    if block_params > 1_000_000:
        raise ValueError(f"block_static_parameters={block_params} exceeds 1000000")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


def apply_rope_pairwise(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    pairs = min(angles.shape[-1], x.shape[-1] // 2)
    part = out[..., : pairs * 2].reshape(*out.shape[:-1], pairs, 2)
    x0 = part[..., 0].clone()
    x1 = part[..., 1].clone()
    cos = torch.cos(angles[..., :pairs])
    sin = torch.sin(angles[..., :pairs])
    rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).reshape(*out.shape[:-1], pairs * 2)
    out[..., : pairs * 2] = rotated
    return out


class PureMamba3SISOBlock(nn.Module):
    def __init__(self, config: Mamba3LMConfig) -> None:
        super().__init__()
        validate_config(config)
        self.config = config
        self.d_inner = config.d_inner
        self.nheads = config.nheads
        self.num_rope_angles = max(1, config.d_state // 2)
        in_proj_dim = 2 * self.d_inner + 2 * config.d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(config.d_model, in_proj_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.B_norm = RMSNorm(config.d_state)
        self.C_norm = RMSNorm(config.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = u.shape
        cfg = self.config
        zx = self.in_proj(u)
        z, x, b_raw, c_raw, dd_dt, dd_a, trap, angles = torch.split(
            zx,
            [self.d_inner, self.d_inner, cfg.d_state, cfg.d_state, self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        z = z.view(bsz, seqlen, self.nheads, cfg.headdim)
        v = x.view(bsz, seqlen, self.nheads, cfg.headdim)
        q = self.C_norm(c_raw)
        k = self.B_norm(b_raw)
        a = -F.softplus(dd_a.float()).clamp_min(1e-4)
        dt = F.softplus(dd_dt.float() + self.dt_bias)
        adt = (a * dt).transpose(1, 2)
        angles = torch.tanh(angles.float()).unsqueeze(2).expand(-1, -1, self.nheads, -1) * math.pi
        q = apply_rope_pairwise(q.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles)
        k = apply_rope_pairwise(k.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles)
        time_idx = torch.arange(seqlen, device=u.device)
        causal = time_idx[:, None] >= time_idx[None, :]
        strict_history = time_idx[:, None] > time_idx[None, :]
        outputs: list[torch.Tensor] = []
        for ih in range(self.nheads):
            qh = q[:, :, ih, :].float()
            kh = k[:, :, ih, :].float()
            vh = v[:, :, ih, :].float()
            qk = torch.einsum("bid,bjd->bij", qh, kh) / math.sqrt(cfg.d_state)
            cs = torch.cumsum(adt[:, ih, :], dim=-1)
            decay = torch.exp((cs[:, :, None] - cs[:, None, :]).clamp(min=-20.0, max=5.0))
            scale = dt[:, :, ih] * torch.sigmoid(trap[:, :, ih].float())
            weights = qk * decay * scale[:, None, :]
            weights = torch.where(causal.unsqueeze(0), weights, torch.zeros_like(weights))
            history = torch.where(strict_history.unsqueeze(0), weights, torch.zeros_like(weights))
            diag = torch.diagonal(weights, dim1=1, dim2=2).unsqueeze(-1) * vh
            mixed = torch.einsum("bij,bjp->bip", history, vh) + diag + self.D[ih].float() * vh
            gate = F.silu(z[:, :, ih, :].float())
            outputs.append((mixed * gate).to(u.dtype))
        y = torch.cat(outputs, dim=-1)
        return self.out_proj(self.dropout(y))


class Mamba3ResidualBlock(nn.Module):
    def __init__(self, config: Mamba3LMConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = PureMamba3SISOBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class WikiTextMamba3LM(nn.Module):
    def __init__(self, config: Mamba3LMConfig) -> None:
        super().__init__()
        validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Mamba3ResidualBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and quantize a WikiText-2 Mamba3 LM")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Mamba3LMConfig(vocab_size=128)
    validate_config(config)
    print(json.dumps({"config": asdict(config), "block_static_parameters": estimate_block_static_parameters(config)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create README skeleton**

Create `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/README.md` with:

```markdown
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

## Outputs

Runtime outputs are written to `demo/wikitext_mamba3_quant/outputs/`.
```

- [ ] **Step 6: Run tests and verify Task 1 passes**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py -q
```

Expected: `2 passed`.

- [ ] **Step 7: Record changed files instead of committing**

Record these files in the working notes because `mamba` is not a git repository:

```text
requirements.txt
demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py
demo/wikitext_mamba3_quant/README.md
```

## Task 2: Add WikiText-2 Data Pipeline

**Files:**
- Modify: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`
- Modify: `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py`

- [ ] **Step 1: Add failing tests for token block batching**

Append this test code to `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py`:

```python
def test_token_block_dataset_batches_are_shiftable():
    demo = load_demo_module()
    tokens = list(range(50))
    dataset = demo.TokenBlockDataset(tokens=tokens, seq_len=8, batch_size=4, device=torch.device("cpu"))
    x = dataset.get_batch(start_index=0)
    assert x.shape == (4, 8)
    assert x[0].tolist() == list(range(8))
    assert x[1].tolist() == list(range(8, 16))
    random_x = dataset.sample_batch()
    assert random_x.shape == (4, 8)
    assert random_x.dtype == torch.long
```

- [ ] **Step 2: Run test and verify it fails because `TokenBlockDataset` is undefined**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py::test_token_block_dataset_batches_are_shiftable -q
```

Expected: FAIL with `AttributeError: module 'wikitext_mamba3_quant' has no attribute 'TokenBlockDataset'`.

- [ ] **Step 3: Implement token dataset and optional HF loaders**

Add these functions/classes after `validate_config` in `train_wikitext_mamba3_quant.py`:

```python
class TokenBlockDataset:
    def __init__(self, tokens: list[int], seq_len: int, batch_size: int, device: torch.device) -> None:
        if len(tokens) < seq_len * batch_size + 1:
            raise ValueError("not enough tokens for requested seq_len and batch_size")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        return max(0, (self.tokens.numel() - 1) // self.seq_len)

    def get_batch(self, start_index: int) -> torch.Tensor:
        rows = []
        for row in range(self.batch_size):
            start = start_index + row * self.seq_len
            end = start + self.seq_len
            if end > self.tokens.numel():
                start = max(0, self.tokens.numel() - self.seq_len)
                end = start + self.seq_len
            rows.append(self.tokens[start:end])
        return torch.stack(rows).to(self.device)

    def sample_batch(self) -> torch.Tensor:
        max_start = self.tokens.numel() - self.seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,))
        rows = [self.tokens[int(start) : int(start) + self.seq_len] for start in starts]
        return torch.stack(rows).to(self.device)


def load_wikitext2_tokens(cache_dir: Path, tokenizer_name: str = "gpt2") -> tuple[dict[str, list[int]], int, str]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(cache_dir / "hf"))
    split_tokens: dict[str, list[int]] = {}
    for split in ["train", "validation", "test"]:
        text = "\n\n".join(row["text"] for row in dataset[split] if row["text"].strip())
        split_tokens[split] = tokenizer.encode(text)
    return split_tokens, int(tokenizer.vocab_size), tokenizer_name
```

- [ ] **Step 4: Run token dataset tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py::test_token_block_dataset_batches_are_shiftable -q
```

Expected: `1 passed`.

- [ ] **Step 5: Record changed files instead of committing**

Record:

```text
demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py
```

## Task 3: Add Training, Evaluation, Metrics, And Curves

**Files:**
- Modify: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`
- Modify: `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py`

- [ ] **Step 1: Add tests for perplexity and evaluation on tiny synthetic data**

Append this test:

```python
def test_evaluate_loss_returns_finite_value():
    demo = load_demo_module()
    torch.manual_seed(11)
    device = torch.device("cpu")
    config = demo.Mamba3LMConfig(vocab_size=64, d_model=64, n_layer=1, expand=1, headdim=32, d_state=2, seq_len=12)
    model = demo.WikiTextMamba3LM(config).to(device)
    dataset = demo.TokenBlockDataset(tokens=[i % 64 for i in range(256)], seq_len=12, batch_size=2, device=device)
    loss = demo.evaluate_loss(model, dataset, batches=2)
    assert math.isfinite(loss)
    assert demo.perplexity_from_loss(loss) > 0.0
```

Also add `import math` at the top of the test file.

- [ ] **Step 2: Run the new test and verify it fails because eval helpers are undefined**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py::test_evaluate_loss_returns_finite_value -q
```

Expected: FAIL with `AttributeError` for `evaluate_loss`.

- [ ] **Step 3: Implement evaluation and plotting helpers**

Add these functions after `compute_lm_loss`:

```python
def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(float(loss), 20.0)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataset: TokenBlockDataset, batches: int) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x = dataset.sample_batch()
        logits = model(x)
        losses.append(float(compute_lm_loss(logits, x).item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def save_loss_curve(train_losses: list[float], val_steps: list[int], val_losses: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, linewidth=1.2, label="train loss")
    if val_steps:
        ax.plot(val_steps, val_losses, marker="o", linewidth=1.2, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy loss")
    ax.set_title("WikiText-2 Mamba3 training")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
```

- [ ] **Step 4: Implement `train_fp32`**

Add this function before `main`:

```python
def train_fp32(
    config: Mamba3LMConfig,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    test_dataset: TokenBlockDataset,
    output_dir: Path,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    lr: float,
    swanlab_run: Any | None = None,
) -> tuple[WikiTextMamba3LM, dict[str, Any]]:
    device = train_dataset.device
    model = WikiTextMamba3LM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    val_ppls: list[float] = []
    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        x = train_dataset.sample_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = compute_lm_loss(logits, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss = float(loss.item())
        train_losses.append(train_loss)
        if swanlab_run is not None:
            swanlab_run.log({"train/loss": train_loss, "step": step})
        if step % eval_interval == 0 or step == steps:
            val_loss = evaluate_loss(model, val_dataset, eval_batches)
            val_ppl = perplexity_from_loss(val_loss)
            val_steps.append(step)
            val_losses.append(val_loss)
            val_ppls.append(val_ppl)
            if swanlab_run is not None:
                swanlab_run.log({"val/loss": val_loss, "val/ppl": val_ppl, "step": step})
            print(f"step {step:04d}/{steps} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
    test_loss = evaluate_loss(model, test_dataset, max(eval_batches, 5))
    test_ppl = perplexity_from_loss(test_loss)
    metrics = {
        "config": asdict(config),
        "num_parameters": count_parameters(model),
        "block_static_parameters": estimate_block_static_parameters(config),
        "runtime_state_per_block": config.runtime_state_per_block,
        "runtime_state_total": config.runtime_state_total,
        "steps": steps,
        "eval_interval": eval_interval,
        "eval_batches": eval_batches,
        "lr": lr,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_perplexities": val_ppls,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_ppl": val_ppls[-1],
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "train_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_loss_curve(train_losses, val_steps, val_losses, output_dir / "loss_curve.png")
    (output_dir / "fp32_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"config": asdict(config), "model": model.state_dict()}, output_dir / "fp32_checkpoint.pt")
    return model, metrics
```

- [ ] **Step 5: Run evaluation helper tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py::test_evaluate_loss_returns_finite_value -q
```

Expected: `1 passed`.

- [ ] **Step 6: Record changed files instead of committing**

Record:

```text
demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py
```

## Task 4: Add PTQ/Fake-Quant Evaluation

**Files:**
- Modify: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`
- Modify: `/home/myclaw/Projects/AI/mamba/tests/test_wikitext_mamba3_quant.py`

- [ ] **Step 1: Add tests for symmetric quantizers**

Append this test:

```python
def test_fake_quant_symmetric_preserves_shape_and_bounds():
    demo = load_demo_module()
    x = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]])
    y = demo.fake_quant_symmetric(x, bits=8)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float(y.abs().max()) <= 2.05
    z = demo.fake_quant_symmetric(x, bits=4)
    assert z.shape == x.shape
    assert not torch.equal(y, z)
```

- [ ] **Step 2: Run quantizer test and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py::test_fake_quant_symmetric_preserves_shape_and_bounds -q
```

Expected: FAIL with `AttributeError` for `fake_quant_symmetric`.

- [ ] **Step 3: Implement quantization helpers and quantized linear wrapper**

Add this code after `save_loss_curve`:

```python
def fake_quant_symmetric(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    scale = x.detach().abs().amax().clamp(min=eps) / qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale


def fake_quant_weight_per_output_channel(w: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    scale = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=eps) / qmax
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    return q * scale


def fake_quant_weight_groupwise(w: torch.Tensor, bits: int, group_size: int = 128, eps: float = 1e-8) -> torch.Tensor:
    chunks = []
    for start in range(0, w.shape[1], group_size):
        part = w[:, start : start + group_size]
        chunks.append(fake_quant_weight_per_output_channel(part, bits=bits, eps=eps))
    return torch.cat(chunks, dim=1)


class FakeQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight_bits: int, activation_bits: int | None, group_size: int = 128) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.bias = source.bias
        if weight_bits == 4:
            weight = fake_quant_weight_groupwise(source.weight.detach(), bits=weight_bits, group_size=group_size)
        else:
            weight = fake_quant_weight_per_output_channel(source.weight.detach(), bits=weight_bits)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_bits is not None:
            x = fake_quant_symmetric(x, bits=self.activation_bits)
        y = F.linear(x, self.weight, self.bias)
        if self.activation_bits is not None:
            y = fake_quant_symmetric(y, bits=self.activation_bits)
        return y
```

- [ ] **Step 4: Implement model cloning for PTQ modes**

Add this code after `FakeQuantLinear`:

```python
def replace_linear_with_fake_quant(module: nn.Module, weight_bits: int, activation_bits: int | None) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits))
        else:
            replace_linear_with_fake_quant(child, weight_bits=weight_bits, activation_bits=activation_bits)


def clone_for_quant_eval(model: WikiTextMamba3LM, mode: str) -> WikiTextMamba3LM:
    config = model.config
    cloned = WikiTextMamba3LM(config).to(next(model.parameters()).device)
    cloned.load_state_dict(model.state_dict(), strict=True)
    if mode == "W8A8":
        replace_linear_with_fake_quant(cloned, weight_bits=8, activation_bits=8)
    elif mode == "W4A8":
        replace_linear_with_fake_quant(cloned, weight_bits=4, activation_bits=8)
    elif mode == "W4A16":
        replace_linear_with_fake_quant(cloned, weight_bits=4, activation_bits=None)
    else:
        raise ValueError(f"unknown quant mode: {mode}")
    return cloned.eval()


def evaluate_quant_modes(
    model: WikiTextMamba3LM,
    val_dataset: TokenBlockDataset,
    test_dataset: TokenBlockDataset,
    output_dir: Path,
    baseline_metrics: dict[str, Any],
    eval_batches: int,
    modes: list[str],
    swanlab_run: Any | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    fp32_val = float(baseline_metrics["final_val_ppl"])
    fp32_test = float(baseline_metrics["test_ppl"])
    for mode in modes:
        qmodel = clone_for_quant_eval(model, mode)
        val_loss = evaluate_loss(qmodel, val_dataset, eval_batches)
        test_loss = evaluate_loss(qmodel, test_dataset, max(eval_batches, 5))
        val_ppl = perplexity_from_loss(val_loss)
        test_ppl = perplexity_from_loss(test_loss)
        result = {
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "val_ppl_ratio": val_ppl / fp32_val,
            "test_ppl_ratio": test_ppl / fp32_test,
            "passes_ppl_10pct": val_ppl <= fp32_val * 1.10 and test_ppl <= fp32_test * 1.10,
        }
        results[mode] = result
        if swanlab_run is not None:
            swanlab_run.log({f"quant/{mode}/val_ppl": val_ppl, f"quant/{mode}/test_ppl": test_ppl, f"quant/{mode}/passes_ppl_10pct": int(result["passes_ppl_10pct"])})
        del qmodel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quant_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
```

- [ ] **Step 5: Run all unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Record changed files instead of committing**

Record:

```text
demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py
```

## Task 5: Add SwanLab Online Logging And Full CLI

**Files:**
- Modify: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py`

- [ ] **Step 1: Install dependencies**

Run:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Expected: `datasets`, `transformers`, and `swanlab` install successfully. If installation fails, capture the error and write it into `docs/mamba3_wikitext_quant_experiment.md` in Task 8.

- [ ] **Step 2: Add SwanLab helper**

Add this function before `train_fp32`:

```python
def init_swanlab(enabled: bool, project: str, run_name: str, config: dict[str, Any]) -> Any | None:
    if not enabled:
        return None
    import swanlab

    try:
        return swanlab.init(project=project, experiment_name=run_name, config=config)
    except Exception as exc:
        raise RuntimeError(
            "SwanLab online logging was requested but initialization failed. "
            "Run swanlab login in mamba/.venv or provide the required online credentials. "
            f"Original error: {exc}"
        ) from exc
```

- [ ] **Step 3: Replace `main` with full CLI**

Replace the current `main` body with this CLI behavior:

```python
def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"W8A8", "W4A8", "W4A16"}
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"unknown quant modes: {bad}")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and quantize a WikiText-2 Mamba3 LM")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEMO_DIR / "data")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--quant-modes", default="W8A8,W4A8,W4A16")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="mamba3-wikitext-quant")
    parser.add_argument("--swanlab-run", default="mamba3-asic-ptq")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    if args.smoke:
        config = Mamba3LMConfig(vocab_size=128, seq_len=16, d_model=64, n_layer=1, d_state=2, expand=1, headdim=32, chunk_size=16)
        validate_config(config)
        model = WikiTextMamba3LM(config).to(device)
        x = torch.randint(0, config.vocab_size, (2, config.seq_len), device=device)
        loss = compute_lm_loss(model(x), x)
        print(json.dumps({"smoke_loss": float(loss.item()), "runtime_state_per_block": config.runtime_state_per_block}, indent=2))
        return

    split_tokens, vocab_size, tokenizer_name = load_wikitext2_tokens(args.cache_dir, tokenizer_name=args.tokenizer)
    config = Mamba3LMConfig(vocab_size=vocab_size, seq_len=args.seq_len)
    validate_config(config)
    train_dataset = TokenBlockDataset(split_tokens["train"], args.seq_len, args.batch_size, device)
    val_dataset = TokenBlockDataset(split_tokens["validation"], args.seq_len, args.batch_size, device)
    test_dataset = TokenBlockDataset(split_tokens["test"], args.seq_len, args.batch_size, device)
    run_config = {"config": asdict(config), "tokenizer": tokenizer_name, "device": str(device), "quant_modes": parse_modes(args.quant_modes)}
    swanlab_run = init_swanlab(args.swanlab, args.swanlab_project, args.swanlab_run, run_config)
    model, fp32_metrics = train_fp32(config, train_dataset, val_dataset, test_dataset, args.output_dir, args.steps, args.eval_interval, args.eval_batches, args.lr, swanlab_run)
    quant_metrics = evaluate_quant_modes(model, val_dataset, test_dataset, args.output_dir, fp32_metrics, args.eval_batches, parse_modes(args.quant_modes), swanlab_run)
    summary = {"fp32": fp32_metrics, "quant": quant_metrics}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"fp32_test_ppl": fp32_metrics["test_ppl"], "quant": quant_metrics}, indent=2))
    if swanlab_run is not None:
        swanlab_run.finish()
```

- [ ] **Step 4: Run CLI smoke test**

Run:

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --smoke --device auto
```

Expected: JSON containing `smoke_loss` and `runtime_state_per_block`.

- [ ] **Step 5: Verify SwanLab import**

Run:

```bash
.venv/bin/python - <<'PY'
import importlib.util
print(importlib.util.find_spec('swanlab') is not None)
PY
```

Expected: `True`.

- [ ] **Step 6: Record changed files instead of committing**

Record:

```text
requirements.txt
```

## Task 6: Write Quamba And Quamba2 Notes

**Files:**
- Create: `/home/myclaw/Projects/AI/mamba/docs/quamba_low_bit_quantization_notes.md`

- [ ] **Step 1: Create Quamba/Quamba2 notes**

Create `/home/myclaw/Projects/AI/mamba/docs/quamba_low_bit_quantization_notes.md` with these sections and concrete content:

```markdown
# Quamba 与 Quamba2 低比特量化思路

## 背景

Quamba 和 Quamba2 都针对 Mamba/Selective State Space Models 做后训练量化。核心判断是：Mamba 的误差敏感点不只在线性层，selective scan 的状态递推、门控、指数衰减和激活离群值都会放大量化误差。

## Quamba

Quamba 的重点是给 Mamba1 建立 PTQ recipe。它先用校准数据统计激活范围，再把 RMSNorm/Linear 等边界整理成更适合量化的形式。权重量化支持 W8/W4，激活量化主要围绕 A8，状态和 selective scan 路径有独立 scale，而不是把整块当成普通 Transformer MLP 量化。

关键启发：

- 线性层权重可以低到 8bit 或 4bit，但 selective scan 的输入、状态和输出需要单独校准。
- 状态递推中的 scale 粒度会影响误差累积，per-tensor scale 往往只是基线。
- Hadamard/rotation 和 norm/linear 融合用于平滑激活分布，减少离群值对 INT8/INT4 的压力。

## Quamba2

Quamba2 把框架扩展到 Mamba2 和更大模型，支持 W8A8、W4A8、W4A16、W4AX。它加入 head/channel grouping、reorder、GPTQ、hybrid blocks，并提供真实部署的 latency/memory profiling。

关键启发：

- W4A8 对部分层不稳定时，可以用 W4AX/hybrid blocks 保留敏感层的更高激活精度。
- Mamba2 的 chunk scan/state passing 需要专门 kernel 和 scale 管理。
- 对部署来说，量化格式和 scale 布局必须服务 kernel 数据流；论文级 PTQ 不是只看 checkpoint size。

## 对 Mamba3 和 ASIC 的启发

本项目的 Mamba3 实验先做 fake-quant PTQ，而不是直接写 INT8/INT4 kernel。这样可以先定位：

- 单 block recurrent state SRAM 预算是否可控。
- W8A8 是否在 validation/test perplexity 上接近 FP32。
- W4A8 失效时误差更可能来自权重、激活还是 scan/state 路径。
- ASIC 上 softplus、sigmoid、exp、RoPE、state update 是否需要保留更高精度或查表近似。

后续 ASIC 优化应优先把 state layout、scale storage、scan 累积精度和门控控制路径定义清楚，再决定 INT8/INT4 MAC 阵列和片上 SRAM 分块。
```

- [ ] **Step 2: Record changed files instead of committing**

Record:

```text
docs/quamba_low_bit_quantization_notes.md
```

## Task 7: Run Training And Quantization Experiment

**Files:**
- Runtime outputs: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/outputs/`

- [ ] **Step 1: Run full unit tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run smoke test**

Run:

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --smoke --device auto
```

Expected: JSON includes `runtime_state_per_block` and finite `smoke_loss`.

- [ ] **Step 3: Run SwanLab online training and PTQ evaluation**

Run:

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --steps 300 --batch-size 4 --seq-len 128 --eval-interval 50 --eval-batches 5 --device auto --quant-modes W8A8,W4A8,W4A16 --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq
```

Expected: training logs print every 50 steps, `metrics.json`, `fp32_metrics.json`, `quant_metrics.json`, `fp32_checkpoint.pt`, and `loss_curve.png` are written under `demo/wikitext_mamba3_quant/outputs/`. SwanLab creates an online run. If SwanLab fails because login is missing, run this command without `--swanlab` to complete local metrics, then document the SwanLab failure in Task 8.

- [ ] **Step 4: If the run is too slow, run bounded fallback experiment**

Run only if Step 3 cannot complete in the current session:

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --steps 60 --batch-size 2 --seq-len 96 --eval-interval 20 --eval-batches 2 --device auto --quant-modes W8A8,W4A8 --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq-short
```

Expected: same output file names with shorter training. Mark the report as a short-run result.

- [ ] **Step 5: Inspect metrics JSON**

Run:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('demo/wikitext_mamba3_quant/outputs/metrics.json')
data = json.loads(p.read_text())
print('fp32_val_ppl', data['fp32']['final_val_ppl'])
print('fp32_test_ppl', data['fp32']['test_ppl'])
for name, metrics in data['quant'].items():
    print(name, metrics['val_ppl'], metrics['test_ppl'], metrics['passes_ppl_10pct'])
PY
```

Expected: prints FP32 and quantized validation/test perplexities plus pass/fail booleans.

## Task 8: Write Final Experiment Report

**Files:**
- Create: `/home/myclaw/Projects/AI/mamba/docs/mamba3_wikitext_quant_experiment.md`
- Modify: `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/README.md`

- [ ] **Step 1: Generate experiment report from metrics JSON**

Run this command after Task 7 creates `demo/wikitext_mamba3_quant/outputs/metrics.json`:

```bash
.venv/bin/python - <<'PY'
import json
import os
import platform
from pathlib import Path

import torch

metrics_path = Path('demo/wikitext_mamba3_quant/outputs/metrics.json')
metrics = json.loads(metrics_path.read_text())
fp32 = metrics['fp32']
quant = metrics['quant']
config = fp32['config']
rows = []
for name, item in quant.items():
    rows.append(
        f"| `{name}` | {item['val_loss']:.4f} | {item['val_ppl']:.4f} | "
        f"{item['test_loss']:.4f} | {item['test_ppl']:.4f} | "
        f"{item['val_ppl_ratio']:.4f} | {item['test_ppl_ratio']:.4f} | {item['passes_ppl_10pct']} |"
    )
quant_table = "\n".join(rows) if rows else "| none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |"
cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'
swanlab_status = os.environ.get('SWANLAB_STATUS', 'SwanLab status was not exported; inspect the Task 7 terminal output for the online run URL or login error.')
command = os.environ.get(
    'MAMBA3_RUN_COMMAND',
    '.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --steps 300 --batch-size 4 --seq-len 128 --eval-interval 50 --eval-batches 5 --device auto --quant-modes W8A8,W4A8,W4A16 --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq',
)
doc = f"""# Mamba3 WikiText-2 量化实验记录

## 实验目标

本实验在 `demo/wikitext_mamba3_quant/` 中训练 pure PyTorch Mamba3-SISO-style causal LM，并用 Quamba/Quamba2 风格 PTQ/fake-quant 评估低比特配置。通过标准是量化后 validation/test perplexity 相比 FP32 恶化不超过 10%。

## 模型约束

- `d_model={config['d_model']}`
- `n_layer={config['n_layer']}`
- `expand={config['expand']}`
- `headdim={config['headdim']}`
- `d_state={config['d_state']}`
- 单 block runtime state：`{fp32['runtime_state_per_block']}` elements
- 全模型 runtime state：`{fp32['runtime_state_total']}` elements per batch item
- 单 block 静态参数量：`{fp32['block_static_parameters']}`
- 总参数量：`{fp32['num_parameters']}`

## 环境与命令

- Python：`{platform.python_version()}`
- PyTorch：`{torch.__version__}`
- CUDA available：`{torch.cuda.is_available()}`
- Device：`{cuda_name}`
- Command：`{command}`

## FP32 训练结果

- final train loss：`{fp32['final_train_loss']:.4f}`
- final validation loss：`{fp32['final_val_loss']:.4f}`
- final validation perplexity：`{fp32['final_val_ppl']:.4f}`
- test loss：`{fp32['test_loss']:.4f}`
- test perplexity：`{fp32['test_ppl']:.4f}`
- train seconds：`{fp32['train_seconds']:.2f}`

## PTQ/Fake-Quant 结果

| mode | val loss | val ppl | test loss | test ppl | val ppl ratio | test ppl ratio | pass +10% |
|---|---:|---:|---:|---:|---:|---:|---:|
{quant_table}

## SwanLab 状态

{swanlab_status}

## ASIC 观察

- 单 block state 为 `{fp32['runtime_state_per_block']}` elements，`{config['n_layer']}` 层模型需要 `{fp32['runtime_state_total']}` elements per batch item；硬件约束必须明确 per-block 和 whole-model 两种口径。
- W8A8 如果通过 10% PPL 标准，可作为第一版 INT8 datapath 候选。
- W4A8 如果失败，应优先检查 scan/state、gate、softplus/exp 控制路径，而不是只扩大 weight scale。
- W4A16 如果明显优于 W4A8，说明激活和状态路径比权重更敏感，ASIC 上应考虑混合精度。
"""
out = Path('docs/mamba3_wikitext_quant_experiment.md')
out.write_text(doc, encoding='utf-8')
print(out)
PY
```

Expected: prints `docs/mamba3_wikitext_quant_experiment.md` and writes a report populated from measured JSON values.

- [ ] **Step 2: Update demo README with real run command and output files**

Add these sections to `/home/myclaw/Projects/AI/mamba/demo/wikitext_mamba3_quant/README.md`:

```markdown
## Full Experiment

```bash
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --steps 300 --batch-size 4 --seq-len 128 --eval-interval 50 --eval-batches 5 --device auto --quant-modes W8A8,W4A8,W4A16 --swanlab --swanlab-project mamba3-wikitext-quant --swanlab-run mamba3-asic-ptq
```

## Generated Files

- `fp32_checkpoint.pt`
- `fp32_metrics.json`
- `quant_metrics.json`
- `metrics.json`
- `loss_curve.png`

See `../../docs/mamba3_wikitext_quant_experiment.md` for measured results and ASIC observations.
```

- [ ] **Step 3: Final verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_wikitext_mamba3_quant.py -q
.venv/bin/python demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py --smoke --device auto
```

Expected: tests pass and smoke test prints finite loss.

- [ ] **Step 4: Final changed-file summary**

Record all changed files:

```text
requirements.txt
demo/wikitext_mamba3_quant/README.md
demo/wikitext_mamba3_quant/train_wikitext_mamba3_quant.py
demo/wikitext_mamba3_quant/outputs/fp32_checkpoint.pt
demo/wikitext_mamba3_quant/outputs/fp32_metrics.json
demo/wikitext_mamba3_quant/outputs/quant_metrics.json
demo/wikitext_mamba3_quant/outputs/metrics.json
demo/wikitext_mamba3_quant/outputs/loss_curve.png
docs/quamba_low_bit_quantization_notes.md
docs/mamba3_wikitext_quant_experiment.md
docs/superpowers/specs/2026-07-07-wikitext-mamba3-quant-design.md
docs/superpowers/plans/2026-07-07-wikitext-mamba3-quant.md
tests/test_wikitext_mamba3_quant.py
```

## Self-Review Notes

- Spec coverage: The plan covers independent demo creation, WikiText-2 with GPT-2 tokenizer, per-block state constraint, block static parameter check, FP32 training, PTQ/fake-quant modes, SwanLab online logging, Quamba/Quamba2 notes, final experiment report, and ASIC observations.
- Placeholder scan: The implementation steps avoid deferred work; final report content is generated from measured JSON values.
- Type consistency: Public names used by tests and implementation are consistent: `Mamba3LMConfig`, `runtime_state_per_block`, `runtime_state_total`, `estimate_block_static_parameters`, `TokenBlockDataset`, `WikiTextMamba3LM`, `compute_lm_loss`, `evaluate_loss`, `fake_quant_symmetric`.
