from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    for field_name in (
        "vocab_size",
        "seq_len",
        "d_model",
        "n_layer",
        "d_state",
        "expand",
        "headdim",
        "chunk_size",
    ):
        value = getattr(config, field_name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    if not (0.0 <= config.dropout < 1.0):
        raise ValueError(f"dropout must be >= 0.0 and < 1.0, got {config.dropout!r}")
    if config.d_inner <= 0:
        raise ValueError(f"d_inner must be positive, got {config.d_inner}")
    if config.d_inner % config.headdim != 0:
        raise ValueError("d_model * expand must be divisible by headdim")
    if config.runtime_state_per_block > 1024:
        raise ValueError(f"runtime_state_per_block={config.runtime_state_per_block} exceeds 1024")
    block_params = estimate_block_static_parameters(config)
    if block_params > 1_000_000:
        raise ValueError(f"block_static_parameters={block_params} exceeds 1000000")


class TokenBlockDataset:
    def __init__(self, tokens: list[int], seq_len: int, batch_size: int, device: torch.device) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if len(tokens) < seq_len * batch_size + 1:
            raise ValueError("not enough tokens for requested seq_len and batch_size")
        self.tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        return max(0, (self.tokens.numel() - 1) // (self.seq_len * self.batch_size))

    def get_batch(self, start_index: int) -> torch.Tensor:
        n = self.tokens.numel()
        max_base = max(0, n - (self.batch_size - 1) * self.seq_len - 1)
        base = min(start_index, max_base)
        rows = []
        for row in range(self.batch_size):
            start = base + row * self.seq_len
            end = start + self.seq_len
            if end > n:
                start = max(0, n - self.seq_len)
                end = start + self.seq_len
            rows.append(self.tokens[start:end])
        return torch.stack(rows)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_starts = max(self.tokens.numel() - self.seq_len, 1)
        starts = torch.randint(0, num_starts, (self.batch_size,), device=self.device)
        x_rows = [self.tokens[int(start) : int(start) + self.seq_len] for start in starts]
        y_rows = [self.tokens[int(start) + 1 : int(start) + 1 + self.seq_len] for start in starts]
        x = torch.stack(x_rows)
        y = torch.stack(y_rows)
        return x, y


def load_wikitext2_tokens(cache_dir: Path, tokenizer_name: str = "gpt2") -> tuple[dict[str, list[int]], int, str]:
    import os

    from datasets import load_dataset
    from transformers import AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "wikitext2_tokens.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        return cached["split_tokens"], cached["vocab_size"], cached["tokenizer_name"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=str(cache_dir / "hf"))
    split_tokens: dict[str, list[int]] = {}
    for split in ["train", "validation", "test"]:
        text = "\n\n".join(row["text"] for row in dataset[split] if row["text"].strip())
        split_tokens[split] = tokenizer.encode(text)
    torch.save(
        {"split_tokens": split_tokens, "vocab_size": int(tokenizer.vocab_size), "tokenizer_name": tokenizer_name},
        cache_path,
    )
    return split_tokens, int(tokenizer.vocab_size), tokenizer_name


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


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(float(loss)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate_loss(model: nn.Module, dataset: TokenBlockDataset, batches: int) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        x, y = dataset.sample_batch()
        logits = model(x)
        losses.append(float(compute_lm_loss(logits, x).item()))
    if was_training:
        model.train()
    return sum(losses) / max(1, len(losses))


def save_loss_curve(train_losses: list[float], val_steps: list[int], val_losses: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        x, y = train_dataset.sample_batch()
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


if __name__ == "__main__":
    main()
