from __future__ import annotations

import json
import math
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo" / "tiny_causal_lm"
OUTPUT_DIR = DEMO_DIR / "outputs"
DATA_DIR = DEMO_DIR / "data"
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEED = 20260611

TRAIN_TEXT = (
    "mamba learns sequences. "
    "tiny models can overfit tiny data. "
    "abc abc abc. "
    "state space models mix history with gates. "
    "mamba learns sequences. "
) * 24


@dataclass
class TinyConfig:
    vocab_size: int
    seq_len: int = 64
    d_model: int = 64
    n_layer: int = 2
    d_state: int = 16
    expand: int = 2
    headdim: int = 32
    chunk_size: int = 16
    dropout: float = 0.0


PRESETS = {
    "tiny": dict(seq_len=64, d_model=64, n_layer=2, d_state=16, expand=2, headdim=32, chunk_size=16, dropout=0.0),
    "small": dict(seq_len=128, d_model=128, n_layer=4, d_state=32, expand=2, headdim=32, chunk_size=32, dropout=0.05),
}


class CharDataset:
    def __init__(self, text: str, seq_len: int, batch_size: int, device: torch.device, chars: list[str] | None = None) -> None:
        chars = sorted(set(text)) if chars is None else chars
        if len(chars) < 2:
            raise ValueError("text must contain at least two unique characters")
        self.text = text
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.data = torch.tensor(self.encode(text), dtype=torch.long, device=device)
        if self.data.numel() <= seq_len + 1:
            raise ValueError("text is too short for requested sequence length")

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(self.itos[int(i)] for i in ids)

    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.data.numel() - self.seq_len - 1, (self.batch_size,), device=self.device)
        x = torch.stack([self.data[start : start + self.seq_len] for start in starts])
        y = torch.stack([self.data[start + 1 : start + self.seq_len + 1] for start in starts])
        return x, y


def split_text(text: str, train_ratio: float = 0.9, val_ratio: float = 0.05) -> tuple[str, str, str]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    n = len(text)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train, val, test = text[:train_end], text[train_end:val_end], text[val_end:]
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("train/val/test split produced an empty split")
    return train, val, test


def load_or_download_text(path: Path, allow_download: bool = True) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8"), f"local:{path}"
    if allow_download:
        try:
            urlretrieve(TINY_SHAKESPEARE_URL, path)
            return path.read_text(encoding="utf-8"), TINY_SHAKESPEARE_URL
        except (URLError, OSError, TimeoutError) as exc:
            fallback = (TRAIN_TEXT * 800)
            return fallback, f"fallback built-in text because download failed: {exc}"
    return TRAIN_TEXT * 800, "fallback built-in text; download disabled"


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
    """Small differentiable Mamba3-SISO-style block implemented only with PyTorch ops."""

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        if config.d_model * config.expand % config.headdim != 0:
            raise ValueError("d_model * expand must be divisible by headdim")
        self.config = config
        self.d_inner = config.d_model * config.expand
        self.nheads = self.d_inner // config.headdim
        self.num_bc_heads = 1
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
            [
                self.d_inner,
                self.d_inner,
                cfg.d_state,
                cfg.d_state,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
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

        outputs: list[torch.Tensor] = []
        time = torch.arange(seqlen, device=u.device)
        causal = time[:, None] >= time[None, :]
        strict_history = time[:, None] > time[None, :]
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
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = PureMamba3SISOBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class TinyMamba3LM(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
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


def compute_loss(model: TinyMamba3LM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


@torch.no_grad()
def evaluate_loss(model: TinyMamba3LM, dataset: CharDataset, batches: int = 20) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = dataset.sample_batch()
        losses.append(float(compute_loss(model, x, y).item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_text(model: TinyMamba3LM, dataset: CharDataset, prompt: str, max_new_tokens: int, temperature: float = 0.8) -> str:
    model.eval()
    ids = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=dataset.device)
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.seq_len :]
        logits = model(context)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat((ids, nxt), dim=1)
    return dataset.decode(ids[0])


def save_loss_curve(
    losses: list[float],
    path: Path,
    val_steps: list[int] | None = None,
    val_losses: list[float] | None = None,
    skip_first: int = 0,
    log_scale: bool = False,
) -> None:
    steps = list(range(1, len(losses) + 1))
    train_pairs = [(step, loss) for step, loss in zip(steps, losses) if step > skip_first]
    train_steps = [step for step, _ in train_pairs]
    train_losses = [loss for _, loss in train_pairs]
    val_pairs = []
    if val_steps and val_losses:
        val_pairs = [(step, loss) for step, loss in zip(val_steps, val_losses) if step > skip_first]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_losses, linewidth=1.5, label="train loss")
    if val_pairs:
        ax.plot([step for step, _ in val_pairs], [loss for _, loss in val_pairs], marker="o", linewidth=1.3, label="val loss")
    title_suffix = f" (after step {skip_first})" if skip_first else ""
    ax.set_title(f"Tiny Mamba3 Causal LM training loss{title_suffix}")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy loss")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_perplexity_curve(steps: list[int], perplexities: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, perplexities, marker="o", linewidth=1.4)
    ax.set_title("Tiny Mamba3 Causal LM validation perplexity")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_config(vocab_size: int, preset: str) -> TinyConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}; expected one of {sorted(PRESETS)}")
    return TinyConfig(vocab_size=vocab_size, **PRESETS[preset])


def write_report(metrics: dict[str, object], before: str, after: str, path: Path) -> None:
    lines = [
        "# Tiny Mamba3 Causal LM 训练报告",
        "",
        "## 任务说明",
        "本实验把字符级语言建模作为端到端任务：给定前面的字符序列，预测下一个字符。模型使用我们自己写的纯 PyTorch Mamba3-SISO-style block，不调用官方 Mamba3 forward 或 fused kernel。",
        "",
        "## 数据集",
        f"- 数据来源：`{metrics['data_source']}`",
        f"- 词表大小：{metrics['vocab_size']}",
        f"- train/val/test 字符数：{metrics['train_chars']} / {metrics['val_chars']} / {metrics['test_chars']}",
        "",
        "## 模型配置",
        f"- preset：`{metrics['preset']}`",
        f"- 参数量：{metrics['num_parameters']}",
        f"- 配置：`{json.dumps(metrics['config'], ensure_ascii=False)}`",
        "",
        "## 训练设置",
        f"- device：`{metrics['device']}`",
        f"- steps：{metrics['steps']}",
        f"- batch size：{metrics['batch_size']}",
        f"- eval interval：{metrics['eval_interval']}",
        "",
        "## 指标",
        f"- 初始 train loss：{metrics['initial_train_loss']:.4f}",
        f"- 最终 train loss：{metrics['final_train_loss']:.4f}",
        f"- 最终 val loss：{metrics['final_val_loss']:.4f}",
        f"- 最终 val perplexity：{metrics['final_val_ppl']:.4f}",
        f"- test loss：{metrics['test_loss']:.4f}",
        f"- test perplexity：{metrics['test_ppl']:.4f}",
        "",
        "## 曲线文件",
        "- `loss_curve.png`：训练 loss 和验证 loss。",
        "- `loss_curve_after50.png`：跳过前 50 step 后的 loss 曲线，便于观察后期趋势。",
        "- `perplexity_curve.png`：验证集 perplexity。",
        "",
        "## 生成效果",
        "### 训练前",
        "```text",
        before,
        "```",
        "",
        "### 训练后",
        "```text",
        after,
        "```",
        "",
        "## 结论",
        "模型在更大的文本数据上完成了 train/validation/test 流程，并输出了验证集和测试集困惑度。这个 demo 的目的不是追求语言模型 SOTA，而是验证复现的 PyTorch Mamba3 block 可以在标准 next-token prediction 任务中训练、验证和测试。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_demo(
    steps: int = 250,
    batch_size: int = 32,
    device: torch.device | None = None,
    preset: str = "tiny",
    eval_interval: int = 50,
    eval_batches: int = 10,
    allow_download: bool = True,
) -> dict[str, object]:
    torch.manual_seed(SEED)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    text, data_source = load_or_download_text(DATA_DIR / "tiny_shakespeare.txt", allow_download=allow_download)
    train_text, val_text, test_text = split_text(text)
    chars = sorted(set(text))
    config = build_config(len(chars), preset)
    train_dataset = CharDataset(train_text, seq_len=config.seq_len, batch_size=batch_size, device=device, chars=chars)
    val_dataset = CharDataset(val_text, seq_len=config.seq_len, batch_size=batch_size, device=device, chars=chars)
    test_dataset = CharDataset(test_text, seq_len=config.seq_len, batch_size=batch_size, device=device, chars=chars)
    model = TinyMamba3LM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    before = generate_text(model, train_dataset, prompt="m", max_new_tokens=240)
    losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    val_ppls: list[float] = []
    model.train()
    for step in range(steps):
        x, y = train_dataset.sample_batch()
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if (step + 1) % eval_interval == 0 or step == steps - 1:
            val_loss = evaluate_loss(model, val_dataset, batches=eval_batches)
            val_ppl = perplexity_from_loss(val_loss)
            val_steps.append(step + 1)
            val_losses.append(val_loss)
            val_ppls.append(val_ppl)
            print(f"step {step + 1:04d}/{steps} train_loss={losses[-1]:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

    test_loss = evaluate_loss(model, test_dataset, batches=max(eval_batches, 20))
    test_ppl = perplexity_from_loss(test_loss)
    after = generate_text(model, train_dataset, prompt="m", max_new_tokens=240)
    save_loss_curve(losses, OUTPUT_DIR / "loss_curve.png", val_steps=val_steps, val_losses=val_losses)
    save_loss_curve(
        losses,
        OUTPUT_DIR / "loss_curve_after50.png",
        val_steps=val_steps,
        val_losses=val_losses,
        skip_first=50,
        log_scale=False,
    )
    save_perplexity_curve(val_steps, val_ppls, OUTPUT_DIR / "perplexity_curve.png")
    log = {
        "config": asdict(config),
        "preset": preset,
        "steps": steps,
        "batch_size": batch_size,
        "eval_interval": eval_interval,
        "device": str(device),
        "num_parameters": count_parameters(model),
        "data_source": data_source,
        "vocab_size": len(chars),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "test_chars": len(test_text),
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "final_val_loss": val_losses[-1],
        "final_val_ppl": val_ppls[-1],
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "loss_curve_skip_first": 50,
        "loss_curve_log_scale": True,
        "train_losses": losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "val_perplexities": val_ppls,
    }
    (OUTPUT_DIR / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "generations.txt").write_text(
        "# Before training\n" + before + "\n\n# After training\n" + after + "\n",
        encoding="utf-8",
    )
    write_report(log, before, after, OUTPUT_DIR / "report.md")
    torch.save({"config": asdict(config), "model": model.state_dict(), "chars": chars}, OUTPUT_DIR / "tiny_mamba3_lm.pt")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure PyTorch Mamba3-style tiny causal LM")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        device = None
    else:
        device = torch.device(args.device)
    log = train_demo(
        steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        preset=args.preset,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        allow_download=not args.no_download,
    )
    print(f"parameters={log['num_parameters']}")
    print(f"initial_train_loss={log['initial_train_loss']:.4f} final_train_loss={log['final_train_loss']:.4f}")
    print(f"test_loss={log['test_loss']:.4f} test_ppl={log['test_ppl']:.2f}")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
