from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_wikitext_mamba3_quant import (
    DEMO_DIR,
    OUTPUT_DIR,
    SEED,
    FakeQuantLinear,
    Mamba3LMConfig,
    PureMamba3SISOBlock,
    RMSNorm,
    TokenBlockDataset,
    WikiTextMamba3LM,
    compute_lm_loss,
    count_parameters,
    estimate_block_static_parameters,
    evaluate_loss,
    fake_quant_symmetric,
    fake_quant_weight_groupwise,
    fake_quant_weight_per_output_channel,
    init_swanlab,
    load_wikitext2_tokens,
    perplexity_from_loss,
    replace_linear_with_fake_quant,
    save_loss_curve,
    validate_config,
)


PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {"d_model": 512, "n_layer": 4, "headdim": 64, "d_state": 2, "expand": 1, "steps": 1500},
    "deep": {"d_model": 512, "n_layer": 8, "headdim": 64, "d_state": 2, "expand": 1, "steps": 1500},
    "widehead": {"d_model": 512, "n_layer": 4, "headdim": 128, "d_state": 2, "expand": 1, "steps": 1500},
    "bigstate": {"d_model": 256, "n_layer": 4, "headdim": 64, "d_state": 4, "expand": 1, "steps": 1500},
    "longtrain": {"d_model": 512, "n_layer": 4, "headdim": 64, "d_state": 2, "expand": 1, "steps": 5000},
    "bigstate-long": {"d_model": 256, "n_layer": 4, "headdim": 64, "d_state": 4, "expand": 1, "steps": 5000},
}


@dataclass
class ExploreConfig:
    vocab_size: int
    seq_len: int = 128
    d_model: int = 512
    n_layer: int = 4
    d_state: int = 2
    expand: int = 1
    headdim: int = 64
    chunk_size: int = 64
    dropout: float = 0.05
    gate_activation: str = "silu"
    scan_activation: str = "softplus"
    rope_mode: str = "trig"
    exp_mode: str = "exp"
    angle_mode: str = "tanh"

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

    def to_mamba_config(self) -> Mamba3LMConfig:
        return Mamba3LMConfig(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            d_model=self.d_model,
            n_layer=self.n_layer,
            d_state=self.d_state,
            expand=self.expand,
            headdim=self.headdim,
            chunk_size=self.chunk_size,
            dropout=self.dropout,
        )


def get_explore_config(preset: str, vocab_size: int, **overrides: Any) -> tuple[ExploreConfig, int]:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}, valid: {list(PRESETS.keys())}")
    params = dict(PRESETS[preset])
    steps = params.pop("steps")
    params.update(overrides)
    params["vocab_size"] = vocab_size
    return ExploreConfig(**params), steps


def approx_exp(x: torch.Tensor) -> torch.Tensor:
    xa = x.clone()
    small = xa.abs() < 1.0
    poly = 1.0 + xa + xa * xa * 0.5
    mid_pos = (xa >= 1.0) & (xa < 3.0)
    mid_neg = (xa <= -1.0) & (xa > -3.0)
    poly_pos = math.e * (1.0 + (xa - 1.0))
    poly_neg = (1.0 / math.e) * (1.0 - (xa + 1.0))
    result = torch.where(small, poly, torch.zeros_like(xa))
    result = torch.where(mid_pos, poly_pos, result)
    result = torch.where(mid_neg, poly_neg, result)
    result = torch.where(xa >= 3.0, torch.tensor(math.e ** 3, dtype=xa.dtype, device=xa.device), result)
    result = torch.where(xa <= -3.0, torch.tensor(math.e ** -3, dtype=xa.dtype, device=xa.device), result)
    return result


def apply_rope_complex(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    pairs = min(angles.shape[-1], x.shape[-1] // 2)
    prefix_shape = x.shape[:-1]
    part = x[..., : pairs * 2].reshape(*prefix_shape, pairs, 2)
    rest = x[..., pairs * 2 :]
    x0 = part[..., 0]
    x1 = part[..., 1]
    theta = angles[..., :pairs]
    cos_approx = 1.0 - theta * theta * 0.5
    sin_approx = theta
    r0 = x0 * cos_approx - x1 * sin_approx
    r1 = x0 * sin_approx + x1 * cos_approx
    rotated = torch.stack((r0, r1), dim=-1).reshape(*prefix_shape, pairs * 2)
    if rest.numel() > 0:
        return torch.cat([rotated, rest], dim=-1)
    return rotated


def apply_rope_rsqrt(x: torch.Tensor, t_vals: torch.Tensor) -> torch.Tensor:
    pairs = min(t_vals.shape[-1], x.shape[-1] // 2)
    prefix_shape = x.shape[:-1]
    part = x[..., : pairs * 2].reshape(*prefix_shape, pairs, 2)
    rest = x[..., pairs * 2 :]
    x0 = part[..., 0]
    x1 = part[..., 1]
    t = t_vals[..., :pairs]
    denom = torch.rsqrt(1.0 + t * t)
    cos_val = denom
    sin_val = t * denom
    r0 = x0 * cos_val - x1 * sin_val
    r1 = x0 * sin_val + x1 * cos_val
    rotated = torch.stack((r0, r1), dim=-1).reshape(*prefix_shape, pairs * 2)
    if rest.numel() > 0:
        return torch.cat([rotated, rest], dim=-1)
    return rotated


class ExploreMamba3Block(nn.Module):
    def __init__(self, config: ExploreConfig) -> None:
        super().__init__()
        self.config = config
        self.d_inner = config.d_inner
        self.nheads = config.nheads
        self.num_rope_angles = max(1, config.d_state // 2)
        mcfg = config.to_mamba_config()
        validate_config(mcfg)
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

        if cfg.scan_activation == "relu":
            a = -F.relu(dd_a.float()).clamp(min=1e-4)
            dt = F.relu(dd_dt.float() + self.dt_bias).clamp(min=1e-4)
        else:
            a = -F.softplus(dd_a.float()).clamp_min(1e-4)
            dt = F.softplus(dd_dt.float() + self.dt_bias)

        adt = (a * dt).transpose(1, 2)
        angles_raw = angles.float()
        if cfg.angle_mode == "rsqrt":
            angles_t = angles_raw.unsqueeze(2).expand(-1, -1, self.nheads, -1)
            q = apply_rope_rsqrt(q.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)
            k = apply_rope_rsqrt(k.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)
        elif cfg.angle_mode == "none":
            q = q.unsqueeze(2).expand(-1, -1, self.nheads, -1)
            k = k.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        else:
            if cfg.angle_mode == "clamp":
                angles_clamped = torch.clamp(angles_raw, -1.0, 1.0)
                angle_scale = math.pi
            elif cfg.angle_mode == "sigmoid":
                angles_clamped = torch.sigmoid(angles_raw) * 2.0 - 1.0
                angle_scale = math.pi
            elif cfg.angle_mode == "linear":
                angles_clamped = angles_raw
                angle_scale = math.pi
            elif cfg.angle_mode == "tanh_half":
                angles_clamped = torch.tanh(angles_raw)
                angle_scale = math.pi / 2
            elif cfg.angle_mode == "tanh_quarter":
                angles_clamped = torch.tanh(angles_raw)
                angle_scale = math.pi / 4
            else:
                angles_clamped = torch.tanh(angles_raw)
                angle_scale = math.pi
            angles_t = angles_clamped.unsqueeze(2).expand(-1, -1, self.nheads, -1) * angle_scale

            if cfg.rope_mode == "complex":
                q = apply_rope_complex(q.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)
                k = apply_rope_complex(k.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)
            else:
                from train_wikitext_mamba3_quant import apply_rope_pairwise
                q = apply_rope_pairwise(q.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)
                k = apply_rope_pairwise(k.unsqueeze(2).expand(-1, -1, self.nheads, -1), angles_t)

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
            diff = (cs[:, :, None] - cs[:, None, :]).clamp(min=-20.0, max=5.0)
            if cfg.exp_mode == "poly":
                decay = approx_exp(diff)
            else:
                decay = torch.exp(diff)
            scale = dt[:, :, ih] * torch.sigmoid(trap[:, :, ih].float())
            weights = qk * decay * scale[:, None, :]
            weights = torch.where(causal.unsqueeze(0), weights, torch.zeros_like(weights))
            history = torch.where(strict_history.unsqueeze(0), weights, torch.zeros_like(weights))
            diag = torch.diagonal(weights, dim1=1, dim2=2).unsqueeze(-1) * vh
            mixed = torch.einsum("bij,bjp->bip", history, vh) + diag + self.D[ih].float() * vh
            if cfg.gate_activation == "relu":
                gate = F.relu(z[:, :, ih, :].float())
            else:
                gate = F.silu(z[:, :, ih, :].float())
            outputs.append((mixed * gate).to(u.dtype))
        y = torch.cat(outputs, dim=-1)
        return self.out_proj(self.dropout(y))


class ExploreResidualBlock(nn.Module):
    def __init__(self, config: ExploreConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = ExploreMamba3Block(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class ExploreLM(nn.Module):
    def __init__(self, config: ExploreConfig) -> None:
        super().__init__()
        validate_config(config.to_mamba_config())
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([ExploreResidualBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class STEQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int) -> torch.Tensor:
        return fake_quant_symmetric(x, bits=bits)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def ste_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    return STEQuant.apply(x, bits)


class QATFakeQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight_bits: int, activation_bits: int | None, group_size: int = 128) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.group_size = group_size
        self.bias = source.bias
        self.weight = nn.Parameter(source.weight.data.clone(), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_bits == 4:
            w_q = fake_quant_weight_groupwise(self.weight, bits=self.weight_bits, group_size=self.group_size)
        else:
            w_q = fake_quant_weight_per_output_channel(self.weight, bits=self.weight_bits)
        if self.activation_bits is not None:
            x = ste_quant(x, bits=self.activation_bits)
        y = F.linear(x, w_q, self.bias)
        if self.activation_bits is not None:
            y = ste_quant(y, bits=self.activation_bits)
        return y


def replace_linear_with_qat(module: nn.Module, weight_bits: int, activation_bits: int | None) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QATFakeQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits))
        else:
            replace_linear_with_qat(child, weight_bits=weight_bits, activation_bits=activation_bits)


def train_model(
    config: ExploreConfig,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    test_dataset: TokenBlockDataset,
    output_dir: Path,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    lr: float,
    swanlab_run: Any | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = train_dataset.device
    model = ExploreLM(config).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
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
    mcfg = config.to_mamba_config()
    metrics = {
        "config": asdict(config),
        "num_parameters": count_parameters(model),
        "block_static_parameters": estimate_block_static_parameters(mcfg),
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
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"config": asdict(config), "model": model.state_dict()}, output_dir / "checkpoint.pt")
    return model, metrics


def train_qat(
    config: ExploreConfig,
    fp32_checkpoint: Path,
    train_dataset: TokenBlockDataset,
    val_dataset: TokenBlockDataset,
    test_dataset: TokenBlockDataset,
    output_dir: Path,
    qat_mode: str,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    lr: float,
    swanlab_run: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = train_dataset.device
    model = ExploreLM(config).to(device)
    ckpt = torch.load(fp32_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded FP32 checkpoint for QAT from {fp32_checkpoint}")

    if qat_mode == "W8A8":
        replace_linear_with_qat(model, weight_bits=8, activation_bits=8)
    elif qat_mode == "W4A16":
        replace_linear_with_qat(model, weight_bits=4, activation_bits=None)
    else:
        raise ValueError(f"unknown qat mode: {qat_mode}")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []
    val_ppls: list[float] = []
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
            print(f"QAT {qat_mode} step {step:04d}/{steps} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
    test_loss = evaluate_loss(model, test_dataset, max(eval_batches, 5))
    test_ppl = perplexity_from_loss(test_loss)
    mcfg = config.to_mamba_config()
    metrics = {
        "qat_mode": qat_mode,
        "config": asdict(config),
        "num_parameters": count_parameters(model),
        "block_static_parameters": estimate_block_static_parameters(mcfg),
        "runtime_state_per_block": config.runtime_state_per_block,
        "runtime_state_total": config.runtime_state_total,
        "steps": steps,
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
    save_loss_curve(train_losses, val_steps, val_losses, output_dir / "qat_loss_curve.png")
    (output_dir / "qat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"config": asdict(config), "model": model.state_dict()}, output_dir / "qat_checkpoint.pt")
    return model, metrics


@torch.no_grad()
def analyze_intermediate_errors(
    fp32_model: nn.Module,
    config: ExploreConfig,
    val_dataset: TokenBlockDataset,
    output_dir: Path,
    modes: list[str],
) -> dict[str, Any]:
    device = next(fp32_model.parameters()).device
    torch.manual_seed(42)
    x, y = val_dataset.sample_batch()

    intermediates_fp32: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                intermediates_fp32[name] = output.detach().clone().float().cpu()
        return hook

    hooks: list[Any] = []
    for i, block in enumerate(fp32_model.blocks):
        mixer = block.mixer
        hooks.append(mixer.in_proj.register_forward_hook(make_hook(f"block{i}_in_proj_out")))
        hooks.append(mixer.out_proj.register_forward_hook(make_hook(f"block{i}_out_proj_out")))
        hooks.append(block.norm.register_forward_hook(make_hook(f"block{i}_norm_out")))

    fp32_model.eval()
    _ = fp32_model(x)
    for h in hooks:
        h.remove()

    results: dict[str, Any] = {}
    for mode in modes:
        qmodel = ExploreLM(config).to(device)
        qmodel.load_state_dict(fp32_model.state_dict(), strict=False)
        if mode == "W8A8":
            replace_linear_with_fake_quant(qmodel, weight_bits=8, activation_bits=8)
        elif mode == "W4A8":
            replace_linear_with_fake_quant(qmodel, weight_bits=4, activation_bits=8)
        elif mode == "W4A16":
            replace_linear_with_fake_quant(qmodel, weight_bits=4, activation_bits=None)
        qmodel = qmodel.eval().to(device)

        intermediates_quant: dict[str, torch.Tensor] = {}
        def make_qhook(name: str):
            def hook(module: nn.Module, inputs: Any, output: Any) -> None:
                if isinstance(output, torch.Tensor):
                    intermediates_quant[name] = output.detach().clone().float().cpu()
            return hook

        qhooks: list[Any] = []
        for i, block in enumerate(qmodel.blocks):
            mixer = block.mixer
            qhooks.append(mixer.in_proj.register_forward_hook(make_qhook(f"block{i}_in_proj_out")))
            qhooks.append(mixer.out_proj.register_forward_hook(make_qhook(f"block{i}_out_proj_out")))
            qhooks.append(block.norm.register_forward_hook(make_qhook(f"block{i}_norm_out")))

        _ = qmodel(x)
        for h in qhooks:
            h.remove()

        mode_errors: dict[str, dict[str, float]] = {}
        for name, fp32_tensor in intermediates_fp32.items():
            if name not in intermediates_quant:
                continue
            q_tensor = intermediates_quant[name]
            if fp32_tensor.shape != q_tensor.shape:
                continue
            diff = fp32_tensor - q_tensor
            mse = float((diff ** 2).mean().item())
            max_abs = float(diff.abs().max().item())
            fp32_mean = float(fp32_tensor.abs().mean().item())
            fp32_max = float(fp32_tensor.abs().max().item())
            rel_error = mse / (fp32_mean ** 2 + 1e-12)
            mode_errors[name] = {
                "mse": mse,
                "max_abs_error": max_abs,
                "fp32_abs_mean": fp32_mean,
                "fp32_abs_max": fp32_max,
                "relative_error": rel_error,
            }
        results[mode] = mode_errors
        del qmodel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "intermediate_error_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba3 exploration: ablation, QAT, intermediate analysis, nonlinear variants")
    parser.add_argument("--mode", choices=["fp32", "qat", "analyze", "nonlinear", "analyze_weights", "ptq_eval", "analyze_activations"], default="fp32")
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEMO_DIR / "data")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--quant-modes", default="W8A8,W4A8,W4A16")
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="mamba3-wikitext-quant")
    parser.add_argument("--swanlab-run", default="explore")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--qat-mode", choices=["W8A8", "W4A16"], default="W8A8")
    parser.add_argument("--qat-steps", type=int, default=1000)
    parser.add_argument("--fp32-checkpoint", type=Path, default=None)
    parser.add_argument("--gate-activation", choices=["silu", "relu"], default="silu")
    parser.add_argument("--scan-activation", choices=["softplus", "relu"], default="softplus")
    parser.add_argument("--rope-mode", choices=["trig", "complex"], default="trig")
    parser.add_argument("--exp-mode", choices=["exp", "poly"], default="exp")
    parser.add_argument("--angle-mode", choices=["tanh", "tanh_half", "tanh_quarter", "clamp", "sigmoid", "linear", "rsqrt", "none"], default="tanh")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    split_tokens, vocab_size, tokenizer_name = load_wikitext2_tokens(args.cache_dir, tokenizer_name=args.tokenizer)

    overrides: dict[str, Any] = {}
    if args.gate_activation != "silu":
        overrides["gate_activation"] = args.gate_activation
    if args.scan_activation != "softplus":
        overrides["scan_activation"] = args.scan_activation
    if args.rope_mode != "trig":
        overrides["rope_mode"] = args.rope_mode
    if args.exp_mode != "exp":
        overrides["exp_mode"] = args.exp_mode
    if args.angle_mode != "tanh":
        overrides["angle_mode"] = args.angle_mode

    config, preset_steps = get_explore_config(args.preset, vocab_size, **overrides)
    steps = args.steps if args.steps > 0 else preset_steps

    run_config = {
        "preset": args.preset,
        "config": asdict(config),
        "mode": args.mode,
        "tokenizer": tokenizer_name,
        "device": str(device),
        "steps": steps,
    }
    swanlab_run = init_swanlab(args.swanlab, args.swanlab_project, args.swanlab_run, run_config)

    train_ds = TokenBlockDataset(split_tokens["train"], args.seq_len, args.batch_size, device)
    val_ds = TokenBlockDataset(split_tokens["validation"], args.seq_len, args.batch_size, device)
    test_ds = TokenBlockDataset(split_tokens["test"], args.seq_len, args.batch_size, device)

    run_dir = args.output_dir / args.swanlab_run
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fp32" or args.mode == "nonlinear":
        model, metrics = train_model(
            config, train_ds, val_ds, test_ds, run_dir,
            steps, args.eval_interval, args.eval_batches, args.lr, swanlab_run,
        )
        quant_metrics = evaluate_quant_modes_standalone(model, config, val_ds, test_ds, run_dir, metrics, args.eval_batches, parse_modes(args.quant_modes), swanlab_run)
        print(json.dumps({"preset": args.preset, "fp32_test_ppl": metrics["test_ppl"], "quant": quant_metrics}, indent=2))

    elif args.mode == "qat":
        if args.fp32_checkpoint is None:
            raise ValueError("--fp32-checkpoint is required for QAT mode")
        model, metrics = train_qat(
            config, args.fp32_checkpoint, train_ds, val_ds, test_ds, run_dir,
            args.qat_mode, args.qat_steps, args.eval_interval, args.eval_batches, args.lr, swanlab_run,
        )
        print(json.dumps({"qat_mode": args.qat_mode, "final_val_ppl": metrics["final_val_ppl"], "test_ppl": metrics["test_ppl"]}, indent=2))

    elif args.mode == "analyze":
        if args.fp32_checkpoint is None:
            raise ValueError("--fp32-checkpoint is required for analyze mode")
        model = ExploreLM(config).to(device)
        ckpt = torch.load(args.fp32_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        results = analyze_intermediate_errors(model, config, val_ds, run_dir, parse_modes(args.quant_modes))
        print(json.dumps({mode: {k: v["relative_error"] for k, v in errs.items()} for mode, errs in results.items()}, indent=2))

    elif args.mode == "analyze_weights":
        if args.fp32_checkpoint is None:
            raise ValueError("--fp32-checkpoint is required for analyze_weights mode")
        model = ExploreLM(config).to(device)
        ckpt = torch.load(args.fp32_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        results = analyze_weight_quantization(model, run_dir)
        for name, info in results.items():
            print(f"\n{name}: shape={info['shape']} fp32_abs_mean={info['fp32_abs_mean']:.4f}")
            for bits in [8, 4]:
                if f"w{bits}" in info:
                    wi = info[f"w{bits}"]
                    print(f"  W{bits}: mse={wi['mse']:.6f} rel_mse={wi['relative_mse']:.6f} max_err={wi['max_abs_error']:.4f} small_zeroed={wi['small_values_zeroed']}/{wi['total_small_values']}")

    elif args.mode == "ptq_eval":
        if args.fp32_checkpoint is None:
            raise ValueError("--fp32-checkpoint is required for ptq_eval mode")
        model = ExploreLM(config).to(device)
        ckpt = torch.load(args.fp32_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        baseline_metrics = {
            "final_val_ppl": perplexity_from_loss(evaluate_loss(model, val_ds, args.eval_batches)),
            "test_ppl": perplexity_from_loss(evaluate_loss(model, test_ds, max(args.eval_batches, 5))),
        }
        print(f"FP32 baseline: val_ppl={baseline_metrics['final_val_ppl']:.2f} test_ppl={baseline_metrics['test_ppl']:.2f}")
        modes = parse_modes(args.quant_modes)
        quant_metrics = evaluate_quant_modes_standalone(
            model, config, val_ds, test_ds, run_dir, baseline_metrics, args.eval_batches, modes, swanlab_run,
        )
        print(json.dumps(quant_metrics, indent=2))

    elif args.mode == "analyze_activations":
        if args.fp32_checkpoint is None:
            raise ValueError("--fp32-checkpoint is required for analyze_activations mode")
        model = ExploreLM(config).to(device)
        ckpt = torch.load(args.fp32_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        results = analyze_activation_quantization(model, val_ds, run_dir)
        for name, stats in sorted(results.items()):
            amp = stats.get("amplification_factor", 0)
            print(f"{name}: mse={stats['mse']:.6f} rel_mse={stats['relative_mse']:.6f} amp={amp:.2f} outlier_err={stats.get('outlier_abs_error',0):.4f} bulk_err={stats.get('bulk_abs_error',0):.4f}")

    if swanlab_run is not None:
        swanlab_run.finish()


class ActQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, activation_bits: int) -> None:
        super().__init__()
        self.activation_bits = activation_bits
        self.bias = source.bias
        self.weight = nn.Parameter(source.weight.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quant_symmetric(x, bits=self.activation_bits)
        y = F.linear(x, self.weight, self.bias)
        y = fake_quant_symmetric(y, bits=self.activation_bits)
        return y


def replace_linear_with_act_quant(module: nn.Module, activation_bits: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, ActQuantLinear(child, activation_bits=activation_bits))
        else:
            replace_linear_with_act_quant(child, activation_bits=activation_bits)


def replace_linear_with_fake_quant_skip_inproj(module: nn.Module, weight_bits: int, activation_bits: int | None) -> None:
    for name, child in list(module.named_children()):
        if name == "in_proj":
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits))
        else:
            replace_linear_with_fake_quant_skip_inproj(child, weight_bits=weight_bits, activation_bits=activation_bits)


def evaluate_quant_modes_standalone(
    model: nn.Module,
    config: ExploreConfig,
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
    device = next(model.parameters()).device
    for mode in modes:
        qmodel = ExploreLM(config).to(device)
        qmodel.load_state_dict(model.state_dict(), strict=False)
        if mode == "W8A8":
            replace_linear_with_fake_quant(qmodel, weight_bits=8, activation_bits=8)
        elif mode == "W8A8_inproj16":
            replace_linear_with_fake_quant_skip_inproj(qmodel, weight_bits=8, activation_bits=8)
        elif mode == "W16A8":
            replace_linear_with_act_quant(qmodel, activation_bits=8)
        elif mode == "W8A16":
            replace_linear_with_fake_quant(qmodel, weight_bits=8, activation_bits=None)
        elif mode == "W4A8":
            replace_linear_with_fake_quant(qmodel, weight_bits=4, activation_bits=8)
        elif mode == "W4A16":
            replace_linear_with_fake_quant(qmodel, weight_bits=4, activation_bits=None)
        elif mode == "W8A8_perchan":
            replace_linear_with_perchan_quant(qmodel, weight_bits=8, activation_bits=8)
        elif mode == "W8A8_pct95":
            replace_linear_with_percentile_quant(qmodel, weight_bits=8, activation_bits=8, percentile=0.95)
        elif mode == "W8A8_pct99":
            replace_linear_with_percentile_quant(qmodel, weight_bits=8, activation_bits=8, percentile=0.99)
        elif mode == "W8A8_pct999":
            replace_linear_with_percentile_quant(qmodel, weight_bits=8, activation_bits=8, percentile=0.999)
        elif mode == "W8A8_pct999_lmhead16":
            replace_linear_with_percentile_quant_skip_lmhead(qmodel, weight_bits=8, activation_bits=8, percentile=0.999)
        qmodel = qmodel.eval()
        val_loss = evaluate_loss(qmodel, val_dataset, eval_batches)
        test_loss = evaluate_loss(qmodel, test_dataset, max(eval_batches, 5))
        val_ppl = perplexity_from_loss(val_loss)
        test_ppl = perplexity_from_loss(test_loss)
        result = {
            "val_loss": val_loss, "val_ppl": val_ppl,
            "test_loss": test_loss, "test_ppl": test_ppl,
            "val_ppl_ratio": val_ppl / fp32_val, "test_ppl_ratio": test_ppl / fp32_test,
            "passes_ppl_10pct": val_ppl <= fp32_val * 1.10 and test_ppl <= fp32_test * 1.10,
        }
        results[mode] = result
        if swanlab_run is not None:
            swanlab_run.log({f"quant/{mode}/val_ppl": val_ppl, f"quant/{mode}/test_ppl": test_ppl})
        del qmodel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output_dir / "quant_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def fake_quant_per_channel_activation(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1])
    scale = x_flat.detach().abs().amax(dim=0, keepdim=True).clamp(min=eps) / qmax
    q = torch.clamp(torch.round(x_flat / scale), qmin, qmax)
    return (q * scale).reshape(orig_shape)


def fake_quant_percentile_activation(x: torch.Tensor, bits: int, percentile: float = 0.999, eps: float = 1e-8) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    abs_x = x.detach().abs().flatten()
    k = max(1, int(abs_x.numel() * (1.0 - percentile)))
    threshold = torch.topk(abs_x, k, largest=True).values[-1].item() if k < abs_x.numel() else abs_x.max().item()
    threshold = max(threshold, eps)
    scale = threshold / qmax
    x_clipped = torch.clamp(x, -threshold, threshold)
    q = torch.clamp(torch.round(x_clipped / scale), qmin, qmax)
    return q * scale


class PerChannelActQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight_bits: int, activation_bits: int) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.bias = source.bias
        if weight_bits == 4:
            weight = fake_quant_weight_groupwise(source.weight.detach(), bits=weight_bits, group_size=128)
        else:
            weight = fake_quant_weight_per_output_channel(source.weight.detach(), bits=weight_bits)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quant_per_channel_activation(x, bits=self.activation_bits)
        y = F.linear(x, self.weight, self.bias)
        y = fake_quant_per_channel_activation(y, bits=self.activation_bits)
        return y


class PercentileActQuantLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight_bits: int, activation_bits: int, percentile: float = 0.999) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.percentile = percentile
        self.bias = source.bias
        if weight_bits == 4:
            weight = fake_quant_weight_groupwise(source.weight.detach(), bits=weight_bits, group_size=128)
        else:
            weight = fake_quant_weight_per_output_channel(source.weight.detach(), bits=weight_bits)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quant_percentile_activation(x, bits=self.activation_bits, percentile=self.percentile)
        y = F.linear(x, self.weight, self.bias)
        y = fake_quant_percentile_activation(y, bits=self.activation_bits, percentile=self.percentile)
        return y


def replace_linear_with_perchan_quant(module: nn.Module, weight_bits: int, activation_bits: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, PerChannelActQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits))
        else:
            replace_linear_with_perchan_quant(child, weight_bits=weight_bits, activation_bits=activation_bits)


def replace_linear_with_percentile_quant(module: nn.Module, weight_bits: int, activation_bits: int, percentile: float = 0.999) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, PercentileActQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits, percentile=percentile))
        else:
            replace_linear_with_percentile_quant(child, weight_bits=weight_bits, activation_bits=activation_bits, percentile=percentile)


def replace_linear_with_percentile_quant_skip_lmhead(module: nn.Module, weight_bits: int, activation_bits: int, percentile: float = 0.999) -> None:
    for name, child in list(module.named_children()):
        if name == "lm_head":
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, PercentileActQuantLinear(child, weight_bits=weight_bits, activation_bits=activation_bits, percentile=percentile))
        else:
            replace_linear_with_percentile_quant_skip_lmhead(child, weight_bits=weight_bits, activation_bits=activation_bits, percentile=percentile)


@torch.no_grad()
def analyze_activation_quantization(
    model: nn.Module,
    val_dataset: TokenBlockDataset,
    output_dir: Path,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = next(model.parameters()).device
    model.eval()
    torch.manual_seed(42)
    x, _ = val_dataset.sample_batch()

    activations: dict[str, torch.Tensor] = {}

    def make_prehook(name: str):
        def hook(module: nn.Module, inputs: tuple) -> None:
            if isinstance(inputs[0], torch.Tensor):
                activations[name] = inputs[0].detach().clone().float().cpu()
        return hook

    hooks: list[Any] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_pre_hook(make_prehook(name)))

    _ = model(x)
    for h in hooks:
        h.remove()

    results: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(activations)
    ncols = 3
    nrows = (n_layers + ncols - 1) // ncols
    fig_scatter, axes_scatter = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes_scatter.flatten() if nrows > 1 else ([axes_scatter] if ncols == 1 else list(axes_scatter))

    for idx, (name, act) in enumerate(sorted(activations.items())):
        act_flat = act.flatten()
        n_sample = min(3000, act_flat.numel())
        perm = torch.randperm(act_flat.numel())[:n_sample]
        sample = act_flat[perm]

        q = fake_quant_symmetric(act, bits=8)
        q_sample = q.flatten()[perm]
        err = (act_flat - q.flatten())[perm]

        stats = {
            "shape": list(act.shape),
            "fp32_abs_mean": float(act.abs().mean()),
            "fp32_abs_max": float(act.abs().max()),
            "fp32_std": float(act.std()),
            "int8_abs_mean": float(q.abs().mean()),
            "mse": float(((act - q) ** 2).mean()),
            "max_abs_error": float((act - q).abs().max()),
            "relative_mse": float(((act - q) ** 2).mean() / ((act ** 2).mean() + 1e-12)),
            "unique_fp32": int(act_flat.unique().numel()),
            "unique_int8": int(q.flatten().unique().numel()),
            "zero_ratio": float((act_flat.abs() < 1e-6).float().mean()),
            "outlier_ratio": float((act_flat.abs() > act_flat.abs().quantile(0.999)).float().mean()),
        }

        p999 = float(act_flat.abs().quantile(0.999))
        p999_error = float((act_flat[act_flat.abs() > p999] - q.flatten()[act_flat.abs() > p999]).abs().mean()) if (act_flat.abs() > p999).any() else 0.0
        bulk_error = float((act_flat[act_flat.abs() <= p999] - q.flatten()[act_flat.abs() <= p999]).abs().mean()) if (act_flat.abs() <= p999).any() else 0.0
        stats["outlier_abs_error"] = p999_error
        stats["bulk_abs_error"] = bulk_error

        W = None
        for n2, m2 in model.named_modules():
            if n2 == name:
                W = m2.weight.detach().float().cpu()
                break
        if W is not None:
            x_flat = act.reshape(-1, act.shape[-1])
            x_q_flat = q.reshape(-1, act.shape[-1])
            input_err = (x_flat - x_q_flat).norm(dim=1).mean().item()
            y = F.linear(x_flat, W)
            y_q = F.linear(x_q_flat, W)
            output_err = (y - y_q).norm(dim=1).mean().item()
            stats["input_l2_error"] = input_err
            stats["output_l2_error"] = output_err
            stats["amplification_factor"] = output_err / (input_err + 1e-12)

        results[name] = stats

        ax = axes_flat[idx]
        ax.scatter(sample.numpy(), q_sample.numpy(), s=1, alpha=0.3, c=err.abs().numpy(), cmap="coolwarm", vmin=0, vmax=float(err.abs().max()))
        ax.set_xlabel("FP32 activation value")
        ax.set_ylabel("INT8 quantized value")
        ax.set_title(name, fontsize=8)
        ax.grid(True, alpha=0.2)

    for i in range(len(activations), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig_scatter.suptitle("Activation FP32→INT8 Quantization Scatter (color = abs error)", fontsize=10)
    fig_scatter.tight_layout()
    fig_scatter.savefig(output_dir / "activation_scatter.png", dpi=150)
    plt.close(fig_scatter)

    names = sorted(results.keys())
    amp_factors = [results[n].get("amplification_factor", 0) for n in names]
    fig_bar, ax_bar = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    ax_bar.barh(range(len(names)), amp_factors, color="steelblue")
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=7)
    ax_bar.set_xlabel("Amplification factor (output L2 error / input L2 error)")
    ax_bar.set_title("Quantization Error Amplification by Layer")
    ax_bar.axvline(x=1.0, color="red", linestyle="--", label="no amplification")
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.2, axis="x")
    fig_bar.tight_layout()
    fig_bar.savefig(output_dir / "amplification_factor.png", dpi=150)
    plt.close(fig_bar)

    (output_dir / "activation_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"W8A8", "W8A8_inproj16", "W16A8", "W8A16", "W4A8", "W4A16",
             "W8A8_perchan", "W8A8_pct95", "W8A8_pct99", "W8A8_pct999",
             "W8A8_pct999_lmhead16"}
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"unknown quant modes: {bad}")
    return modes


@torch.no_grad()
def analyze_weight_quantization(
    fp32_model: nn.Module,
    output_dir: Path,
    bits_list: list[int] = None,
) -> dict[str, Any]:
    if bits_list is None:
        bits_list = [8, 4]
    results: dict[str, Any] = {}
    for name, module in fp32_model.named_modules():
        if not isinstance(module, (nn.Linear, FakeQuantLinear)):
            continue
        if isinstance(module, FakeQuantLinear):
            continue
        w = module.weight.detach().float().cpu()
        layer_info: dict[str, Any] = {
            "shape": list(w.shape),
            "fp32_abs_mean": float(w.abs().mean()),
            "fp32_abs_max": float(w.abs().max()),
            "fp32_std": float(w.std()),
        }
        for bits in bits_list:
            if bits == 4:
                w_q = fake_quant_weight_groupwise(w, bits=bits, group_size=128)
            else:
                w_q = fake_quant_weight_per_output_channel(w, bits=bits)
            diff = w - w_q
            per_channel_max = diff.abs().amax(dim=1)
            per_channel_mean = diff.abs().mean(dim=1)
            small_vals = (w.abs() < 0.01).sum().item()
            large_vals = (w.abs() > 1.0).sum().item()
            zero_after_q = (w_q.abs() < 1e-10).sum().item()
            layer_info[f"w{bits}"] = {
                "mse": float((diff ** 2).mean()),
                "max_abs_error": float(diff.abs().max()),
                "relative_mse": float((diff ** 2).mean() / (w ** 2).mean()),
                "per_channel_max_mean": float(per_channel_max.mean()),
                "per_channel_max_std": float(per_channel_max.std()),
                "small_values_zeroed": int(zero_after_q),
                "total_small_values": int(small_vals),
                "total_large_values": int(large_vals),
                "scale_per_channel_cv": float(w.abs().amax(dim=1).std() / (w.abs().amax(dim=1).mean() + 1e-12)),
            }
        results[name] = layer_info
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weight_analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
