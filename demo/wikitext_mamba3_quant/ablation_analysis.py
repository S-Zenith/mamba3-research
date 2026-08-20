from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore import (
    ExploreConfig, ExploreLM, get_explore_config, TokenBlockDataset,
    load_wikitext2_tokens, fake_quant_symmetric, fake_quant_percentile_activation,
    compute_lm_loss, evaluate_loss, perplexity_from_loss,
    replace_linear_with_fake_quant, replace_linear_with_percentile_quant,
)
from train_wikitext_mamba3_quant import RMSNorm

OUT = Path("docs/reports/assets")
OUT.mkdir(parents=True, exist_ok=True)
DEMO = Path("demo/wikitext_mamba3_quant")
SEED = 20260707

def load_model(checkpoint_path: str, gate_act: str = "relu", angle_mode: str = "tanh") -> ExploreLM:
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    cfg = ExploreConfig(**ckpt["config"])
    model = ExploreLM(cfg).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, cfg

def get_batch(cfg, seq_len=128, split="validation"):
    cache = DEMO / "data"
    split_tokens, vocab_size, _ = load_wikitext2_tokens(cache)
    ds = TokenBlockDataset(split_tokens[split], seq_len, 8, torch.device("cuda"))
    torch.manual_seed(42)
    x, y = ds.sample_batch()
    return x


# ============================================================
# Task 1 & 2: Weight and activation sorted comparison
# ============================================================
def compare_weights(model_a, model_b, name_a, name_b, label, out_prefix):
    """For each Linear layer, sort model_a's weights and plot against model_b's corresponding values."""
    layers = []
    for name, mod in model_a.named_modules():
        if isinstance(mod, nn.Linear) and not isinstance(mod, type(model_a.blocks[0].mixer.in_proj)):
            layers.append(name)
    
    # Also get in_proj, out_proj
    target_layers = []
    for i in range(4):
        target_layers.extend([
            (f"blocks.{i}.mixer.in_proj", f"b{i}.in_proj"),
            (f"blocks.{i}.mixer.out_proj", f"b{i}.out_proj"),
        ])
    target_layers.append(("lm_head", "lm_head"))
    
    n = len(target_layers)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for idx, (full_name, short_name) in enumerate(target_layers):
        ax = axes[idx]
        mod_a = dict(model_a.named_modules())[full_name]
        mod_b = dict(model_b.named_modules())[full_name]
        w_a = mod_a.weight.detach().float().cpu().flatten().numpy()
        w_b = mod_b.weight.detach().float().cpu().flatten().numpy()
        
        # Sort by model_a values
        sort_idx = np.argsort(w_a)
        sorted_a = w_a[sort_idx]
        sorted_b = w_b[sort_idx]
        
        # Sample for plotting
        n_sample = min(3000, len(sorted_a))
        step = max(1, len(sorted_a) // n_sample)
        s_a = sorted_a[::step]
        s_b = sorted_b[::step]
        
        ax.scatter(s_a, s_b, s=1, alpha=0.3, color="steelblue")
        lim = max(abs(s_a).max(), abs(s_b).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")
        corr = np.corrcoef(sorted_a, sorted_b)[0, 1]
        ax.set_title(f"{short_name} (corr={corr:.3f})", fontsize=8)
        ax.set_xlabel(f"{name_a} weight (sorted)")
        ax.set_ylabel(f"{name_b} weight")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    
    for i in range(len(target_layers), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f"Weight Matrix Comparison: {label}\n({name_a} sorted vs {name_b})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT / f"{out_prefix}_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def compare_activations(model_a, model_b, name_a, name_b, label, out_prefix):
    """Compare activation values by running both models on same batch, collecting intermediate activations."""
    cfg_a = model_a.config
    x = get_batch(cfg_a)
    
    acts_a = {}
    acts_b = {}
    
    def make_hook(storage, name):
        def fn(mod, inputs):
            if isinstance(inputs[0], torch.Tensor):
                storage[name] = inputs[0].detach().float().cpu().flatten().numpy()
        return fn
    
    hooks_a = []
    hooks_b = []
    for name, mod in model_a.named_modules():
        if isinstance(mod, nn.Linear):
            hooks_a.append(mod.register_forward_pre_hook(make_hook(acts_a, name)))
    for name, mod in model_b.named_modules():
        if isinstance(mod, nn.Linear):
            hooks_b.append(mod.register_forward_pre_hook(make_hook(acts_b, name)))
    
    with torch.no_grad():
        _ = model_a(x)
        _ = model_b(x)
    
    for h in hooks_a + hooks_b:
        h.remove()
    
    target_layers = []
    for i in range(4):
        target_layers.extend([
            (f"blocks.{i}.mixer.in_proj", f"b{i}.in_proj"),
            (f"blocks.{i}.mixer.out_proj", f"b{i}.out_proj"),
        ])
    target_layers.append(("lm_head", "lm_head"))
    
    n = len(target_layers)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for idx, (full_name, short_name) in enumerate(target_layers):
        ax = axes[idx]
        if full_name not in acts_a or full_name not in acts_b:
            ax.set_visible(False)
            continue
        a_vals = acts_a[full_name]
        b_vals = acts_b[full_name]
        
        sort_idx = np.argsort(a_vals)
        sorted_a = a_vals[sort_idx]
        sorted_b = b_vals[sort_idx]
        
        n_sample = min(3000, len(sorted_a))
        step = max(1, len(sorted_a) // n_sample)
        s_a = sorted_a[::step]
        s_b = sorted_b[::step]
        
        ax.scatter(s_a, s_b, s=1, alpha=0.3, color="steelblue")
        lim = max(abs(s_a).max(), abs(s_b).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")
        corr = np.corrcoef(sorted_a, sorted_b)[0, 1]
        ax.set_title(f"{short_name} (corr={corr:.3f})", fontsize=8)
        ax.set_xlabel(f"{name_a} act (sorted)")
        ax.set_ylabel(f"{name_b} act")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    
    for i in range(len(target_layers), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f"Activation Comparison: {label}\n({name_a} sorted vs {name_b})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT / f"{out_prefix}_activations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def run_task1():
    """Task 1: Fixed rsqrt, compare silu vs relu."""
    print("\n=== Task 1: rsqrt+silu vs rsqrt+relu ===")
    model_silu, cfg_silu = load_model(str(DEMO / "outputs/ablation-rsqrt-silu/checkpoint.pt"), "silu", "rsqrt")
    model_relu, cfg_relu = load_model(str(DEMO / "outputs/exp5-3_angle_rsqrt/checkpoint.pt"), "relu", "rsqrt")
    
    compare_weights(model_silu, model_relu, "silu", "relu", "Fixed rsqrt: SiLU vs ReLU", "task1_silu_vs_relu")
    compare_activations(model_silu, model_relu, "silu", "relu", "Fixed rsqrt: SiLU vs ReLU", "task1_silu_vs_relu")


def run_task2():
    """Task 2: Fixed relu, compare tanh vs rsqrt."""
    print("\n=== Task 2: tanh+relu vs rsqrt+relu ===")
    model_tanh, cfg_tanh = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    model_rsqrt, cfg_rsqrt = load_model(str(DEMO / "outputs/exp5-3_angle_rsqrt/checkpoint.pt"), "relu", "rsqrt")
    
    compare_weights(model_tanh, model_rsqrt, "tanh", "rsqrt", "Fixed ReLU: tanh vs rsqrt", "task2_tanh_vs_rsqrt")
    compare_activations(model_tanh, model_rsqrt, "tanh", "rsqrt", "Fixed ReLU: tanh vs rsqrt", "task2_tanh_vs_rsqrt")


# ============================================================
# Task 3: Long sequence evaluation
# ============================================================
@torch.no_grad()
def eval_long_sequence(model, cfg, seq_len, eval_batches=10):
    cache = DEMO / "data"
    split_tokens, _, _ = load_wikitext2_tokens(cache)
    device = next(model.parameters()).device
    
    val_ds = TokenBlockDataset(split_tokens["validation"], seq_len, 8, device)
    test_ds = TokenBlockDataset(split_tokens["test"], seq_len, 8, device)
    
    val_loss = evaluate_loss(model, val_ds, eval_batches)
    test_loss = evaluate_loss(model, test_ds, max(eval_batches, 5))
    return val_loss, test_loss, perplexity_from_loss(val_loss), perplexity_from_loss(test_loss)


def run_task3():
    """Task 3: Long sequence evaluation."""
    print("\n=== Task 3: Long sequence evaluation ===")
    
    configs = [
        ("tanh+silu (baseline)", "combined-silu", "silu", "tanh"),
        ("tanh+relu", "combined-relu", "relu", "tanh"),
        ("rsqrt+relu (optimized)", "exp5-3_angle_rsqrt", "relu", "rsqrt"),
    ]
    
    seq_lens = [128, 256, 512, 1024]
    results = {}
    
    for name, run_dir, gate, angle in configs:
        ckpt_path = str(DEMO / f"outputs/{run_dir}/checkpoint.pt")
        model, cfg = load_model(ckpt_path, gate, angle)
        
        results[name] = {"fp32": {}, "w8a8_pct999": {}}
        for sl in seq_lens:
            print(f"  {name}, seq_len={sl}...")
            val_loss, test_loss, val_ppl, test_ppl = eval_long_sequence(model, cfg, sl)
            results[name]["fp32"][sl] = {"val_loss": val_loss, "test_loss": test_loss, "val_ppl": val_ppl, "test_ppl": test_ppl}
            print(f"    FP32: val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")
            
            # Quantized
            qmodel = ExploreLM(cfg).to(next(model.parameters()).device)
            qmodel.load_state_dict(model.state_dict(), strict=False)
            replace_linear_with_percentile_quant(qmodel, weight_bits=8, activation_bits=8, percentile=0.999)
            qmodel.eval()
            val_loss_q, test_loss_q, val_ppl_q, test_ppl_q = eval_long_sequence(qmodel, cfg, sl)
            results[name]["w8a8_pct999"][sl] = {"val_loss": val_loss_q, "test_loss": test_loss_q, "val_ppl": val_ppl_q, "test_ppl": test_ppl_q}
            print(f"    W8A8+pct999: val_loss={val_loss_q:.4f}, val_ppl={val_ppl_q:.2f}, ratio={val_ppl_q/val_ppl:.2f}")
            del qmodel
            torch.cuda.empty_cache()
    
    # Save results
    (OUT / "task3_long_seq_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    for name, color, marker in [
        ("tanh+silu (baseline)", "#E85D5D", "o"),
        ("tanh+relu", "#F0A030", "s"),
        ("rsqrt+relu (optimized)", "#4A90D9", "^"),
    ]:
        vals = [results[name]["fp32"][sl]["val_loss"] for sl in seq_lens]
        ax.plot(seq_lens, vals, marker=marker, color=color, linewidth=2, label=f"{name} FP32")
        vals_q = [results[name]["w8a8_pct999"][sl]["val_loss"] for sl in seq_lens]
        ax.plot(seq_lens, vals_q, marker=marker, color=color, linewidth=1.5, linestyle="--", label=f"{name} W8A8+pct999")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Long Sequence Performance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xscale("log", base=2)
    
    ax = axes[1]
    for name, color, marker in [
        ("tanh+silu (baseline)", "#E85D5D", "o"),
        ("tanh+relu", "#F0A030", "s"),
        ("rsqrt+relu (optimized)", "#4A90D9", "^"),
    ]:
        ratios = [results[name]["w8a8_pct999"][sl]["val_ppl"] / results[name]["fp32"][sl]["val_ppl"] for sl in seq_lens]
        ax.plot(seq_lens, ratios, marker=marker, color=color, linewidth=2, label=name)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("W8A8+pct999 PPL Ratio (vs FP32)")
    ax.set_title("Quantization Degradation vs Sequence Length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xscale("log", base=2)
    ax.axhline(y=1.1, color="green", linestyle="--", linewidth=1, label="10% threshold")
    
    fig.tight_layout()
    path = OUT / "13_long_sequence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
    
    return results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    run_task1()
    run_task2()
    run_task3()
    print("\nAll tasks complete.")
