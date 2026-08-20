from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore import (
    ExploreConfig, ExploreLM, get_explore_config, TokenBlockDataset,
    load_wikitext2_tokens, fake_quant_symmetric, fake_quant_percentile_activation,
    fake_quant_per_channel_activation, compute_lm_loss,
)
from train_wikitext_mamba3_quant import FakeQuantLinear, RMSNorm

OUT = Path("demo/wikitext_mamba3_quant/outputs/report_plots")
OUT.mkdir(parents=True, exist_ok=True)
DEMO = Path("demo/wikitext_mamba3_quant")
SEED = 20260707

# ============================================================
# Helper: load model from checkpoint
# ============================================================
def load_model(checkpoint_path: str, gate_act: str = "relu", angle_mode: str = "tanh") -> ExploreLM:
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    cfg_dict = ckpt["config"]
    cfg = ExploreConfig(**cfg_dict)
    model = ExploreLM(cfg).cuda()
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, cfg

def get_val_batch(cfg, seq_len=128):
    cache = DEMO / "data"
    split_tokens, vocab_size, _ = load_wikitext2_tokens(cache)
    ds = TokenBlockDataset(split_tokens["validation"], seq_len, 8, torch.device("cuda"))
    torch.manual_seed(42)
    x, y = ds.sample_batch()
    return x

# ============================================================
# Plot 1: Ablation bar chart (4 configs × 3 metrics)
# ============================================================
def plot_ablation_bar():
    configs = ["tanh+silu", "tanh+relu", "rsqrt+silu", "rsqrt+relu"]
    fp32 = [8.37, 9.45, 8.50, 6.80]
    w8a8 = [17.03, 12.81, 21.35, 7.52]
    pct999 = [9.61, 9.97, 10.37, 7.05]

    x = np.arange(len(configs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w, fp32, w, label="FP32", color="#4A90D9", alpha=0.85)
    bars2 = ax.bar(x, w8a8, w, label="W8A8", color="#E85D5D", alpha=0.85)
    bars3 = ax.bar(x + w, pct999, w, label="W8A8+pct999", color="#F0A030", alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation: Three Optimizations", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 24)
    fig.tight_layout()
    fig.savefig(OUT / "01_ablation_bar.png", dpi=180)
    plt.close(fig)
    print("Saved 01_ablation_bar.png")

# ============================================================
# Plot 2: Outlier scale dominance (histogram + scale annotation)
# ============================================================
def plot_outlier_dominance():
    model, cfg = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x = get_val_batch(cfg)

    with torch.no_grad():
        acts = {}
        def hook(name):
            def fn(mod, inputs):
                acts[name] = inputs[0].detach().float().cpu().flatten()
            return fn

        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                hooks.append(mod.register_forward_pre_hook(hook(name)))
        _ = model(x)
        for h in hooks:
            h.remove()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    layers = ["blocks.0.mixer.out_proj", "blocks.1.mixer.out_proj",
              "blocks.2.mixer.out_proj", "blocks.3.mixer.out_proj"]

    for ax, layer_name in zip(axes.flat, layers):
        if layer_name not in acts:
            continue
        vals = acts[layer_name].numpy()
        abs_vals = np.abs(vals)
        p999 = np.percentile(abs_vals, 99.9)
        max_val = abs_vals.max()

        ax.hist(vals[np.abs(vals) < p999 * 3], bins=200, color="#4A90D9", alpha=0.7, label="bulk values")
        outliers = vals[np.abs(vals) >= p999 * 3]
        if len(outliers) > 0:
            ax.hist(outliers, bins=50, color="#E85D5D", alpha=0.7, label=f"outliers (>{p999*3:.1f})")

        ax.axvline(x=max_val, color="red", linestyle="--", linewidth=1.5, label=f"max={max_val:.1f}")
        ax.axvline(x=p999, color="orange", linestyle="--", linewidth=1.5, label=f"p99.9={p999:.1f}")
        ax.axvline(x=-max_val, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(x=-p999, color="orange", linestyle="--", linewidth=1.5)

        scale_max = max_val / 127
        scale_p999 = p999 / 127
        ax.set_title(f"{layer_name}\nmax_scale={scale_max:.2f} vs pct999_scale={scale_p999:.2f} ({scale_max/scale_p999:.1f}x larger)", fontsize=8)
        ax.set_xlabel("activation value")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Outlier Dominance in per-tensor INT8 Scale (tanh+relu)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "02_outlier_dominance.png", dpi=180)
    plt.close(fig)
    print("Saved 02_outlier_dominance.png")

# ============================================================
# Plot 3: INT8 staircase / information loss
# ============================================================
def plot_staircase():
    model, cfg = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x = get_val_batch(cfg)

    with torch.no_grad():
        for name, mod in model.named_modules():
            if name == "blocks.3.mixer.out_proj":
                inp = None
                def prehook(m, inputs):
                    nonlocal inp
                    inp = inputs[0].detach().float().cpu()
                h = mod.register_forward_pre_hook(prehook)
                _ = model(x)
                h.remove()

                if inp is None:
                    continue
                flat = inp.flatten()
                q = fake_quant_symmetric(inp, bits=8).flatten()
                q_perchan = fake_quant_per_channel_activation(inp, bits=8).flatten()
                q_pct = fake_quant_percentile_activation(inp, bits=8, percentile=0.999).flatten()

                n_sample = 5000
                perm = torch.randperm(flat.numel())[:n_sample]
                s_orig = flat[perm].numpy()
                s_w8a8 = q[perm].numpy()
                s_perchan = q_perchan[perm].numpy()
                s_pct999 = q_pct[perm].numpy()

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

                for ax, s_q, title in zip(axes,
                    [(s_w8a8, "W8A8 per-tensor"), (s_perchan, "W8A8 per-channel"), (s_pct999, "W8A8 pct999")],
                    ["W8A8 (per-tensor)", "W8A8 (per-channel)", "W8A8 (pct999)"]):
                    q_vals, title_val = s_q
                    ax.scatter(s_orig, q_vals, s=2, alpha=0.3, color="steelblue")
                    lim = max(abs(s_orig).max(), abs(q_vals).max())
                    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x (ideal)")
                    n_unique_orig = len(np.unique(s_orig))
                    n_unique_q = len(np.unique(q_vals))
                    ax.set_title(f"{title}\nunique: {n_unique_orig} → {n_unique_q} ({n_unique_q/n_unique_orig*100:.1f}% retained)", fontsize=9)
                    ax.set_xlabel("FP32 activation")
                    ax.set_ylabel("INT8 quantized")
                    ax.grid(True, alpha=0.2)
                    ax.legend(fontsize=8)

                fig.suptitle("INT8 Quantization Staircase Effect on block3.out_proj input (tanh+relu)", fontsize=11, fontweight="bold")
                fig.tight_layout()
                fig.savefig(OUT / "03_staircase.png", dpi=180)
                plt.close(fig)
                print("Saved 03_staircase.png")
                break

# ============================================================
# Plot 4: SiLU vs ReLU gate output distribution
# ============================================================
def plot_silu_vs_relu_gate():
    model_silu, cfg_silu = load_model(str(DEMO / "outputs/combined-silu/checkpoint.pt"), "silu", "tanh")
    model_relu, cfg_relu = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x = get_val_batch(cfg_relu)

    def get_gate_output(model, x, cfg):
        with torch.no_grad():
            emb = model.token_embedding(x)
            block = model.blocks[0]
            normed = block.norm(emb)
            mixer = block.mixer
            zx = mixer.in_proj(normed)
            z = zx[:, :, :mixer.d_inner].reshape(zx.shape[0], zx.shape[1], mixer.nheads, mixer.config.headdim)
            z_flat = z.float().cpu().flatten()

            gate_silu = torch.nn.functional.silu(z_flat)
            gate_relu = torch.nn.functional.relu(z_flat)
            return z_flat.numpy(), gate_silu.numpy(), gate_relu.numpy()

    z, g_silu, g_relu = get_gate_output(model_silu, x, cfg_silu)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.hist(z, bins=200, color="gray", alpha=0.5, label="z (pre-gate)")
    ax.set_title("z values (before gate)")
    ax.set_xlabel("value")
    ax.legend()

    ax = axes[1]
    ax.hist(g_silu, bins=200, color="#E85D5D", alpha=0.7, label="SiLU(z)")
    ax.set_title(f"SiLU gate output\nstd={np.std(g_silu):.3f}, max={np.max(g_silu):.3f}")
    ax.set_xlabel("value")
    ax.legend()

    ax = axes[2]
    ax.hist(g_relu, bins=200, color="#4A90D9", alpha=0.7, label="ReLU(z)")
    ax.set_title(f"ReLU gate output\nstd={np.std(g_relu):.3f}, max={np.max(g_relu):.3f}")
    ax.set_xlabel("value")
    ax.legend()

    fig.suptitle("Gate Activation: SiLU vs ReLU Distribution (block0, tanh+silu model)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "04_silu_vs_relu.png", dpi=180)
    plt.close(fig)
    print("Saved 04_silu_vs_relu.png")

# ============================================================
# Plot 5: rsqrt vs tanh angle distribution
# ============================================================
def plot_angle_distribution():
    model_tanh, cfg_tanh = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    model_rsqrt, cfg_rsqrt = load_model(str(DEMO / "outputs/exp5-3_angle_rsqrt/checkpoint.pt"), "relu", "rsqrt")
    x = get_val_batch(cfg_tanh)

    def get_angles(model, x, cfg):
        with torch.no_grad():
            emb = model.token_embedding(x)
            block = model.blocks[0]
            normed = block.norm(emb)
            mixer = block.mixer
            zx = mixer.in_proj(normed)
            num_rope = mixer.num_rope_angles
            angles_raw = zx[:, :, -num_rope:].float().cpu().flatten().numpy()
            return angles_raw

    tanh_raw = get_angles(model_tanh, x, cfg_tanh)
    rsqrt_raw = get_angles(model_rsqrt, x, cfg_rsqrt)

    tanh_angles = np.tanh(tanh_raw) * np.pi
    rsqrt_t = rsqrt_raw
    rsqrt_angles = np.arctan(rsqrt_t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(tanh_angles, bins=100, color="#E85D5D", alpha=0.7, label=f"tanh→×π (range ±{np.pi:.2f})")
    ax.set_title(f"tanh angle distribution\nmean={np.mean(tanh_angles):.3f}, std={np.std(tanh_angles):.3f}")
    ax.set_xlabel("angle (radians)")
    ax.set_xlim(-np.pi - 0.5, np.pi + 0.5)
    ax.legend()

    ax = axes[1]
    ax.hist(rsqrt_angles, bins=100, color="#4A90D9", alpha=0.7, label=f"arctan(t) (range ±{np.pi/2:.2f})")
    ax.set_title(f"rsqrt angle distribution\nmean={np.mean(rsqrt_angles):.3f}, std={np.std(rsqrt_angles):.3f}")
    ax.set_xlabel("angle (radians)")
    ax.set_xlim(-np.pi - 0.5, np.pi + 0.5)
    ax.legend()

    fig.suptitle("Rotation Angle Distribution: tanh+π vs rsqrt (arctan)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "05_angle_dist.png", dpi=180)
    plt.close(fig)
    print("Saved 05_angle_dist.png")

# ============================================================
# Plot 6: lm_head paradox
# ============================================================
def plot_lmhead_paradox():
    exp5_2 = json.loads(Path("demo/wikitext_mamba3_quant/outputs/exp5-2_pct999_lmhead/quant_metrics.json").read_text())

    modes = ["W8A8", "W8A8_pct999", "W8A8_pct999_lmhead16"]
    labels = ["W8A8\n(all INT8)", "W8A8+pct999\n(lm_head INT8)", "W8A8+pct999\n(lm_head FP16)"]
    vals = [exp5_2[m]["val_loss"] for m in modes]
    ratios = [exp5_2[m]["val_ppl_ratio"] for m in modes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#E85D5D", "#F0A030", "#4A90D9"]
    bars = ax1.bar(range(len(modes)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)
    ax1.set_xticks(range(len(modes)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("lm_head Quantization: INT8 vs FP16")
    ax1.axhline(y=9.45, color="green", linestyle="--", label="FP32 baseline=9.45")
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis="y")

    bars2 = ax2.bar(range(len(modes)), ratios, color=colors, alpha=0.85)
    for bar, r in zip(bars2, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, r + 0.3, f"{r:.2f}x", ha="center", fontsize=10)
    ax2.set_xticks(range(len(modes)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("PPL Ratio (val)")
    ax2.set_title("Ratio vs FP32")
    ax2.axhline(y=1.1, color="green", linestyle="--", label="10% threshold")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle("lm_head Paradox: Keeping FP16 Makes It Worse", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_lmhead_paradox.png", dpi=180)
    plt.close(fig)
    print("Saved 06_lmhead_paradox.png")

# ============================================================
# Plot 7: Information loss (unique values, entropy)
# ============================================================
def plot_info_loss():
    act_data = json.loads(Path("demo/wikitext_mamba3_quant/outputs/exp4-1_act_analysis/explore/activation_analysis.json").read_text())

    layers = sorted(act_data.keys())
    fp32_unique = [act_data[l]["unique_fp32"] for l in layers]
    int8_unique = [act_data[l]["unique_int8"] for l in layers]
    retention = [i / f * 100 for f, i in zip(fp32_unique, int8_unique)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = np.arange(len(layers))
    w = 0.35
    ax.barh(x - w/2, fp32_unique, w, color="#4A90D9", label="FP32 unique values")
    ax.barh(x + w/2, int8_unique, w, color="#E85D5D", label="INT8 unique values")
    ax.set_yticks(x)
    ax.set_yticklabels([l.replace("blocks.", "b").replace(".mixer.", ".") for l in layers], fontsize=7)
    ax.set_xlabel("unique value count")
    ax.set_title("Unique Values: FP32 vs INT8")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")

    ax = axes[1]
    colors = ["#E85D5D" if r < 0.5 else "#F0A030" if r < 1.0 else "#4A90D9" for r in retention]
    ax.barh(x, retention, color=colors, alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels([l.replace("blocks.", "b").replace(".mixer.", ".") for l in layers], fontsize=7)
    ax.set_xlabel("INT8 unique / FP32 unique (%)")
    ax.set_title("Information Retention Rate")
    ax.axvline(x=0.5, color="red", linestyle="--", label="0.5%")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle("INT8 Quantization Information Loss (tanh+relu model)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "07_info_loss.png", dpi=180)
    plt.close(fig)
    print("Saved 07_info_loss.png")

# ============================================================
# Plot 8: Amplification factor comparison (silu vs relu vs rsqrt+relu)
# ============================================================
def plot_amplification_compare():
    silu_data = json.loads(Path("demo/wikitext_mamba3_quant/outputs/ablation-tanh-silu-act/explore/activation_analysis.json").read_text())
    relu_data = json.loads(Path("demo/wikitext_mamba3_quant/outputs/exp4-1_act_analysis/explore/activation_analysis.json").read_text())
    rsqrt_relu_data = json.loads(Path("demo/wikitext_mamba3_quant/outputs/ablation-rsqrt-relu-act/explore/activation_analysis.json").read_text())

    layers = sorted(relu_data.keys())
    silu_amp = [silu_data.get(l, {}).get("amplification_factor", 0) for l in layers]
    relu_amp = [relu_data.get(l, {}).get("amplification_factor", 0) for l in layers]
    rsqrt_amp = [rsqrt_relu_data.get(l, {}).get("amplification_factor", 0) for l in layers]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(layers))
    w = 0.25
    ax.bar(x - w, silu_amp, w, color="#E85D5D", label="tanh+silu", alpha=0.85)
    ax.bar(x, relu_amp, w, color="#4A90D9", label="tanh+relu", alpha=0.85)
    ax.bar(x + w, rsqrt_amp, w, color="#6DBE6D", label="rsqrt+relu", alpha=0.85)

    ax.set_xticks(x)
    short_names = [l.replace("blocks.", "b").replace(".mixer.", ".") for l in layers]
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Amplification Factor")
    ax.set_title("Error Amplification by Layer and Config")
    ax.axhline(y=1.0, color="red", linestyle="--", label="no amplification")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "08_amplification_compare.png", dpi=180)
    plt.close(fig)
    print("Saved 08_amplification_compare.png")

# ============================================================
# Plot 9: Training loss curves (tanh+silu vs rsqrt+relu)
# ============================================================
def plot_training_curves():
    silu_m = json.loads(Path("demo/wikitext_mamba3_quant/outputs/combined-silu/metrics.json").read_text())
    rsqrt_m = json.loads(Path("demo/wikitext_mamba3_quant/outputs/exp5-3_angle_rsqrt/metrics.json").read_text())

    silu_losses = silu_m.get("fp32", silu_m)["train_losses"]
    rsqrt_losses = rsqrt_m.get("fp32", rsqrt_m)["train_losses"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(silu_losses)+1), silu_losses, linewidth=1, color="#E85D5D", alpha=0.7, label="tanh+silu (baseline)")
    ax.plot(range(1, len(rsqrt_losses)+1), rsqrt_losses, linewidth=1, color="#6DBE6D", alpha=0.7, label="rsqrt+relu (optimized)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: Baseline vs Optimized")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT / "09_training_curves.png", dpi=180)
    plt.close(fig)
    print("Saved 09_training_curves.png")


# ============================================================
# Plot 10: Staircase zoom (near-zero region)
# ============================================================
def plot_staircase_zoom():
    model, cfg = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x = get_val_batch(cfg)

    with torch.no_grad():
        for name, mod in model.named_modules():
            if name == "blocks.0.mixer.out_proj":
                inp = None
                def prehook(m, inputs):
                    nonlocal inp
                    inp = inputs[0].detach().float().cpu()
                h = mod.register_forward_pre_hook(prehook)
                _ = model(x)
                h.remove()
                if inp is None:
                    continue

                flat = inp.flatten()
                q_pt = fake_quant_symmetric(inp, bits=8).flatten()
                q_pct = fake_quant_percentile_activation(inp, bits=8, percentile=0.999).flatten()

                mask = (flat.abs() < 3.0)
                s_orig = flat[mask].numpy()
                s_pt = q_pt[mask].numpy()
                s_pct = q_pct[mask].numpy()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                for ax, s_q, title, color in [
                    (axes[0], s_pt, "W8A8 per-tensor", "#E85D5D"),
                    (axes[1], s_pct, "W8A8 pct999", "#4A90D9"),
                ]:
                    ax.scatter(s_orig, s_q, s=3, alpha=0.4, color=color)
                    ax.plot([-3, 3], [-3, 3], "gray", linestyle="--", linewidth=1, label="y=x (ideal)")
                    steps = np.unique(s_q)
                    for step_val in steps:
                        if abs(step_val) < 3:
                            ax.axhline(y=step_val, color="gray", alpha=0.15, linewidth=0.5)
                    n_unique = len(np.unique(s_q))
                    ax.set_title(f"{title} (zoom [-3, 3])\n{n_unique} INT8 levels in this range", fontsize=10)
                    ax.set_xlabel("FP32 activation value")
                    ax.set_ylabel("INT8 quantized value")
                    ax.set_xlim(-3, 3)
                    ax.set_ylim(-3, 3)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.2)
                    ax.set_aspect("equal")

                fig.suptitle("INT8 Staircase Effect (zoomed near zero, block0.out_proj input)", fontsize=12, fontweight="bold")
                fig.tight_layout()
                fig.savefig(OUT / "10_staircase_zoom.png", dpi=180)
                plt.close(fig)
                print("Saved 10_staircase_zoom.png")
                break


# ============================================================
# Plot 11: ReLU robustness mechanism
# ============================================================
def plot_relu_robustness_mechanism():
    z = torch.linspace(-5, 5, 1000)
    silu = F.silu(z)
    relu = F.relu(z)
    silu_grad = torch.sigmoid(z) * (1 + z * (1 - torch.sigmoid(z)))
    relu_grad = (z > 0).float()

    dz = 0.03
    z_q = torch.round(z / dz) * dz
    silu_err = (F.silu(z_q) - silu).abs()
    relu_err = (F.relu(z_q) - relu).abs()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(z.numpy(), silu.numpy(), color="#E85D5D", linewidth=2, label="SiLU(z) = z·σ(z)")
    ax.plot(z.numpy(), relu.numpy(), color="#4A90D9", linewidth=2, label="ReLU(z) = max(0,z)")
    ax.set_title("Activation Function", fontsize=11)
    ax.set_xlabel("z")
    ax.set_ylabel("f(z)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    ax = axes[0, 1]
    ax.plot(z.numpy(), silu_grad.numpy(), color="#E85D5D", linewidth=2, label="SiLU' (z)")
    ax.plot(z.numpy(), relu_grad.numpy(), color="#4A90D9", linewidth=2, label="ReLU' (z)")
    ax.set_title("Derivative (error gain)", fontsize=11)
    ax.set_xlabel("z")
    ax.set_ylabel("f'(z)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.5, 2.0)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, label="gain=1 (passthrough)")
    ax.axvline(x=0, color="gray", linewidth=0.5)

    ax = axes[1, 0]
    ax.plot(z.numpy(), silu_err.numpy(), color="#E85D5D", linewidth=1.5, label="SiLU output error")
    ax.plot(z.numpy(), relu_err.numpy(), color="#4A90D9", linewidth=1.5, label="ReLU output error")
    ax.fill_between(z.numpy(), 0, silu_err.numpy(), alpha=0.15, color="#E85D5D")
    ax.fill_between(z.numpy(), 0, relu_err.numpy(), alpha=0.15, color="#4A90D9")
    ax.set_title(f"Output Error from Input Quantization (Δz={dz})", fontsize=11)
    ax.set_xlabel("z")
    ax.set_ylabel("|f(z+Δz) - f(z)|")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    model_silu, cfg_silu = load_model(str(DEMO / "outputs/combined-silu/checkpoint.pt"), "silu", "tanh")
    model_relu, cfg_relu = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x_batch = get_val_batch(cfg_relu)

    def get_gate_z(model, x):
        with torch.no_grad():
            emb = model.token_embedding(x)
            block = model.blocks[0]
            normed = block.norm(emb)
            mixer = block.mixer
            zx = mixer.in_proj(normed)
            z = zx[:, :, :mixer.d_inner]
            return z.float().cpu().flatten().numpy()

    z_silu_model = get_gate_z(model_silu, x_batch)
    z_relu_model = get_gate_z(model_relu, x_batch)

    ax = axes[1, 1]
    ax.hist(z_silu_model, bins=100, alpha=0.5, color="#E85D5D", density=True, label=f"SiLU model z (std={np.std(z_silu_model):.3f})")
    ax.hist(z_relu_model, bins=100, alpha=0.5, color="#4A90D9", density=True, label=f"ReLU model z (std={np.std(z_relu_model):.3f})")
    ax.set_title("z Distribution (block0, pre-gate)", fontsize=11)
    ax.set_xlabel("z value")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-3, 3)

    fig.suptitle("ReLU Robustness Mechanism: Bounded Error Gain", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "11_relu_robustness.png", dpi=180)
    plt.close(fig)
    print("Saved 11_relu_robustness.png")


# ============================================================
# Plot 12: lm_head logit clipping regularization
# ============================================================
def plot_lmhead_clipping():
    model, cfg = load_model(str(DEMO / "outputs/combined-relu/checkpoint.pt"), "relu", "tanh")
    x = get_val_batch(cfg)

    with torch.no_grad():
        emb = model.token_embedding(x)
        h = emb
        for block in model.blocks:
            h = block(h)
        h = model.norm_f(h)
    logits_fp32 = model.lm_head(h)
    logits_fp32_cpu = logits_fp32.float().cpu().detach()
    logits_flat = logits_fp32_cpu.flatten()

    abs_max = logits_flat.abs().max().item()
    scale = abs_max / 127.0
    logits_int8 = (torch.clamp(torch.round(logits_fp32_cpu / scale), -128, 127) * scale).detach()
    logits_clip = fake_quant_percentile_activation(logits_fp32_cpu, bits=8, percentile=0.999).detach()

    labels = x[:, 1:].cpu()
    loss_fp32 = F.cross_entropy(logits_fp32_cpu[:, :-1, :].reshape(-1, logits_fp32_cpu.shape[-1]), labels.reshape(-1)).item()
    loss_int8 = F.cross_entropy(logits_int8[:, :-1, :].reshape(-1, logits_int8.shape[-1]), labels.reshape(-1)).item()
    loss_clip = F.cross_entropy(logits_clip[:, :-1, :].reshape(-1, logits_clip.shape[-1]), labels.reshape(-1)).item()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    sample_idx = torch.randperm(logits_flat.numel())[:5000]
    ax.scatter(logits_flat[sample_idx].detach().numpy(), logits_int8.flatten()[sample_idx].detach().numpy(),
               s=1, alpha=0.3, color="#E85D5D", label=f"per-tensor INT8 (loss={loss_int8:.4f})")
    ax.scatter(logits_flat[sample_idx].detach().numpy(), logits_clip.flatten()[sample_idx].detach().numpy(),
               s=1, alpha=0.3, color="#4A90D9", label=f"pct999 INT8 (loss={loss_clip:.4f})")
    lim = max(abs(logits_flat[sample_idx]).max(), 5)
    ax.plot([-lim, lim], [-lim, lim], "gray", linestyle="--", linewidth=0.5)
    ax.set_title(f"Logit Quantization Scatter\nFP32 loss={loss_fp32:.4f}", fontsize=10)
    ax.set_xlabel("FP32 logit")
    ax.set_ylabel("INT8 logit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 15)

    ax = axes[0, 1]
    pos_logits = logits_flat[logits_flat > 0]
    neg_logits = logits_flat[logits_flat < 0]
    ax.hist(pos_logits.numpy(), bins=200, color="#E85D5D", alpha=0.5, label=f"FP32 positive (max={pos_logits.max():.1f})")
    ax.hist(neg_logits.numpy(), bins=200, color="#4A90D9", alpha=0.5, label=f"FP32 negative (min={neg_logits.min():.1f})")
    pct_val = np.percentile(logits_flat.abs().numpy(), 99.9)
    ax.axvline(x=pct_val, color="orange", linestyle="--", linewidth=1.5, label=f"p99.9={pct_val:.2f}")
    ax.axvline(x=abs_max, color="red", linestyle="--", linewidth=1.5, label=f"max={abs_max:.2f}")
    ax.axvline(x=-pct_val, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(x=-abs_max, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Logit Distribution: max vs p99.9", fontsize=10)
    ax.set_xlabel("logit value")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-abs_max * 1.1, abs_max * 1.1)

    ax = axes[1, 0]
    seq_idx = 0
    top_k = 20
    log_probs_fp32 = F.log_softmax(logits_fp32_cpu[seq_idx, 0, :], dim=-1).numpy()
    log_probs_int8 = F.log_softmax(logits_int8[seq_idx, 0, :], dim=-1).numpy()
    log_probs_clip = F.log_softmax(logits_clip[seq_idx, 0, :], dim=-1).numpy()

    sorted_idx = np.argsort(log_probs_fp32)[::-1][:top_k]
    x_pos = np.arange(top_k)
    width = 0.25
    ax.bar(x_pos - width, log_probs_fp32[sorted_idx], width, color="#4A90D9", alpha=0.7, label="FP32")
    ax.bar(x_pos, log_probs_int8[sorted_idx], width, color="#E85D5D", alpha=0.7, label="INT8 per-tensor")
    ax.bar(x_pos + width, log_probs_clip[sorted_idx], width, color="#F0A030", alpha=0.7, label="INT8 pct999")
    ax.set_title(f"Top-{top_k} Token Log-Probabilities (seq=0, pos=0)", fontsize=10)
    ax.set_xlabel("token rank")
    ax.set_ylabel("log probability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1, 1]
    token_idx = 0
    probs_fp32 = F.softmax(logits_fp32_cpu[seq_idx, token_idx, :], dim=-1).numpy()
    probs_int8 = F.softmax(logits_int8[seq_idx, token_idx, :], dim=-1).numpy()
    probs_clip = F.softmax(logits_clip[seq_idx, token_idx, :], dim=-1).numpy()

    sorted_idx = np.argsort(probs_fp32)[::-1][:top_k]
    ax.bar(x_pos - width, probs_fp32[sorted_idx], width, color="#4A90D9", alpha=0.7, label="FP32")
    ax.bar(x_pos, probs_int8[sorted_idx], width, color="#E85D5D", alpha=0.7, label="INT8 per-tensor")
    ax.bar(x_pos + width, probs_clip[sorted_idx], width, color="#F0A030", alpha=0.7, label="INT8 pct999")
    ax.set_title(f"Top-{top_k} Token Probabilities (seq=0, pos=0)", fontsize=10)
    ax.set_xlabel("token rank")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("lm_head Quantization: Logit Clipping as Regularization", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "12_lmhead_clipping.png", dpi=180)
    plt.close(fig)
    print("Saved 12_lmhead_clipping.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    plot_ablation_bar()
    plot_outlier_dominance()
    plot_staircase()
    plot_silu_vs_relu_gate()
    plot_angle_distribution()
    plot_lmhead_paradox()
    plot_info_loss()
    plot_amplification_compare()
    plot_training_curves()
    plot_staircase_zoom()
    plot_relu_robustness_mechanism()
    plot_lmhead_clipping()
    print(f"\nAll plots saved to {OUT}")
