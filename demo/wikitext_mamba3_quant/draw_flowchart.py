import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig, ax = plt.subplots(figsize=(18, 28))
ax.set_xlim(0, 10)
ax.set_ylim(0, 32)
ax.axis("off")
ax.set_title("Mamba3 SISO Block: Linear vs Nonlinear Operations", fontsize=16, fontweight="bold", pad=20)

LINEAR_COLOR = "#4A90D9"
NONLIN_COLOR = "#E85D5D"
ACCUM_COLOR = "#F0A030"
IO_COLOR = "#6DBE6D"

def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=9):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.1", 
                                    facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, 
            color=text_color, fontweight="bold", wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

y = 31
draw_box(ax, 5, y, 4, 0.8, "Input u  [bsz, seq, d_model]", IO_COLOR, fontsize=10)

y -= 1.2
draw_box(ax, 5, y, 5, 0.8, "RMSNorm  (x * rsqrt(mean(x²) + ε) * w)", NONLIN_COLOR, fontsize=9)
draw_arrow(ax, 5, 31-0.4, 5, y+0.4)

y -= 1.2
draw_box(ax, 5, y, 5, 0.8, "in_proj  (Linear: d_model → 2*d_inner + 2*d_state + 3*nheads + angles)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, 31-2.4-0.4, 5, y+0.4)

y -= 0.8
ax.text(5, y, "split → z, x, B_raw, C_raw, dd_dt, dd_A, trap, angles", ha="center", fontsize=8, style="italic")
draw_arrow(ax, 5, 31-3.6-0.4, 5, y+0.2)

y -= 1.0
ax.text(1.5, y, "z branch", ha="center", fontsize=9, color="purple", fontweight="bold")
ax.text(5, y, "B/C branch", ha="center", fontsize=9, color="purple", fontweight="bold")
ax.text(8.5, y, "A/dt branch", ha="center", fontsize=9, color="purple", fontweight="bold")
ax.text(5, y-0.5, "(scan params)", ha="center", fontsize=8, color="purple")

y -= 1.0
draw_box(ax, 1.5, y, 2.5, 0.7, "z.view(nheads, headdim)", LINEAR_COLOR, fontsize=8)
draw_box(ax, 5, y, 2.5, 0.7, "B_norm = RMSNorm(B_raw)\nC_norm = RMSNorm(C_raw)", NONLIN_COLOR, fontsize=8)
draw_box(ax, 8.5, y, 2.8, 0.7, "A = -softplus(dd_A)\ndt = softplus(dd_dt + bias)", NONLIN_COLOR, fontsize=8)

y -= 1.0
draw_box(ax, 8.5, y, 2.8, 0.7, "adt = A * dt  (element-wise)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 8.5, y+1.0-0.35, 8.5, y+0.35)

y -= 1.0
draw_box(ax, 8.5, y, 2.8, 0.7, "cs = cumsum(adt)  [accumulation]", ACCUM_COLOR, fontsize=8)
draw_arrow(ax, 8.5, y+1.0-0.35, 8.5, y+0.35)

y -= 1.0
draw_box(ax, 8.5, y, 2.8, 0.7, "decay = exp(cs_j - cs_i)\n(clamped -20..5)", NONLIN_COLOR, fontsize=8)
draw_arrow(ax, 8.5, y+1.0-0.35, 8.5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 2.5, 0.7, "angles_t = tanh(angles) * π", NONLIN_COLOR, fontsize=8)
draw_arrow(ax, 5, y+2.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 2.5, 0.7, "cos = cos(angles_t)\nsin = sin(angles_t)", NONLIN_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 3.5, 0.7, "RoPE: q' = q*cos - k*sin, k' = q*sin + k*cos", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
ax.text(5, y, "For each head h:", ha="center", fontsize=9, color="purple", fontweight="bold")

y -= 0.8
draw_box(ax, 5, y, 4.5, 0.7, "qk = einsum(q_h, k_h) / sqrt(d_state)  [batch matmul]", LINEAR_COLOR, fontsize=8)

y -= 1.0
draw_box(ax, 5, y, 4.5, 0.7, "weights = qk * decay * scale", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 0.8
draw_box(ax, 8, y, 2, 0.6, "scale = dt * sigmoid(trap)", NONLIN_COLOR, fontsize=8)
draw_arrow(ax, 8, y+0.3, 6.5, y)

y -= 1.0
draw_box(ax, 5, y, 4.5, 0.7, "causal_mask (where i >= j)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 5, 0.7, "history = einsum(weights_masked, v_h)  [batch matmul]", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 5, 0.7, "mixed = history + diag + D * v  (element-wise)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 1.5, y, 2.5, 0.7, "gate = SiLU/ReLU(z)", NONLIN_COLOR, fontsize=8)
draw_arrow(ax, 1.5, y-6.0-0.35, 1.5, y+0.35)  

draw_box(ax, 5, y, 3, 0.7, "y = mixed * gate  (element-wise)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)
draw_arrow(ax, 2.8, y, 3.5, y)

y -= 1.0
draw_box(ax, 5, y, 4, 0.7, "cat(y_per_head)  [concat]", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 4, 0.7, "dropout(y)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 4, 0.7, "out_proj  (Linear: d_inner → d_model)", LINEAR_COLOR, fontsize=8)
draw_arrow(ax, 5, y+1.0-0.35, 5, y+0.35)

y -= 1.0
draw_box(ax, 5, y, 4, 0.8, "Output  [bsz, seq, d_model]", IO_COLOR, fontsize=10)
draw_arrow(ax, 5, y+1.0-0.4, 5, y+0.4)

legend_y = 1.5
ax.text(2.5, legend_y+0.6, "Legend:", fontsize=10, fontweight="bold")
draw_box(ax, 2.5, legend_y, 2.0, 0.5, "Linear", LINEAR_COLOR, fontsize=8)
draw_box(ax, 5.0, legend_y, 2.0, 0.5, "Nonlinear", NONLIN_COLOR, fontsize=8)
draw_box(ax, 7.5, legend_y, 2.0, 0.5, "Accumulation", ACCUM_COLOR, fontsize=8)
draw_box(ax, 9.5, legend_y, 1.5, 0.5, "I/O", IO_COLOR, fontsize=8)

nonlinear_ops = [
    "RMSNorm (rsqrt)",
    "softplus (A, dt)",
    "tanh (angles)",
    "cos/sin (RoPE)",
    "exp (decay)",
    "sigmoid (trap/scale)",
    "SiLU/ReLU (gate)",
]
ax.text(7, legend_y-1.5, "Nonlinear ops: " + ", ".join(nonlinear_ops), fontsize=8, style="italic",
        bbox=dict(facecolor="lightyellow", alpha=0.8))

out = Path("demo/wikitext_mamba3_quant/outputs/mamba3_block_flowchart.png")
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out}")
