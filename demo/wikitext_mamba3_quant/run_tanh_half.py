import sys, json, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from explore import (
    get_explore_config, load_wikitext2_tokens, TokenBlockDataset,
    ExploreLM, train_model, evaluate_quant_modes_standalone, parse_modes,
)

DEMO = Path("demo/wikitext_mamba3_quant")
CACHE = DEMO / "data"

configs = [
    ("tanh_half+silu", "silu", "tanh_half"),
    ("tanh_half+relu", "relu", "tanh_half"),
]

split_tokens, vocab_size, _ = load_wikitext2_tokens(CACHE)
device = torch.device("cuda")

for name, gate, angle in configs:
    print(f"\n=== Training {name} ===", flush=True)
    cfg, steps = get_explore_config("bigstate-long", vocab_size, gate_activation=gate, angle_mode=angle)
    
    train_ds = TokenBlockDataset(split_tokens["train"], 128, 8, device)
    val_ds = TokenBlockDataset(split_tokens["validation"], 128, 8, device)
    test_ds = TokenBlockDataset(split_tokens["test"], 128, 8, device)
    
    run_dir = DEMO / "outputs" / f"ablation-{name.replace('+','-')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    model, metrics = train_model(
        cfg, train_ds, val_ds, test_ds, run_dir,
        steps, 500, 10, 3e-4, None,
    )
    print(f"FP32: val_loss={metrics['final_val_loss']:.4f}, test_loss={metrics['test_loss']:.4f}, time={time.time()-t0:.0f}s", flush=True)
    
    qm = evaluate_quant_modes_standalone(
        model, cfg, val_ds, test_ds, run_dir, metrics, 10,
        ["W8A8", "W8A8_pct999"], None,
    )
    for mode, m in qm.items():
        print(f"  {mode}: val={m['val_loss']:.4f} ratio={m['val_ppl_ratio']:.2f}", flush=True)
    
    del model
    torch.cuda.empty_cache()

print("\nDone.")
