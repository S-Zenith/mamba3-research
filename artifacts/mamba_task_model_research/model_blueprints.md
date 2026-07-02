# Model Blueprints for PyTorch Mamba3 Training

## Blueprint 1: Tiny Causal LM

Purpose: first end-to-end training smoke test.

```text
TinyMamba3LM
  token_embedding: Embedding(vocab_size, d_model)
  blocks: ModuleList[Mamba3ResidualBlock] x n_layer
  norm_f: RMSNorm(d_model)
  lm_head: Linear(d_model, vocab_size, bias=False)
  tie lm_head.weight = token_embedding.weight
```

Recommended config:

```python
config = {
    "vocab_size": 256,
    "seq_len": 128,
    "d_model": 128,
    "n_layer": 4,
    "d_state": 32,
    "expand": 2,
    "headdim": 32,
    "chunk_size": 32,
    "is_mimo": False,
}
```

Estimated parameters: about 0.4M-0.7M, depending on exact Mamba3 projection dimensions.

Training objective:

```text
logits = model(input_ids[:, :-1])
loss = cross_entropy(logits.reshape(-1, vocab), input_ids[:, 1:].reshape(-1))
```

Why this is first: it uses 1D sequence data, the same shape as Mamba3, and avoids external task frameworks.

## Blueprint 2: Small Causal LM

Purpose: first meaningful small language-model run.

```python
config = {
    "vocab_size": 4096,
    "seq_len": 256,
    "d_model": 256,
    "n_layer": 8,
    "d_state": 64,
    "expand": 2,
    "headdim": 64,
    "chunk_size": 32,
    "is_mimo": False,
}
```

Estimated parameters: about 6M-8M.

Use this after the tiny LM can overfit a small text sample.

## Blueprint 3: Vim-like Patch Classifier

Purpose: image classification with Mamba3 as a sequence backbone.

```text
image [B,3,H,W]
  -> Conv2d(3,d_model,kernel_size=patch,stride=patch)
  -> flatten patches [B,N,d_model]
  -> add pos_embed [1,N,d_model]
  -> Mamba3ResidualBlock x n_layer
  -> mean pool
  -> Linear(d_model,num_classes)
```

Toy config:

```python
config = {
    "image_size": 32,
    "patch_size": 4,
    "num_classes": 10,
    "d_model": 128,
    "n_layer": 4,
    "d_state": 32,
    "expand": 2,
    "headdim": 32,
    "chunk_size": 16,
    "is_mimo": False,
}
```

Small config:

```python
config = {
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 1000,
    "d_model": 192,
    "n_layer": 12,
    "d_state": 64,
    "expand": 2,
    "headdim": 64,
    "chunk_size": 16,
    "is_mimo": False,
}
```

## Blueprint 4: MIMO Mamba3 LM Variant

Purpose: closer to Mamba-3 official block.

```python
config = {
    "vocab_size": 4096,
    "seq_len": 128,
    "d_model": 256,
    "n_layer": 4,
    "d_state": 64,
    "expand": 2,
    "headdim": 64,
    "ngroups": 2,
    "mimo_rank": 4,
    "chunk_size": 8,
    "is_mimo": True,
}
```

Notes:

- `ngroups=2` and `chunk_size=8` were the tested combination for the current TileLang reference comparison environment.
- For pure PyTorch training, `ngroups=1` is possible, but keeping `ngroups=2` helps compare with the official kernel path.
- Use only after SISO training works.

## Residual Block Connection Pattern

Use this pattern instead of calling official `Block`, so training remains independent of fused kernels:

```python
class Mamba3ResidualBlock(nn.Module):
    def __init__(self, d_model, mixer):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = mixer

    def forward(self, x):
        return x + self.mixer(self.norm(x))
```

For stability, keep model parameters in fp32 at first. Later, add AMP after the small model trains.

## Minimal Training Milestones

1. Forward-only smoke: random tokens -> logits shape.
2. Backward smoke: one cross-entropy step has finite gradients.
3. Overfit 1 batch: loss decreases over 100-500 steps.
4. Tiny dataset: character-level text perplexity decreases.
5. Scale d_model/n_layer only after steps 1-4 pass.
