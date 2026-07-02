import importlib.util
import sys
from pathlib import Path

import torch


SCRIPT = Path("demo/tiny_causal_lm/tiny_mamba3_lm.py")


def load_module():
    spec = importlib.util.spec_from_file_location("tiny_mamba3_lm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_char_dataset_encodes_and_decodes_text():
    module = load_module()
    dataset = module.CharDataset("abcabc", seq_len=3, batch_size=2, device=torch.device("cpu"))

    encoded = dataset.encode("cab")
    decoded = dataset.decode(encoded)

    assert decoded == "cab"
    x, y = dataset.sample_batch()
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)


def test_tiny_mamba3_lm_forward_shape():
    module = load_module()
    config = module.TinyConfig(vocab_size=8, seq_len=8, d_model=32, n_layer=1, d_state=8, headdim=16, chunk_size=8)
    model = module.TinyMamba3LM(config)
    x = torch.randint(0, config.vocab_size, (2, config.seq_len))

    logits = model(x)

    assert logits.shape == (2, config.seq_len, config.vocab_size)


def test_one_training_step_produces_finite_loss_and_gradients():
    module = load_module()
    config = module.TinyConfig(vocab_size=8, seq_len=8, d_model=32, n_layer=1, d_state=8, headdim=16, chunk_size=8)
    model = module.TinyMamba3LM(config)
    x = torch.randint(0, config.vocab_size, (2, config.seq_len))
    y = torch.randint(0, config.vocab_size, (2, config.seq_len))

    loss = module.compute_loss(model, x, y)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_generate_returns_requested_number_of_characters():
    module = load_module()
    dataset = module.CharDataset("abcabcabc", seq_len=4, batch_size=2, device=torch.device("cpu"))
    config = module.TinyConfig(vocab_size=dataset.vocab_size, seq_len=4, d_model=32, n_layer=1, d_state=8, headdim=16, chunk_size=4)
    model = module.TinyMamba3LM(config)

    text = module.generate_text(model, dataset, prompt="a", max_new_tokens=5, temperature=1.0)

    assert len(text) == 6


def test_split_text_creates_non_empty_ordered_splits():
    module = load_module()
    train, val, test = module.split_text("abcdefghijklmnopqrstuvwxyz", train_ratio=0.7, val_ratio=0.15)

    assert train == "abcdefghijklmnopqr"
    assert val == "stu"
    assert test == "vwxyz"
    assert train + val + test == "abcdefghijklmnopqrstuvwxyz"


def test_perplexity_from_loss_matches_exp():
    module = load_module()

    assert module.perplexity_from_loss(0.0) == 1.0
    assert abs(module.perplexity_from_loss(1.0) - 2.718281828) < 1e-6


def test_evaluate_loss_returns_finite_loss():
    module = load_module()
    dataset = module.CharDataset("abcabcabcabcabcabc", seq_len=4, batch_size=2, device=torch.device("cpu"))
    config = module.TinyConfig(vocab_size=dataset.vocab_size, seq_len=4, d_model=32, n_layer=1, d_state=8, headdim=16, chunk_size=4)
    model = module.TinyMamba3LM(config)

    loss = module.evaluate_loss(model, dataset, batches=2)

    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_save_loss_curve_supports_skipping_warmup_and_log_scale(tmp_path):
    module = load_module()
    path = tmp_path / "loss.png"

    module.save_loss_curve(
        [100.0, 10.0, 2.0, 1.0],
        path,
        val_steps=[2, 4],
        val_losses=[3.0, 1.5],
        skip_first=1,
        log_scale=True,
    )

    assert path.exists()
    assert path.stat().st_size > 0
