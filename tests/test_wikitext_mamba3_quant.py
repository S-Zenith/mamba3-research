from pathlib import Path
import importlib.util
import math
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "wikitext_mamba3_quant" / "train_wikitext_mamba3_quant.py"


def load_demo_module():
    spec = importlib.util.spec_from_file_location("wikitext_mamba3_quant", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 0),
        ("seq_len", 0),
        ("d_model", 0),
        ("n_layer", 0),
        ("d_state", 0),
        ("expand", 0),
        ("headdim", 0),
        ("chunk_size", 0),
        ("vocab_size", 1.5),
    ],
)
def test_validate_config_rejects_invalid_positive_integer_fields(field, value):
    demo = load_demo_module()
    kwargs = {field: value}
    if field != "vocab_size":
        kwargs["vocab_size"] = 128
    config = demo.Mamba3LMConfig(**kwargs)
    with pytest.raises(ValueError, match=field):
        demo.validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        lambda demo: demo.Mamba3LMConfig(vocab_size=128, dropout=-0.1),
        lambda demo: demo.Mamba3LMConfig(vocab_size=128, dropout=1.0),
        lambda demo: demo.Mamba3LMConfig(vocab_size=128, d_model=65, headdim=32),
        lambda demo: demo.Mamba3LMConfig(vocab_size=128, d_model=1024, expand=1, headdim=64, d_state=2),
        lambda demo: demo.Mamba3LMConfig(vocab_size=128, d_model=768, expand=1, headdim=64, d_state=1),
    ],
)
def test_validate_config_rejects_constraint_violations(config):
    demo = load_demo_module()
    with pytest.raises(ValueError):
        demo.validate_config(config(demo))


def test_token_block_dataset_batches_are_shiftable():
    demo = load_demo_module()
    tokens = list(range(50))
    dataset = demo.TokenBlockDataset(tokens=tokens, seq_len=8, batch_size=4, device=torch.device("cpu"))
    x = dataset.get_batch(start_index=0)
    assert x.shape == (4, 8)
    assert x[0].tolist() == list(range(8))
    assert x[1].tolist() == list(range(8, 16))
    random_x, random_y = dataset.sample_batch()
    assert random_x.shape == (4, 8)
    assert random_y.shape == (4, 8)
    assert random_x.dtype == torch.long
    assert random_y.dtype == torch.long
    assert torch.equal(random_x[:, 1:], random_y[:, :-1])
    x_next = dataset.get_batch(start_index=1)
    assert torch.equal(x[:, 1:], x_next[:, :-1])


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


def test_parse_modes_accepts_valid_modes_and_strips_whitespace():
    demo = load_demo_module()
    modes = demo.parse_modes("W8A8, W4A8 ,W4A16")
    assert modes == ["W8A8", "W4A8", "W4A16"]


def test_parse_modes_rejects_unknown_mode():
    demo = load_demo_module()
    with pytest.raises(ValueError, match="unknown quant modes"):
        demo.parse_modes("W8A8,BOGUS")


def test_init_swanlab_returns_none_when_disabled():
    demo = load_demo_module()
    run = demo.init_swanlab(enabled=False, project="p", run_name="r", config={})
    assert run is None
