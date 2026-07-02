from benchmarks.benchmark_mamba_blocks import build_cases, build_blocks


def test_quick_case_is_small_enough_for_demo() -> None:
    cases = build_cases(quick=True)

    assert len(cases) == 1
    assert cases[0].batch_size == 1
    assert cases[0].seq_len <= 128
    assert cases[0].d_model <= 256


def test_build_blocks_includes_baseline_and_mamba3_block() -> None:
    import torch

    names = [name for name, _block, _note in build_blocks(d_model=128, device=torch.device("cpu"))]

    assert "torch_baseline" in names
    assert "mamba_ssm_mamba3" in names
