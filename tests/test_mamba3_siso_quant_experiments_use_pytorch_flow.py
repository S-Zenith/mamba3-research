from pathlib import Path


EXPERIMENTS = [
    Path("artifacts/mamba3_siso_pytorch_repro/INT8-quant-test/int8_quant_test.py"),
    Path("artifacts/mamba3_siso_pytorch_repro/AE4M3-WINT8-quant-test/ae4m3_wint8_quant_test.py"),
    Path("artifacts/mamba3_siso_pytorch_repro/E4M3-quant-test/e4m3_quant_test.py"),
    Path("artifacts/mamba3_siso_pytorch_repro/INT8-StateBF16-quant-test/int8_statebf16_quant_test.py"),
]


def test_quant_experiment_scripts_do_not_build_official_model():
    for path in EXPERIMENTS:
        source = path.read_text(encoding="utf-8")
        assert "build_official_model" not in source
        assert "Mamba3(" not in source


def test_pytorch_flow_helper_exists():
    source = Path("artifacts/mamba3_siso_pytorch_repro/pytorch_siso_flow.py").read_text(encoding="utf-8")
    assert "class SisoWeights" in source
    assert "load_siso_weights" in source
    assert "prepare_siso_inputs" in source
    assert "out_proj" in source
