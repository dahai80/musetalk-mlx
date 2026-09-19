import json
from pathlib import Path

import numpy as np
import pytest

from musetalk_mlx.eval.metrics import EvalReport, psnr, ssim


def test_psnr_identical():
    a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    assert psnr(a, a, data_range=255.0) == 999.0


def test_psnr_threshold():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 2, dtype=np.uint8)
    # small perturbation should exceed 38dB
    assert psnr(a, b, data_range=255.0) > 38.0


def test_ssim_identical():
    a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    assert ssim(a, a, data_range=255.0) == 1.0


def test_ssim_threshold():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-2, 3, a.shape), 0, 255).astype(np.uint8)
    assert ssim(a, b, data_range=255.0) > 0.95


def test_report_meets_direction():
    r = EvalReport(psnr=40.0, ssim=0.96, csim=0.99, lse_d=8.0, lse_c=6.0)
    assert r.meets("psnr") is True  # >= 38
    assert r.meets("ssim") is True  # >= 0.95
    assert r.meets("csim") is True  # >= 0.98
    assert r.meets("lse_d") is True  # <= 9
    assert r.meets("lse_c") is True  # >= 5.5
    r2 = EvalReport(psnr=30.0, lse_d=12.0, lse_c=4.0)
    assert r2.meets("psnr") is False
    assert r2.meets("lse_d") is False
    assert r2.meets("lse_c") is False


def test_report_missing_is_none():
    r = EvalReport()
    assert r.meets("psnr") is None


def test_report_serializes():
    r = EvalReport(psnr=40.0, gated=["csim"])
    d = r.to_dict()
    assert d["psnr"] == 40.0
    assert d["gated"] == ["csim"]
    assert "thresholds" in d
    assert json.dumps(d)  # JSON-serializable


SYNCNET_WEIGHTS = Path("weights/eval/auxiliary/syncnet_v2.model")


def _torch_available():
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not SYNCNET_WEIGHTS.exists() or not _torch_available(),
    reason="syncnet weights not downloaded or torch not installed (eval runs in openclaw env)",
)
def test_syncnet_loads():
    from musetalk_mlx.eval.syncnet import SyncNetS

    net = SyncNetS()
    loaded, _total = net.load_state(str(SYNCNET_WEIGHTS))
    assert loaded > 0
