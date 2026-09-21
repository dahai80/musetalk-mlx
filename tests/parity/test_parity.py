import logging
from pathlib import Path

import numpy as np
import pytest

from tests.paths import DIST

log = logging.getLogger(__name__)

FIXTURES = Path(__file__).parent / "fixtures"
WEIGHTS = DIST

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "meta.json").exists() or not WEIGHTS.exists(),
    reason="parity fixtures or MLX dist missing "
    "(set MT_MLX_DIST; fixtures via tools/gen_parity_fixtures.py in a torch env)",
)


def _pipe():
    from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline

    return MuseTalkPipeline.from_pretrained_mlx(WEIGHTS)


def _cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


@pytest.fixture(scope="module")
def pipe():
    return _pipe()


def test_whisper_parity(pipe):
    # FR-MLX-002: complex-op threshold cosine >= 0.98.
    import mlx.core as mx

    d = np.load(FIXTURES / "whisper.npz")
    out = np.array(pipe.whisper_encoder(mx.array(d["mel"])))
    c = _cos(out, d["ref"])
    log.info("whisper cosine %.6f (out %s ref %s)", c, out.shape, d["ref"].shape)
    assert c >= 0.98, f"whisper parity {c:.4f} < 0.98"


def test_vae_encode_parity(pipe):
    # FR-MLX-004: complex-op threshold cosine >= 0.98 on the encoded latent.
    import mlx.core as mx

    d = np.load(FIXTURES / "vae.npz")
    enc = pipe.vae.encode(mx.array(d["img"]))
    lat = np.array(enc.mean)
    c = _cos(lat, d["lat"])
    log.info("vae encode cosine %.6f", c)
    assert c >= 0.98, f"vae encode parity {c:.4f} < 0.98"


def test_vae_decode_psnr(pipe):
    # FR-MLX-004: image output PSNR >= 38dB vs torch reference.
    import mlx.core as mx

    d = np.load(FIXTURES / "vae_decode.npz")
    img = np.array(pipe.vae.decode(mx.array(d["z"] / d["scaling"])))
    # torch ref is [-1,1] NCHW; MLX decode output NCHW [~-1..1]
    ref = d["img"]
    if img.ndim == 4 and img.shape[1] == 3:
        img_nchw = img
    else:
        img_nchw = np.transpose(img, (0, 3, 1, 2))
    psnr = _psnr(img_nchw, ref)
    log.info("vae decode PSNR %.2fdB", psnr)
    assert psnr >= 38.0, f"vae decode PSNR {psnr:.2f} < 38dB"


def test_unet_parity(pipe):
    # FR-MLX-003: complex-op threshold cosine >= 0.98 on predicted latent.
    # Fixture audio is post-PE (matches generate_faces internal ordering).
    import mlx.core as mx

    d = np.load(FIXTURES / "unet.npz")
    pred = np.array(pipe.unet(mx.array(d["lat8"]), mx.array([0]), mx.array(d["audio"])))
    c = _cos(pred, d["ref"])
    log.info("unet cosine %.6f", c)
    assert c >= 0.98, f"unet parity {c:.4f} < 0.98"
