import logging

import numpy as np
import pytest

log = logging.getLogger(__name__)

from tests.paths import DIST, TEST_AUDIO, TEST_VIDEO, weights_ok  # noqa: E402

WEIGHTS = DIST
VIDEO = TEST_VIDEO
AUDIO = TEST_AUDIO

pytestmark = pytest.mark.skipif(
    not weights_ok(),
    reason="MLX weights missing (set MT_MLX_DIST or place *.safetensors in weights-mlx/)",
)


@pytest.fixture(scope="module")
def pipe():
    from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline

    return MuseTalkPipeline.from_pretrained_mlx(WEIGHTS)


def _center_crop_bgr(bgr, size=256):
    import cv2

    h, w = bgr.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    return cv2.resize(bgr[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def test_vae_encode_decode_roundtrip_shapes(pipe):
    # FR-MLX-004: VAE encode -> 8ch latent -> decode -> 256x256x3 uint8.
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    _, bgr = cap.read()
    cap.release()
    assert bgr is not None
    crop = _center_crop_bgr(bgr)
    latent = pipe.get_latents_for_unet(crop)
    assert latent.shape == (1, 8, 32, 32), latent.shape
    # decode the masked-target half (first 4 channels)

    face = pipe.decode_latents(latent[:, :4])
    # generate_faces is the real path; just confirm decode produces an image
    assert face.ndim == 4 and face.shape[2:] == (256, 3), face.shape


def test_unet_generate_faces_no_nan(pipe):
    # FR-MLX-003: UNet(t=0, audio cross-attn) -> recon, no NaN/black-screen.
    import cv2

    cap = cv2.VideoCapture(str(VIDEO))
    _, bgr = cap.read()
    cap.release()
    crop = _center_crop_bgr(bgr)
    latent = pipe.get_latents_for_unet(crop)
    chunk = np.zeros((50, 384), dtype=np.float32)
    import mlx.core as mx

    dtype = getattr(pipe, "_dtype", mx.float32)
    face = pipe.generate_faces(latent, mx.array(chunk[None]).astype(dtype))[0]
    assert face.shape == (256, 256, 3), face.shape
    assert np.isfinite(face).all(), "NaN in generated face"
    assert face.std() > 1.0, "black-screen: generated face has no variance"


def test_encode_audio_returns_chunks_and_tail(pipe):
    # FR-END-001 / fusion-mlx #914: encode_audio returns (chunks, tail).
    import librosa
    import mlx.core as mx
    from fusion_mlx.video.musetalk_mlx.whisper.log_mel import log_mel_spectrogram

    wav, _ = librosa.load(str(AUDIO), sr=16000)
    wav = wav[: 16000 * 2]  # 2s
    mel = log_mel_spectrogram(mx.array(wav))
    chunks, tail = pipe.encode_audio(mel, len(wav), fps=30)
    assert chunks.ndim == 3 and chunks.shape[1:] == (50, 384), chunks.shape
    assert chunks.shape[0] > 0, "no audio chunks emitted"
    assert tail is not None, "prefix tail not returned (#914)"
    log.info("encode_audio: %d chunks, tail shape %s", chunks.shape[0], getattr(tail, "shape", None))


def test_encode_audio_prefix_carries_across_windows(pipe):
    # #914: passing the prior tail as prefix to the next window must not error
    # and must still produce chunks (boundary smoothing).
    import librosa
    import mlx.core as mx
    from fusion_mlx.video.musetalk_mlx.whisper.log_mel import log_mel_spectrogram

    wav, _ = librosa.load(str(AUDIO), sr=16000)
    w1, w2 = wav[: 16000 * 5], wav[16000 * 5 : 16000 * 10]
    if w2.size == 0:
        pytest.skip("audio shorter than 10s")
    mel1 = log_mel_spectrogram(mx.array(w1))
    chunks1, tail = pipe.encode_audio(mel1, len(w1), fps=30)
    mel2 = log_mel_spectrogram(mx.array(w2))
    chunks2, _ = pipe.encode_audio(mel2, len(w2), fps=30, prefix=tail)
    assert chunks2.shape[1:] == (50, 384)
    log.info("prefix carry: w1=%d w2=%d chunks", chunks1.shape[0], chunks2.shape[0])
