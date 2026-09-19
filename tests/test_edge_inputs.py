import numpy as np
import pytest

from musetalk_mlx.utils.audio import AudioWindower

VIDEO = "/Users/dahai/migration/MuseTalk/data/video/sun.mp4"


class _StubPipe:
    # Model-free pipe stub: zero chunks + identity latents, so edge inputs
    # exercise windower/session control flow without loading any model.

    def encode_audio(self, mel, n, fps=30, prefix=None):
        import mlx.core as mx

        n_chunks = max(1, n // 533)
        chunks = mx.zeros((n_chunks, 50, 384))
        tail = mx.zeros((50, 384))
        return chunks, tail

    def get_latents_for_unet(self, crop):
        import mlx.core as _m

        return _m.zeros((1, 8, 32, 32))

    def unet(self, latent, tstep, audio):
        import mlx.core as _m

        return _m.zeros((latent.shape[0], 4, 32, 32))

    def decode_latents(self, pred):
        import mlx.core as _m

        n = pred.shape[0]
        return _m.zeros((n, 256, 256, 3), dtype=_m.uint8)

    def generate_faces(self, latent, audio):
        import mlx.core as _m

        b = latent.shape[0]  # noqa: F841
        n = audio.shape[0]
        return _m.zeros((n, 256, 256, 3), dtype=_m.uint8)

    _dtype = None


@pytest.fixture
def session(monkeypatch):
    from musetalk_mlx import MuseTalkSession

    monkeypatch.setattr(MuseTalkSession, "_build_pipe", staticmethod(lambda w, m: _StubPipe()))
    s = MuseTalkSession(None, VIDEO, mlx_dir=None)
    # no landmark backend -> tracker idle path (standby frames)
    s._tracker.backend = None
    return s


def test_windower_short_audio_never_emits():
    # <10ms audio: window never fills -> session stays in standby (None).
    w = AudioWindower()
    w.push(np.zeros(100, dtype=np.float32))  # ~6ms
    win, pts = w.pop_window()
    assert win is None and pts is None


def test_session_silence_no_crash_standby(session):
    # All-zero audio: window encodes, tracker idle -> standby base frames.
    session.push_audio(np.zeros(16000 * 6, dtype=np.float32))
    out = session.get_output_frame()
    assert out is not None
    frame, pts = out
    assert frame.dtype == np.uint8
    assert pts >= 0.0


def test_session_clipping_no_crash(session):
    # Full-scale square wave (爆音): must not crash, output stays valid.
    pcm = np.sign(np.sin(np.linspace(0, 1000 * 2 * np.pi, 16000 * 6))) * 1.0
    session.push_audio(pcm.astype(np.float32))
    out = session.get_output_frame()
    assert out is not None
    frame, pts = out
    assert frame.dtype == np.uint8
    assert pts >= 0.0
