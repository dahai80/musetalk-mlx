import logging
from pathlib import Path

import numpy as np
import pytest

from musetalk_mlx.face.landmarks import LandmarkTracker

log = logging.getLogger(__name__)

WEIGHTS = Path("/tmp/mtlk_mlx_dist")
VIDEO = Path("/Users/dahai/migration/MuseTalk/data/video/sun.mp4")
AUDIO = Path("/Users/dahai/migration/MuseTalk/data/audio/eng.wav")

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="MLX dist at /tmp/mtlk_mlx_dist not built")


class _StubLandmarkBackend:
    # Forces the face-gen path through the real session without DWPose (#915).
    # Returns a fixed 68-pt face in the frame center so crop+VAE+UNet+blend run.

    def __init__(self):
        self.calls = 0

    def face_landmarks(self, frame_bgr):
        self.calls += 1
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        s = min(w, h) // 4
        lm = np.zeros((68, 2), dtype=np.float32)
        lm[:, 0] = np.linspace(cx - s, cx + s, 68)
        lm[:, 1] = np.linspace(cy - s, cy + s, 68)
        lm[29] = (cx, cy - s // 4)
        return lm


@pytest.fixture(scope="module")
def session():
    from musetalk_mlx import MuseTalkSession, config

    old = config.PRECOMPUTE
    config.PRECOMPUTE = False  # exercise the LIVE tracker path below
    try:
        s = MuseTalkSession(WEIGHTS, VIDEO, fps=30, mlx_dir=WEIGHTS)
    finally:
        config.PRECOMPUTE = old
    s._tracker = LandmarkTracker(pose_backend=_StubLandmarkBackend())
    return s


def test_session_cached_path_skips_tracker(session):
    # Precompute cache hit: render uses cached (landmarks, bbox, latent);
    # the live tracker backend is never called.
    import mlx.core as mx

    if not session._bg_pool:
        pytest.skip("no bg pool")
    lm = session._tracker.backend.face_landmarks(session._bg_pool[0])
    crop, bbox = session._cropper.crop(session._bg_pool[0], lm)
    lat = session.pipe.get_latents_for_unet(crop)
    mx.eval(lat)
    session._bg_cache = [(lm, bbox, lat)] * len(session._bg_pool)
    calls0 = session._tracker.backend.calls
    session.push_audio(np.zeros(16000 * 6, dtype=np.float32))
    # The batched path pastes/emits on the worker thread — the first call can
    # legitimately return None (PRD contract: None = not ready yet, callers
    # poll). Poll like a production consumer instead of asserting first-call.
    import time as _time

    out = None
    for _ in range(100):
        out = session.get_output_frame()
        if out is not None:
            break
        _time.sleep(0.01)
    assert out is not None
    assert session._tracker.backend.calls == calls0
    frame, _ = out
    assert frame.dtype == np.uint8


def test_session_renders_generated_frames(session):
    # End-to-end: push audio -> get_output_frame returns BGR frames with
    # audio-inherited PTS, the face-gen path (crop->VAE->UNet->blend) executed.
    import librosa

    wav, _ = librosa.load(str(AUDIO), sr=16000)
    wav = wav[: 16000 * 6]  # 6s -> one full 5s window
    session.push_audio(wav)
    frames = []
    while len(frames) < 10:
        out = session.get_output_frame()
        if out is None:
            break
        frames.append(out)
    assert len(frames) >= 1, "no frames rendered"
    frame, pts = frames[0]
    assert frame.ndim == 3 and frame.dtype == np.uint8
    assert pts >= 0.0
    # PTS monotonic
    pts_list = [p for _, p in frames]
    assert pts_list == sorted(pts_list)
    # face-gen actually ran (tracker backend was called)
    assert session._tracker.backend.calls > 0
    log.info("rendered %d frames, pts[0]=%.3f pts[-1]=%.3f", len(frames), pts_list[0], pts_list[-1])


def test_session_pts_is_audio_inherited(session):
    # FR-LK-001: PTS derives from consumed audio samples, never the system clock.
    import librosa

    wav, _ = librosa.load(str(AUDIO), sr=16000)
    n = min(len(wav), 16000 * 6)
    session.push_audio(wav[:n])
    out = session.get_output_frame()
    if out is None:
        pytest.skip("no window ready")
    _, pts = out
    # first step PTS ~ 0 (consumed starts at 0); definitely not wall-clock
    assert 0.0 <= pts < 1.0


def test_frame_sink_receives_pts(session):
    import librosa

    from musetalk_mlx.pipeline.frame_sink import NumpyFrameSink

    wav, _ = librosa.load(str(AUDIO), sr=16000)
    session.push_audio(wav[: 16000 * 6])
    sink = NumpyFrameSink()
    n = 0
    while n < 5:
        out = session.get_output_frame()
        if out is None:
            break
        sink.emit(out[0], out[1])
        n += 1
    assert len(sink.frames) >= 1
    assert sink.last_pts is not None
    assert all(p is not None for _, p in sink.frames)
