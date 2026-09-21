import logging

import numpy as np

log = logging.getLogger(__name__)


class FrameSink:
    # Output destination for rendered frames (FR-LK-001): every emit carries
    # the audio-inherited PTS. The live egress path is LiveKitAdapter.
    # publish_frame, which does one BGR->BGRA copy (one-copy, not zero-copy —
    # audit 0921 DC1). True IOSurface/CVPixelBuffer zero-copy needs the
    # fusion-mlx #913 MetalZeroCopyBridge, which is not wired into the Python
    # MLX production path (route divergence, §1 of audit 0921); ZeroCopySink
    # was removed as dead code.

    def emit(self, frame_bgr: np.ndarray, pts: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NumpyFrameSink(FrameSink):
    # Bounded ring: an unbounded list OOMs on long offline renders (1080p x
    # 30fps x 60s ~= 10GB). max_frames caps it, dropping the oldest (audit
    # P3-5). Default cap = 60s @ 30fps (audit 0921 P0-3 — the prior default
    # None left the sink unbounded); pass max_frames=None explicitly to keep
    # a full offline recording.

    DEFAULT_MAX_FRAMES = 30 * 60

    def __init__(self, max_frames: int | None = DEFAULT_MAX_FRAMES):
        self.frames = []  # list of (frame, pts)
        self.max_frames = max_frames
        self.last_pts = None

    def emit(self, frame_bgr, pts: float) -> None:
        self.frames.append((frame_bgr.copy(), pts))
        if self.max_frames is not None and len(self.frames) > self.max_frames:
            self.frames.pop(0)
        self.last_pts = pts


def make_sink(kind: str = "numpy", **kw) -> FrameSink:
    # Only the numpy sink is wired (live path goes through LiveKitAdapter).
    # "zerocopy" is accepted for backward compat but falls back to numpy —
    # the #913 bridge is not on the Python production path (audit 0921 DC1).
    if kind != "numpy":
        log.info("sink kind '%s' not available; numpy sink (one-copy)", kind)
    return NumpyFrameSink(kw.get("max_frames", NumpyFrameSink.DEFAULT_MAX_FRAMES))
