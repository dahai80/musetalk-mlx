import logging

import numpy as np

log = logging.getLogger(__name__)


class FrameSink:
    # Output destination for rendered frames (FR-LK-001): every emit carries the
    # audio-inherited PTS. Zero-copy comes from the fusion-mlx #913
    # MetalZeroCopyBridge; sinks copy only when the bridge is unavailable.

    def emit(self, frame_bgr: np.ndarray, pts: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NumpyFrameSink(FrameSink):
    # Bounded ring: an unbounded list OOMs on long offline renders (1080p x
    # 30fps x 60s ~= 10GB). max_frames caps it, dropping the oldest (audit
    # P3-5). Default cap = 60s @ 30fps (audit 0921 P0-3 — the prior default
    # None left make_sink("numpy") and the zerocopy fallback unbounded); pass
    # max_frames=None explicitly to keep a full offline recording.

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


class ZeroCopySink(FrameSink):
    # FR-LK-001: mlx array -> CVPixelBuffer via the fusion-mlx #913
    # MetalZeroCopyBridge (IOSurface-backed, native path when the C++ ext is
    # present, functional one-copy fallback otherwise). PTS stashed on
    # .last_pts for the LiveKit adapter to stamp onto RTCVideoFrame.

    def __init__(self, width: int, height: int):
        from fusion_mlx.metal.zero_copy import MetalZeroCopyBridge

        self.width = width
        self.height = height
        self.last_pts = None
        self.bridge = MetalZeroCopyBridge(width, height)

    def emit_array(self, frame_rgb_norm: np.ndarray, pts: float) -> None:
        # (H,W,3) float in [0,1] -> CVPixelBuffer; scale/offset per #913 API.
        # Stamp PTS ONLY after the bridge succeeds — a failed emit with an
        # advanced last_pts desyncs the video timeline (audit P1-20).
        self.bridge.array_to_cvbuffer(frame_rgb_norm, scale=255.0, offset=0.0)
        self.last_pts = pts

    def emit(self, frame_bgr, pts: float) -> None:
        # uint8 BGR convenience path. Keep uint8 contiguous (BGR->RGB view) and
        # let the bridge normalize — a float32 cast here allocates ~25MB/frame
        # and the "ZeroCopy" name lied (audit P1-18). If the bridge needs
        # normalized float, the caller should use emit_array directly.
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        self.bridge.array_to_cvbuffer(rgb, scale=255.0, offset=0.0)
        self.last_pts = pts

    def close(self) -> None:
        # Release the IOSurface/CVPixelBuffer pool — without this a long
        # session leaks Metal memory and LEAK_BUDGET_MB is unenforced (audit P1-19).
        release = getattr(self.bridge, "release", None)
        if release is not None:
            try:
                release()
            except Exception as e:
                log.warning("zero-copy bridge release failed (%s)", e)


def make_sink(kind: str = "numpy", **kw) -> FrameSink:
    if kind == "zerocopy":
        # Fall back to NumpyFrameSink if the C++ ext / Metal init is missing —
        # a missing bridge must not crash pipeline startup (audit P1-21).
        try:
            return ZeroCopySink(kw.get("width", 1920), kw.get("height", 1080))
        except Exception as e:
            log.warning("zero-copy sink unavailable (%s); numpy fallback", e)
            return NumpyFrameSink(kw.get("max_frames", NumpyFrameSink.DEFAULT_MAX_FRAMES))
    return NumpyFrameSink(kw.get("max_frames", NumpyFrameSink.DEFAULT_MAX_FRAMES))
