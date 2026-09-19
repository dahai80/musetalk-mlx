import logging

import numpy as np

log = logging.getLogger(__name__)


class FrameSink:
    # Output destination for rendered frames (FR-LK-001): every emit carries the
    # audio-inherited PTS. Zero-copy comes from the fusion-mlx #913
    # MetalZeroCopyBridge; sinks copy only when the bridge is unavailable.

    def emit(self, frame_bgr: np.ndarray, pts: float) -> None:
        raise NotImplementedError


class NumpyFrameSink(FrameSink):
    def __init__(self):
        self.frames = []  # list of (frame, pts)
        self.last_pts = None

    def emit(self, frame_bgr, pts: float) -> None:
        self.frames.append((frame_bgr.copy(), pts))
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
        self.last_pts = pts
        self.bridge.array_to_cvbuffer(frame_rgb_norm, scale=255.0, offset=0.0)

    def emit(self, frame_bgr, pts: float) -> None:
        # uint8 BGR convenience path: normalize to [0,1] RGB for the bridge.
        rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
        self.emit_array(rgb, pts)


def make_sink(kind: str = "numpy", **kw) -> FrameSink:
    if kind == "zerocopy":
        return ZeroCopySink(kw.get("width", 1920), kw.get("height", 1080))
    return NumpyFrameSink()
