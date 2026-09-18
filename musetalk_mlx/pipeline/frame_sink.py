import logging

import numpy as np

log = logging.getLogger(__name__)


class FrameSink:
    # Output destination for rendered frames (FR-LK-001): every emit carries the
    # audio-inherited PTS. CVPixelBuffer zero-copy lives in fusion-mlx (#913);
    # the default sinks copy until that lands.

    def emit(self, frame_bgr: np.ndarray, pts: float) -> None:
        raise NotImplementedError


class NumpyFrameSink(FrameSink):
    def __init__(self):
        self.frames = []  # list of (frame, pts)
        self.last_pts = None

    def emit(self, frame_bgr, pts: float) -> None:
        self.frames.append((frame_bgr.copy(), pts))
        self.last_pts = pts


class CVPixelBufferSink(FrameSink):
    # Wraps BGR uint8 -> CVPixelBuffer (BGRA) via the CoreVideo bridge. Zero-copy
    # from an MLX array requires fusion-mlx #913; until then this is a single
    # CPU copy into an IOSurface-backed CVPixelBufferPool. PTS is stashed on
    # .last_pts for the LiveKit adapter to stamp onto RTCVideoFrame.

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.last_pts = None
        self._pool = None
        try:
            from CoreFoundation import CFRelease
            from CoreVideo import CVPixelBufferPoolCreate, kCVPixelFormatType_32BGRA

            self._CFRelease = CFRelease
            self._fmt = kCVPixelFormatType_32BGRA
            pool, _ = CVPixelBufferPoolCreate(None, None, {1: self._fmt, 2: width, 3: height})
            self._pool = pool
            log.info("CVPixelBufferSink pool ready %dx%d BGRA", width, height)
        except Exception as e:
            log.info("CVPixelBuffer pool unavailable (%s); copy-fallback to ndarray", e)
            self._fmt = None

    def emit(self, frame_bgr, pts: float) -> None:
        self.last_pts = pts
        if self._pool is None:
            return
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            log.warning(
                "frame %dx%d != sink %dx%d; skipping CVPixelBuffer emit", w, h, self.width, self.height
            )
            return
        # fusion-mlx #913 will replace this body with a zero-copy array_to_cvbuffer.
        log.debug("CVPixelBuffer emit pts=%.3fs (copy path)", pts)


def make_sink(kind: str = "numpy", **kw) -> FrameSink:
    if kind == "cvpixel":
        return CVPixelBufferSink(kw.get("width", 1920), kw.get("height", 1080))
    return NumpyFrameSink()
