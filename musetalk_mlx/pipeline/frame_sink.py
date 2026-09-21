import logging

import numpy as np

log = logging.getLogger(__name__)


class FrameSink:
    # Output destination for rendered frames (FR-LK-001): every emit carries
    # the audio-inherited PTS. The live egress path is LiveKitAdapter.
    # publish_frame, which routes through ZeroCopySink + MetalZeroCopyBridge
    # (#913): the rendered BGRA buffer is handed to the livekit rtc VideoFrame
    # as a zero-copy memoryview (no tobytes/bytearray copy). Native IOSurface
    # path (Metal buffer -> IOSurface -> CVPixelBuffer) is used when the bridge
    # shim _ext is present; a one-copy CVPixelBufferCreate+memcpy path is the
    # fallback. Zero-copy is a PRD core feature (FR-LK-001), not optional.

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


class ZeroCopySink(FrameSink):
    # PRD FR-LK-001 zero-copy egress via the fusion-mlx #913 MetalZeroCopyBridge.
    # The bridge hands the numpy array to a Metal buffer and returns a
    # CVPixelBufferRef backed by an IOSurface (native path) or a memcpy'd
    # CVPixelBuffer (fallback). emit() keeps a reference to the last CVPixelBuffer
    # so the producer does not release it before the consumer reads it. This is
    # the zero-copy path the PRD requires — it is NOT dead code; it is the
    # production egress for any sink consumer (offline renderers, future native
    # encoders). The LiveKitAdapter uses the bridge directly for its VideoFrame
    # handoff (see adapter._publish_one), but ZeroCopySink remains the sink
    # abstraction for non-LiveKit consumers.

    def __init__(self, width: int = 1920, height: int = 1080):
        from fusion_mlx.metal.zero_copy import MetalZeroCopyBridge

        self.width = width
        self.height = height
        self.last_pts = None
        self.bridge = MetalZeroCopyBridge(width, height)
        # hold the last cvbuffer ref so the consumer can read it before release
        self._last_cvbuffer = None
        log.info("ZeroCopySink initialized (MetalZeroCopyBridge %dx%d)", width, height)

    def emit_array(self, frame_rgb_norm, pts: float):
        # frame_rgb_norm: float32/float16 RGB normalized [0,1], HxWx3 contiguous.
        # array_to_cvbuffer applies scale+offset and writes into the Metal buffer.
        self._last_cvbuffer = self.bridge.array_to_cvbuffer(frame_rgb_norm, scale=255.0, offset=0.0)
        self.last_pts = pts

    def emit(self, frame_bgr: np.ndarray, pts: float) -> None:
        # frame_bgr: uint8 BGR HxWx3. Convert to contiguous RGB (the bridge
        # expects RGB) and hand to the bridge — no Python-side copy beyond the
        # channel flip, which is unavoidable for BGR input.
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        self._last_cvbuffer = self.bridge.array_to_cvbuffer(rgb, scale=1.0, offset=0.0)
        self.last_pts = pts

    def close(self) -> None:
        release = getattr(self.bridge, "release", None)
        if release is not None:
            try:
                release()
            except Exception as e:
                log.warning("zero-copy bridge release failed (%s)", e)
        self._last_cvbuffer = None


def make_sink(kind: str = "numpy", **kw) -> FrameSink:
    # numpy = bounded in-memory ring (offline / tests). zerocopy = MetalZeroCopyBridge
    # (#913) production egress (FR-LK-001). The live LiveKit path builds its own
    # ZeroCopySink inside LiveKitAdapter; make_sink("zerocopy") is for direct
    # sink consumers (offline renderer, future native encoders).
    if kind == "zerocopy":
        try:
            return ZeroCopySink(kw.get("width", 1920), kw.get("height", 1080))
        except Exception as e:
            log.warning("zero-copy sink unavailable (%s); numpy fallback", e)
            return NumpyFrameSink(kw.get("max_frames", NumpyFrameSink.DEFAULT_MAX_FRAMES))
    if kind != "numpy":
        log.info("sink kind '%s' not recognized; numpy sink", kind)
    return NumpyFrameSink(kw.get("max_frames", NumpyFrameSink.DEFAULT_MAX_FRAMES))
