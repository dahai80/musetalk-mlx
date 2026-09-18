import logging

import numpy as np

log = logging.getLogger(__name__)


class _MockTrack:
    # Test stand-in for livekit rtc tracks; records emitted frames/PTS.
    def __init__(self):
        self.frames = []
        self.audio = []

    def capture_frame(self, rtc_frame):
        self.frames.append(rtc_frame)


class LiveKitAdapter:
    # PRD FR-LK-001/002: CVPixelBuffer-equivalent frame + audio-inherited PTS ->
    # LiveKit video track, zero-copy (no numpy round-trip, no CPU conversion once
    # fusion-mlx #913 lands). Video PTS = inherited PCM PTS; system clock forbidden.
    # Bidirectional: inbound LiveKit audio track -> push_audio(pcm, pts) (FR-LK-002).

    def __init__(
        self,
        room_url: str,
        token: str,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        force_mock: bool = False,
    ):
        self.room_url = room_url
        self.token = token
        self.width = width
        self.height = height
        self.fps = fps
        self._source = None
        self._track = None
        self._room = None
        self._mock = None  # set when livekit unavailable or forced (tests / dev)
        self._on_audio = None
        if force_mock:
            self._rtc = None
            self._mock = _MockTrack()
            log.info("LiveKitAdapter in forced mock mode")
            return
        try:
            from livekit import rtc

            self._rtc = rtc
            log.info("LiveKitAdapter created (livekit rtc available)")
        except ImportError:
            self._rtc = None
            self._mock = _MockTrack()
            log.warning("livekit not installed; LiveKitAdapter in mock mode (install [livekit] extra)")

    def connect(self) -> None:
        if self._rtc is None:
            log.info("mock connect to %s", self.room_url)
            return
        import asyncio

        async def _go():
            self._room = self._rtc.Room()
            await self._room.connect(self.room_url, self.token)
            self._source = self._rtc.VideoSource(self.width, self.height, self._rtc.VideoFormat.BGRA)
            self._track = self._rtc.LocalVideoTrack.create_video_track("musetalk", self._source)
            await self._room.local_participant.publish_track(self._track)
            log.info("LiveKit connected + video track published")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop:
            self._connect_task = asyncio.ensure_future(_go())
        else:
            asyncio.run(_go())

    def set_audio_callback(self, fn) -> None:
        # fn(pcm: np.ndarray float32 16k mono, pts: float) -> called on inbound audio.
        self._on_audio = fn

    def publish_frame(self, frame_bgr: np.ndarray, pts: float) -> None:
        # FR-LK-001: stamp RTCVideoFrame with the inherited audio PTS, never the
        # system clock. BGR -> BGRA uint8 for the rtc frame.
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            import cv2

            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            h, w = self.height, self.width
        bgra = np.ascontiguousarray(np.dstack([frame_bgr, np.full((h, w), 255, dtype=np.uint8)]))
        if self._rtc is not None and self._source is not None:
            arr = self._rtc.ArgbFrame.create(self._rtc.VideoFormat.BGRA, w, h)
            arr.data[:] = bgra.tobytes()
            arr.timestamp = int(pts * 1e9)  # PTS in nanoseconds (audio-inherited)
            self._source.capture_frame(arr)
        elif self._mock is not None:
            self._mock.frames.append({"pts": pts, "w": w, "h": h, "data": bgra})
        else:
            raise RuntimeError("LiveKitAdapter not connected")

    def on_inbound_audio(self, pcm: np.ndarray, pts: float) -> None:
        # FR-LK-002: inbound LiveKit audio -> session.push_audio with PTS.
        if self._on_audio is not None:
            self._on_audio(pcm, pts)

    def close(self) -> None:
        if self._room is not None:
            import asyncio

            try:
                asyncio.get_running_loop()
                self._disc_task = asyncio.ensure_future(self._room.disconnect())
            except (RuntimeError, AttributeError):
                pass
        self._room = self._source = self._track = None
        log.info("LiveKitAdapter closed")
