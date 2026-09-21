import asyncio
import logging
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class _MockTrack:
    # Test stand-in for livekit rtc tracks; records emitted frame dicts.
    # Bounded: a 2h mock run appends one ~6MB BGRA per frame — an unbounded
    # list leaks ~40GB in dev/test loops (audit 0921 P1-2).
    def __init__(self, max_frames: int = 3600):
        self.frames = deque(maxlen=max_frames)
        self.audio = deque(maxlen=4096)


class LiveKitAdapter:
    # PRD FR-LK-001/002: CVPixelBuffer-equivalent frame + audio-inherited PTS ->
    # LiveKit video track. Video PTS = inherited PCM PTS (capture_frame
    # timestamp_us, microseconds); system clock forbidden. Bidirectional: inbound
    # LiveKit audio track -> push_audio(pcm, pts) (FR-LK-002).
    #
    # NOTE on zero-copy: the live path still does one BGR->BGRA uint8 copy +
    # bytearray() per frame. True IOSurface/CVPixelBuffer zero-copy is the
    # fusion-mlx #913 bridge; until it lands this is "one-copy", not zero-copy
    # (audit P2-10 — the prior code/docstring claimed zero-copy falsely).

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
        # Token is held for connect() only; never logged (audit P2-10).
        self.token = token
        self.width = width
        self.height = height
        self.fps = fps
        self._source = None
        self._track = None
        self._room = None
        self._mock = None  # set when livekit unavailable or forced (tests / dev)
        self._on_audio = None
        self._on_interrupt = None  # barge-in hook (audit 0921 P3)
        self._connect_task = None
        self._audio_tasks = []  # inbound AudioStream drains — keep refs to avoid GC
        self._ready = asyncio.Event() if force_mock else None
        self._closed = False
        self._reconnect_task = None  # background watchdog (audit H1)
        self._room_state = "disconnected"  # disconnected|connecting|connected
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
        self._ready = asyncio.Event()

        async def _go():
            try:
                self._room_state = "connecting"
                self._room = self._rtc.Room()
                await self._room.connect(self.room_url, self.token)
                await self._publish_media()
                self._room_state = "connected"
                self._start_reconnect_watchdog()
                log.info("LiveKit connected + video track published")
                self._ready.set()
            except Exception as e:
                log.error("LiveKit connect failed (%s)", e)
                self._room_state = "disconnected"
                self._ready.set()
                raise

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop:
            self._connect_task = asyncio.ensure_future(_go())
        else:
            asyncio.run(_go())

    async def _publish_media(self) -> None:
        # Create the video source/track and (re)publish after every successful
        # room connect. LiveKit track publications do NOT survive a reconnect —
        # reusing the old VideoSource after reconnecting leaves capture_frame
        # feeding a detached source and the video stream silently dead (audit
        # 0921 P0-1). Also re-wires inbound audio (P1-4: stale drain tasks are
        # cancelled first so a resubscribe cannot double-consume a track).
        if self._room is None or self._rtc is None:
            return
        for t in self._audio_tasks:
            t.cancel()
        self._audio_tasks.clear()
        # VideoSource takes ONLY (width, height) — no format arg (livekit
        # 1.1.x). Format is specified per-frame on VideoFrame (audit fix).
        self._source = self._rtc.VideoSource(self.width, self.height)
        self._track = self._rtc.LocalVideoTrack.create_video_track("musetalk", self._source)
        await self._room.local_participant.publish_track(self._track)
        self._subscribe_inbound_audio()

    def _start_reconnect_watchdog(self) -> None:
        # H1: a dropped room (network jitter, server restart) left publish_frame
        # silently dropping frames forever. Watch room state; on disconnect,
        # attempt bounded reconnection so a long session self-heals instead of
        # going dark. Best-effort — if reconnect fails it logs and retries.
        if self._rtc is None or self._room is None:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return

        async def _watch():
            backoff = 1.0
            while not self._closed and self._room is not None:
                try:
                    await asyncio.sleep(1.0)
                except asyncio.CancelledError:
                    return
                if self._closed or self._room is None:
                    return
                state = getattr(self._room, "connection_state", None)
                # livekit 1.1.x Room exposes connection_state (enum) or
                # is_active(); treat any non-connected as a drop.
                connected = self._room_state == "connected"
                if state is not None:
                    try:
                        connected = state.name == "CONN_CONNECTED" or str(state).endswith("CONNECTED")
                    except Exception:
                        pass
                if connected:
                    backoff = 1.0
                    continue
                log.warning("LiveKit room disconnected; attempting reconnect (audit H1)")
                self._room_state = "connecting"
                try:
                    await self._room.connect(self.room_url, self.token)
                    # Media must be republished too — the room coming back is
                    # not enough (audit 0921 P0-1).
                    await self._publish_media()
                    self._room_state = "connected"
                    backoff = 1.0
                    log.info("LiveKit reconnected + media republished")
                except Exception as e:
                    backoff = min(backoff * 2, 30.0)
                    log.warning("reconnect failed (%s); retry in %.0fs", e, backoff)
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        return

        try:
            loop = asyncio.get_running_loop()
            self._reconnect_task = loop.create_task(_watch())
        except RuntimeError:
            pass  # no loop running (sync path) — watchdog skipped

    def wait_ready(self, timeout_s: float = 5.0) -> bool:
        # Block (sync caller) or await-able check until the room is connected.
        # Without this, the first ~100-500ms of frames after connect() are
        # silently dropped (publish_frame before _source is set) — a black
        # screen on cold start with only a debug log (audit R7).
        if self._mock is not None or self._rtc is None:
            return True
        if self._ready is None:
            return self._source is not None
        deadline = time.monotonic() + timeout_s
        while not self._ready.is_set():
            if time.monotonic() >= deadline:
                log.warning("LiveKit wait_ready timed out (%.1fs)", timeout_s)
                return False
            time.sleep(0.01)
        return self._source is not None

    def _subscribe_inbound_audio(self) -> None:
        # FR-LK-002: wire inbound remote-participant audio tracks to push_audio.
        # Subscribes to existing + future remote participants' audio (audit fix
        # — the prior on_inbound_audio callback was never invoked).
        if self._room is None or self._rtc is None:
            return

        def _push_track(track):
            try:
                from livekit.rtc.audio_stream import AudioStream

                async def _drain():
                    # One malformed frame (truncated buffer, dtype mismatch) must
                    # not kill the drain task and silently mute inbound audio
                    # forever. Catch per-frame, log, continue (audit R6).
                    while True:
                        try:
                            async for frame in AudioStream(track):
                                try:
                                    if self._on_audio is not None:
                                        pcm = (
                                            np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                                            / 32768.0
                                        )
                                        pts = frame.timestamp_us / 1e6
                                        self._on_audio(pcm, pts)
                                except Exception as e:
                                    log.warning("inbound audio frame dropped (%s)", e)
                                    continue
                        except Exception as e:
                            log.warning("inbound AudioStream broke (%s); task exiting", e)
                            return

                self._audio_tasks.append(asyncio.ensure_future(_drain()))
            except Exception as e:
                log.warning("inbound audio subscribe failed for track (%s)", e)

        for p in self._room.remote_participants.values():
            for t in p.track_publications.values():
                if t.kind == self._rtc.TrackKind.KIND_AUDIO:
                    _push_track(t.track)

        @self._room.on("track_subscribed")
        def _on_sub(track, publication, participant):
            if track.kind == self._rtc.TrackKind.KIND_AUDIO:
                _push_track(track)

    def set_audio_callback(self, fn) -> None:
        # fn(pcm: np.ndarray float32 16k mono, pts: float) -> called on inbound audio.
        self._on_audio = fn

    def set_interrupt_callback(self, fn) -> None:
        # Barge-in hook (audit 0921 P3): host wires fn() -> session.interrupt().
        # Explicit trigger only — VAD-based auto-interrupt is a future item;
        # call this whenever the host decides the current utterance is stale.
        self._on_interrupt = fn

    def publish_frame(self, frame_bgr: np.ndarray, pts: float) -> None:
        # FR-LK-001: stamp RTCVideoFrame with the inherited audio PTS, never the
        # system clock. BGR -> BGRA uint8 for the rtc frame.
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"publish_frame expects BGR HxWx3, got shape {frame_bgr.shape}")
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            import cv2

            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            h, w = self.height, self.width
        # Build BGRA in ONE contiguous buffer: write the alpha channel into a
        # pre-allocated uint8 view. The prior path did dstack (copy) + tobytes
        # (copy) + bytearray (copy) = 3 copies / ~60MB per 1080p frame (audit R4).
        bgra = np.empty((h, w, 4), dtype=np.uint8)
        bgra[:, :, :3] = frame_bgr
        bgra[:, :, 3] = 255
        if self._rtc is not None and self._source is not None:
            # VideoFrame(w, h, VideoBufferType.BGRA, data); capture_frame stamps
            # timestamp_us (microseconds) — the audio-inherited PTS. bytearray
            # over the contiguous buffer is the one unavoidable copy into the
            # rtc frame (it owns its bytes).
            frame = self._rtc.VideoFrame(w, h, self._rtc.VideoBufferType.BGRA, bytearray(bgra.tobytes()))
            self._source.capture_frame(frame, timestamp_us=int(pts * 1e6))
        elif self._mock is not None:
            self._mock.frames.append(
                {"pts": pts, "w": w, "h": h, "data": bgra, "timestamp_us": int(pts * 1e6)}
            )
        else:
            # Not connected yet (async host, _go still running) — drop rather
            # than crash so a slow connect does not kill the render loop.
            log.debug("publish_frame before ready (pts=%.3fs); dropped", pts)

    def on_inbound_audio(self, pcm: np.ndarray, pts: float) -> None:
        # FR-LK-002: inbound LiveKit audio -> session.push_audio with PTS.
        if self._on_audio is not None:
            self._on_audio(pcm, pts)

    async def aclose(self) -> None:
        # Async close: properly await disconnect so the server-side participant
        # is cleaned up. Unpublish the track first (audit P2-9). Cancel the
        # reconnect watchdog so it does not race teardown (audit H1).
        self._closed = True
        if self._reconnect_task is not None and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self._track is not None and self._room is not None:
            try:
                await self._room.local_participant.unpublish_track(self._track.sid)
            except Exception as e:
                log.warning("unpublish track failed (%s)", e)
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception as e:
                log.warning("LiveKit disconnect failed (%s)", e)
        self._room = self._source = self._track = None
        self.token = None  # token held for connect only; drop on close (audit 3.1)
        log.info("LiveKitAdapter closed")

    def close(self) -> None:
        # Sync close: run the async cleanup if no loop is running; otherwise
        # schedule it. Never silently swallow — log disconnect failures so a
        # zombie participant is observable (audit P2-9).
        self._closed = True
        if self._room is None and self._mock is not None:
            self._mock = None
            self.token = None
            log.info("LiveKitAdapter closed (mock)")
            return
        try:
            asyncio.get_running_loop()
            # A loop is running — schedule cleanup; cannot block on it here.
            if self._room is not None:
                asyncio.ensure_future(self.aclose())  # noqa: RUF006 - fire-and-forget teardown
            else:
                self.token = None
                log.info("LiveKitAdapter closed")
        except RuntimeError:
            if self._room is not None:
                try:
                    asyncio.run(self.aclose())
                except Exception as e:
                    log.warning("LiveKit sync close failed (%s)", e)
            else:
                self.token = None
                log.info("LiveKitAdapter closed")
