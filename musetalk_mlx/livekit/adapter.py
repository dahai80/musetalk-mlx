import logging

log = logging.getLogger(__name__)


class LiveKitAdapter:
    # Phase 3 (PRD FR-LK-001/002): CVPixelBuffer + audio-inherited PTS ->
    # LiveKit video track, zero-copy (no numpy round-trip, no CPU conversion).
    # Skeleton — the livekit extra is not installed by default.

    def __init__(self, room_url: str, token: str):
        self.room_url = room_url
        self.token = token
        self._track = None
        log.info("LiveKitAdapter skeleton created (phase3 deliverable)")

    def publish_frame(self, frame_bgr, pts: float) -> None:
        # TODO(phase3): wrap into RTCVideoFrame carrying the inherited PTS.
        # System-clock PTS stamping is forbidden (PRD FR-LK-001).
        raise NotImplementedError("LiveKit publish is a phase3 deliverable")
