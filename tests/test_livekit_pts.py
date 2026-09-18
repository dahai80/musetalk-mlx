import numpy as np

from musetalk_mlx.livekit.adapter import LiveKitAdapter


def test_publish_frame_preserves_pts_mock():
    # livekit extra may be absent -> mock mode; still exercises the PTS path.
    a = LiveKitAdapter("wss://x", "tok", width=120, height=80, force_mock=True)
    a.connect()
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    a.publish_frame(frame, 3.21)
    assert a._mock is not None
    assert len(a._mock.frames) == 1
    assert a._mock.frames[0]["pts"] == 3.21


def test_publish_frame_resizes_to_sink():
    a = LiveKitAdapter("wss://x", "tok", width=64, height=48, force_mock=True)
    a.connect()
    frame = np.zeros((90, 160, 3), dtype=np.uint8)  # wrong size
    a.publish_frame(frame, 0.0)
    assert a._mock.frames[0]["w"] == 64
    assert a._mock.frames[0]["h"] == 48


def test_audio_callback_roundtrip():
    a = LiveKitAdapter("wss://x", "tok", width=64, height=48, force_mock=True)
    received = []
    a.set_audio_callback(lambda pcm, pts: received.append((pcm, pts)))
    pcm = np.ones(16000, dtype=np.float32)
    a.on_inbound_audio(pcm, 1.5)
    assert received == [(pcm, 1.5)]


def test_pts_is_audio_inherited_not_system_clock():
    # The adapter must never stamp from the system clock; pts flows straight
    # from the frame argument (which the session derives from consumed samples).
    a = LiveKitAdapter("wss://x", "tok", width=32, height=32, force_mock=True)
    a.connect()
    a.publish_frame(np.zeros((32, 32, 3), dtype=np.uint8), 7.77)
    assert a._mock.frames[0]["pts"] == 7.77
