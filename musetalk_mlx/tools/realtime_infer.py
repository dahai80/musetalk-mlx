import argparse
import logging
import sys

import librosa

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Realtime MuseTalk -> LiveKit (FR-LK-001/002)")
    p.add_argument("--weights", required=True)
    p.add_argument("--mlx-dir", default=None)
    p.add_argument("--video", required=True, help="base video (looped)")
    p.add_argument("--livekit-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--audio", default=None, help="file fallback when no inbound LiveKit audio")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession
    from musetalk_mlx.livekit.adapter import LiveKitAdapter

    session = MuseTalkSession(a.weights, a.video, fps=a.fps, mlx_dir=a.mlx_dir)
    adapter = LiveKitAdapter(a.livekit_url, a.token, width=a.width, height=a.height, fps=a.fps)
    # Barge-in (audit 0921 P3): SIGUSR1 triggers interrupt() for manual E2E
    # testing. Host apps call session.interrupt()/adapter interrupt hook
    # directly; VAD-based auto-trigger is a future item.
    import signal

    def _barge_in(_sig, _frm):
        session.interrupt()

    signal.signal(signal.SIGUSR1, _barge_in)
    # PTS domain: the windower derives output PTS from consumed audio samples
    # (relative 0). Mixing a file source (--audio, relative 0) with inbound
    # LiveKit audio (real-time PTS) would make output PTS discontinuous and
    # break the LiveKit encoder. Enforce a SINGLE audio source; if both are
    # supplied, prefer LiveKit inbound and warn (audit E9).
    audio_source = "livekit" if a.audio is None else "file"
    if a.audio is not None:
        log.warning(
            "--audio file source active: output PTS is sample-relative (0-based). "
            "Do NOT also feed inbound LiveKit audio — PTS domain mismatch (audit E9)."
        )
    adapter.set_audio_callback(lambda pcm, pts: session.push_audio(pcm))
    adapter.connect()
    if not adapter.wait_ready(timeout_s=5.0):
        log.error("LiveKit adapter not ready after 5s; aborting")
        adapter.close()
        session.close()
        return 1
    log.info("realtime session up; audio source=%s -> LiveKit", audio_source)

    if a.audio:
        wav, _ = librosa.load(a.audio, sr=16000)
        session.push_audio(wav)

    from musetalk_mlx.pipeline.pacing import PacedPublisher

    # Push model (audit 0921 A-4): deadline-driven pacing replaces the pull
    # loop's idle sleep — no busy wait, overrun accounting, jitter report.
    pacer = PacedPublisher(a.fps, report_path="results/realtime_pacing.json")
    pacer.start()
    try:
        while True:
            pacer.tick(session.get_output_frame, adapter.publish_frame)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        pacer.write_report()
        adapter.close()
        session.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
