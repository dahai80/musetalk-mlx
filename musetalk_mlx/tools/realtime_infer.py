import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import librosa

log = logging.getLogger(__name__)


class _RttProbe:
    # File-source RTT probe (audit 0921 A-3 + release-audit P0-2): audio is
    # fed in --chunk-ms chunks paced at real time; RTT of a published frame =
    # publish wall time minus the push wall time of the first feeder chunk
    # whose sample coverage includes the frame PTS.
    #
    # Two RTT scopes reported (release-audit P0-2):
    #   - stable_rtt: steady-state incremental delay — frames whose PTS is past
    #     the first full window (window_samples). Windower already full, so the
    #     covering push was recent. This is the "audio-to-video RTT" the PRD
    #     ≤80ms target measures in streaming steady state.
    #   - first_packet_rtt: cold-start delay — frames whose PTS falls within
    #     the first window. Includes the AudioWindower 5s window fill (needs a
    #     full 5s before first emit). Real user-felt latency on the first
    #     utterance; reported honestly, not hidden.
    def __init__(self, sr: int, window_samples: int = 5 * 16000):
        self.sr = sr
        self.window_samples = window_samples
        self._pushes = deque(maxlen=8192)  # (wall, samples pushed so far)
        self._total = 0
        self._rtts = []  # (pts, rtt_seconds)

    def on_push(self, n: int) -> None:
        self._total += n
        self._pushes.append((time.monotonic(), self._total))

    def on_publish(self, pts: float) -> None:
        target = int(pts * self.sr)
        for wall, end in self._pushes:
            if end >= target:
                self._rtts.append((pts, time.monotonic() - wall))
                return
        log.debug("no covering push for pts=%.3f", pts)

    def write_report(self, path: str) -> dict:
        stable = [r for pts, r in self._rtts if pts * self.sr >= self.window_samples]
        first = [r for pts, r in self._rtts if pts * self.sr < self.window_samples]

        def pct(xs, q):
            if not xs:
                return 0.0
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(q * len(xs)))] * 1000

        report = {
            "frames_total": len(self._rtts),
            "stable_rtt": {
                "frames": len(stable),
                "p50_ms": round(pct(stable, 0.50), 2),
                "p95_ms": round(pct(stable, 0.95), 2),
                "max_ms": round(pct(stable, 1.0), 2),
                "scope": "steady-state incremental (windower full); PRD <=80ms target",
            },
            "first_packet_rtt": {
                "frames": len(first),
                "p50_ms": round(pct(first, 0.50), 2),
                "max_ms": round(pct(first, 1.0), 2),
                "scope": "cold-start (includes 5s AudioWindower fill); user-felt first-utterance latency",
            },
        }
        path_ = Path(path)
        path_.parent.mkdir(parents=True, exist_ok=True)
        path_.write_text(json.dumps(report, indent=2))
        log.info("RTT report -> %s: %s", path_, report)
        return report


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
    p.add_argument("--rtt-report", default="results/realtime_rtt.json")
    p.add_argument("--chunk-ms", type=int, default=100, help="file-source feeder chunk size")
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession, config
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

    def _term(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
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

    probe = None
    if a.audio:
        wav, _ = librosa.load(a.audio, sr=16000)
        if audio_source == "file":
            # File source: feed in real-time paced chunks so publish-side RTT
            # can attribute each frame to its covering audio chunk (audit A-3).
            probe = _RttProbe(session.sr, window_samples=int(session.sr * config.WINDOW_S))
            chunk = session.sr * a.chunk_ms // 1000

            def _feeder():
                step = chunk / session.sr
                t0 = time.monotonic()
                for i, off in enumerate(range(0, len(wav), chunk)):
                    piece = wav[off : off + chunk]
                    session.push_audio(piece)
                    probe.on_push(len(piece))
                    target = t0 + (i + 1) * step
                    now = time.monotonic()
                    if target > now:
                        time.sleep(target - now)
                log.info("feeder done: %.1fs of audio streamed", len(wav) / session.sr)

            threading.Thread(target=_feeder, name="audio-feeder", daemon=True).start()
        else:
            session.push_audio(wav)

    from musetalk_mlx.pipeline.pacing import PacedPublisher

    # Push model (audit 0921 A-4): deadline-driven pacing replaces the pull
    # loop's idle sleep — no busy wait, overrun accounting, jitter report.
    pacer = PacedPublisher(a.fps, report_path="results/realtime_pacing.json")
    pacer.start()

    def _publish(frame, pts):
        adapter.publish_frame(frame, pts)
        if probe is not None:
            probe.on_publish(pts)

    try:
        while True:
            pacer.tick(session.get_output_frame, _publish)
    except KeyboardInterrupt:
        log.info("stopping")
    except BaseException:
        log.exception("publish loop terminated")
        raise
    finally:
        pacer.write_report()
        if probe is not None:
            probe.write_report(a.rtt_report)
        adapter.close()
        session.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
