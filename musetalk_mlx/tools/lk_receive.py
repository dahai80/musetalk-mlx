import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import deque
from itertools import pairwise
from pathlib import Path

log = logging.getLogger(__name__)


async def _run(url, token, duration, arrival):
    # Subscribe to the publisher's video track and record per-frame arrival
    # wall-times (audit 0921 A-3). PTS monotonicity is enforced on the
    # publisher side (session _expected_pts + pacing report
    # pts_max_step_dev_ms); LiveKit delivers frames in capture order, so the
    # receiver verifies delivery continuity (FPS, inter-arrival) instead.
    from livekit import rtc

    room = rtc.Room()

    @room.on("track_subscribed")
    def _on_sub(track, pub, participant):
        log.info("track subscribed: %s kind=%s", pub.name, track.kind)
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return

        async def _drain():
            stream = rtc.VideoStream(track)
            async for ev in stream:
                arrival.append(time.monotonic())

        tasks = getattr(_run, "_tasks", [])
        tasks.append(asyncio.ensure_future(_drain()))
        _run._tasks = tasks

    await room.connect(url, token)
    log.info("connected to %s; receiving %.0fs", url, duration)
    await asyncio.sleep(duration)
    await room.disconnect()
    log.info("disconnected")


def main() -> int:
    p = argparse.ArgumentParser(
        description="LiveKit subscriber probe: received FPS + inter-arrival jitter (audit 0921 A-3)"
    )
    p.add_argument("--livekit-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--duration", type=float, default=30.0, help="seconds to receive")
    p.add_argument("--report", default="results/lk_receive.json")
    a = p.parse_args()

    arrival = deque(maxlen=100000)
    try:
        asyncio.run(_run(a.livekit_url, a.token, a.duration, arrival))
    except Exception as e:
        log.error("probe failed: %s", e)
        return 2

    if len(arrival) < 2:
        log.error("received %d video frames; probe FAIL", len(arrival))
        return 1
    gaps = [(b - f) * 1000.0 for f, b in pairwise(arrival)]
    elapsed = arrival[-1] - arrival[0]
    report = {
        "frames": len(arrival),
        "elapsed_s": round(elapsed, 2),
        "fps": round((len(arrival) - 1) / elapsed, 2) if elapsed > 0 else 0.0,
        "inter_arrival_p50_ms": round(statistics.median(gaps), 2),
        "inter_arrival_p99_ms": round(sorted(gaps)[int(0.99 * len(gaps))], 2),
        "inter_arrival_max_ms": round(max(gaps), 2),
    }
    report["fps_ok"] = report["fps"] >= 28
    path = Path(a.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    log.info("lk_receive report -> %s", path)
    return 0 if report["fps_ok"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
