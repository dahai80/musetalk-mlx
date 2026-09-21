import json
import logging
import time
from collections import deque
from itertools import pairwise
from pathlib import Path

log = logging.getLogger(__name__)


class PacedPublisher:
    # Clock-driven push loop (audit 0921 A-4 / PRD "生产禁 Python 主循环" interim
    # step): publish against a monotonic deadline instead of the old pull loop's
    # sleep(period*0.5) idle poll — no busy wait, bounded jitter, overrun
    # accounting. True production (no Python loop) = Phase 3 LiveKit queue
    # integration; this is the documented interim.
    #
    # When the loop falls behind by more than one period it RESYNCS (deadline :=
    # now + period) instead of burst-catching-up — burst catch-up would emit a
    # frame train at unbounded rate after any slow render.

    REPORT_EVERY_S = 10.0

    def __init__(self, fps: int, report_path: str | None = None):
        self.period = 1.0 / fps
        self.report_path = report_path
        self.published = 0
        self.overruns = 0
        self.intervals = deque(maxlen=4096)  # publish-to-publish seconds
        self.pts = deque(maxlen=4096)
        self._deadline = None
        self._last_pub_t = None
        self._last_report_t = None
        self._start_t = None

    def start(self) -> None:
        self._deadline = time.monotonic() + self.period
        self._start_t = self._last_report_t = time.monotonic()

    def tick(self, get_frame, publish) -> None:
        # One paced step: pull one frame (if ready), publish, then sleep to the
        # deadline. get_frame() -> (frame, pts) | None.
        out = get_frame()
        now = time.monotonic()
        if out is not None:
            frame, pts = out
            publish(frame, pts)
            self.published += 1
            if self._last_pub_t is not None:
                self.intervals.append(now - self._last_pub_t)
            self._last_pub_t = now
            self.pts.append(pts)
        if now > self._deadline:
            self.overruns += 1
            self._deadline = now + self.period  # resync, no burst catch-up
            return
        sleep_s = self._deadline - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._deadline += self.period
        self._maybe_report()

    @property
    def overrun_ratio(self) -> float:
        total = self.published + self.overruns
        return self.overruns / total if total else 0.0

    def summary(self) -> dict:
        iv = sorted(self.intervals)
        pct = lambda q: iv[min(len(iv) - 1, int(q * len(iv)))] * 1000 if iv else 0.0  # noqa: E731
        drift = 0.0
        for a, b in pairwise(self.pts):
            drift = max(drift, abs((b - a) - self.period))
        elapsed = (time.monotonic() - self._start_t) if self._start_t else 0.0
        return {
            "fps": self.published / elapsed if elapsed else 0.0,
            "published": self.published,
            "overruns": self.overruns,
            "overrun_ratio": round(self.overrun_ratio, 5),
            "interval_p50_ms": round(pct(0.50), 2),
            "interval_p95_ms": round(pct(0.95), 2),
            "interval_max_ms": round(pct(1.0), 2),
            "pts_max_step_dev_ms": round(drift * 1000, 2),
        }

    def write_report(self) -> None:
        if not self.report_path:
            return
        path = Path(self.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2))
        log.info("pacing report -> %s", path)

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report_t < self.REPORT_EVERY_S:
            return
        self._last_report_t = now
        s = self.summary()
        log.info(
            "pacing: fps=%.1f p50=%.1fms p95=%.1fms max=%.1fms overruns=%d (%.2f%%)",
            s["fps"],
            s["interval_p50_ms"],
            s["interval_p95_ms"],
            s["interval_max_ms"],
            s["overruns"],
            s["overrun_ratio"] * 100,
        )
