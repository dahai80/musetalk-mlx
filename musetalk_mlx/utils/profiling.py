import logging
import time

import numpy as np

log = logging.getLogger(__name__)

STAGES = ("stft", "whisper", "unet", "vae", "vae_dec", "warp", "frame_out")


def phys_footprint() -> int:
    # Process physical footprint in bytes (macOS); 0 when unavailable.
    try:
        from fusion_mlx.utils.proc_memory import get_phys_footprint

        return int(get_phys_footprint())
    except Exception:
        return 0


def max_contiguous_alloc(size_bytes: int, tries: int = 3) -> bool:
    # Fragmentation probe (PRD 内存碎片监控): can the process still allocate a
    # block of ``size_bytes``? Test-allocates and frees. Only for stress runs,
    # never per-frame.
    try:
        blocks = []
        for _ in range(tries):
            blocks.append(np.empty(size_bytes, dtype=np.uint8))
        del blocks
        return True
    except (MemoryError, OSError):
        return False


class StageProfiler:
    # Per-stage timing (PRD 可观测性): STFT / Whisper / UNet / VAE / warp 融合 /
    # 帧输出. Aggregate stats are logged on summary().

    def __init__(self, report_every: int = 300):
        self.report_every = report_every
        self.frames = 0
        self._t = {}  # stage -> [count, total_s, max_s]
        self._open = None
        self._t0 = 0.0

    def begin(self, stage: str) -> None:
        self._open = stage
        self._t0 = time.perf_counter()

    def end(self) -> None:
        if self._open is None:
            return
        try:
            import mlx.core as mx

            mx.eval(mx.zeros((1,)))  # in-order stream: sync pending GPU work
        except ImportError:
            pass
        dt = time.perf_counter() - self._t0
        st = self._t.setdefault(self._open, [0, 0.0, 0.0])
        st[0] += 1
        st[1] += dt
        st[2] = max(st[2], dt)
        self._open = None

    def tick(self) -> None:
        self.frames += 1
        if self.report_every and self.frames % self.report_every == 0:
            self.summary()

    def summary(self) -> dict:
        mb = phys_footprint() / (1024 * 1024)
        parts = []
        for s in STAGES:
            if s in self._t:
                c, tot, mx_ = self._t[s]
                parts.append(f"{s}: {tot / c * 1000:.1f}ms avg (max {mx_ * 1000:.1f}ms n={c})")
        log.info("[profiler] frame=%d mem=%.0fMB %s", self.frames, mb, " | ".join(parts))
        return {s: self._t[s] for s in self._t}


def pts_deviation(pts: float, expected_pts: float) -> float:
    # PRD 可观测性: PTS sync deviation (rendered step PTS vs consumed-audio PTS).
    return abs(pts - expected_pts)
