import logging
import time

import numpy as np

log = logging.getLogger(__name__)

STAGES = ("stft", "whisper", "unet", "unet_build", "render", "render_eval", "vae", "vae_dec", "warp", "frame_out")


def phys_footprint() -> int:
    # Process physical footprint in bytes (macOS); 0 when unavailable.
    try:
        from fusion_mlx.utils.proc_memory import get_phys_footprint

        return int(get_phys_footprint())
    except Exception:
        return 0


def mlx_memory_breakdown() -> dict:
    # MLX holds allocated-but-freed memory in its own allocator cache
    # (set_cache_limit) — phys_footprint includes it, so a leak-vs-cache
    # attribution is impossible from phys_footprint alone. Return the MLX-side
    # counters so stress/leak reports can separate cache growth from real
    # leak (audit E5). All keys are bytes; missing API -> 0.
    out = {"active": 0, "cache": 0, "peak": 0}
    try:
        import mlx.core as mx

        for k, fn in (
            ("active", "get_active_memory"),
            ("cache", "get_cache_memory"),
            ("peak", "get_peak_memory"),
        ):
            f = getattr(mx, fn, None) or getattr(getattr(mx, "metal", None), fn, None)
            if f is not None:
                out[k] = int(f())
    except Exception:
        pass
    return out


def max_contiguous_alloc(size_bytes: int, tries: int = 3) -> bool:
    # Fragmentation probe (PRD 内存碎片监控): can the process still allocate a
    # block of ``size_bytes``? Test-allocates and frees ONE block per try —
    # allocating tries*size concurrently (3GB at once on a 4GB budget) OOMs the
    # process itself and produces false fragmentation negatives (audit fix).
    try:
        for _ in range(tries):
            block = np.empty(size_bytes, dtype=np.uint8)
            del block
        return True
    except (MemoryError, OSError):
        return False


class StageProfiler:
    # Per-stage timing (PRD 可观测性): STFT / Whisper / UNet / VAE / warp 融合 /
    # 帧输出. Aggregate stats are logged on summary().

    # sync_on_end: when True, end() calls mx.eval to drain the GPU stream so
    # stage timings include pending async work. Default OFF — syncing on EVERY
    # stage adds a round-trip to each measurement AND to the live frame path
    # when the profiler is attached, dragging 30fps (audit P2-8). Enable only
    # in dedicated profiling/benchmark runs.
    def __init__(self, report_every: int = 300, sync_on_end: bool = False):
        self.report_every = report_every
        self.sync_on_end = sync_on_end
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
        if self.sync_on_end:
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
        mlx = mlx_memory_breakdown()
        parts = []
        for s in STAGES:
            if s in self._t:
                c, tot, mx_ = self._t[s]
                parts.append(f"{s}: {tot / c * 1000:.1f}ms avg (max {mx_ * 1000:.1f}ms n={c})")
        log.info(
            "[profiler] frame=%d phys=%.0fMB mlx(active=%.0fMB cache=%.0fMB peak=%.0fMB) %s",
            self.frames,
            mb,
            mlx["active"] / (1024 * 1024),
            mlx["cache"] / (1024 * 1024),
            mlx["peak"] / (1024 * 1024),
            " | ".join(parts),
        )
        return {s: self._t[s] for s in self._t}


def pts_deviation(pts: float, expected_pts: float) -> float:
    # PRD 可观测性: PTS sync deviation (rendered step PTS vs consumed-audio PTS).
    return abs(pts - expected_pts)
