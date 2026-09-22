import argparse
import json
import logging
import sys
import time
from pathlib import Path

import librosa

from musetalk_mlx import config
from musetalk_mlx.utils.profiling import max_contiguous_alloc, mlx_memory_breakdown, phys_footprint

log = logging.getLogger(__name__)

MEM_BUDGET_BYTES = int(4.0 * 1024**3)
LEAK_BUDGET_BYTES = 50 * 1024**2


def main() -> int:
    p = argparse.ArgumentParser(
        description="Long-session stability stress (PRD 2h leak <=50MB + fragmentation)"
    )
    p.add_argument("--weights", default=None)
    p.add_argument("--mlx-dir", default=None)
    p.add_argument("--audio", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--minutes", type=float, default=120.0, help="duration in minutes (default 2h)")
    p.add_argument("--report", default="results/stress_report.json")
    p.add_argument("--probe-mb", type=int, default=1024, help="max-contiguous probe block size (MB)")
    p.add_argument(
        "--with-reload",
        action="store_true",
        help="trigger session.reload() every --reload-interval minutes (reload coverage)",
    )
    p.add_argument("--reload-interval", type=float, default=30.0, help="reload cadence in minutes")
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession

    wav, _ = librosa.load(a.audio, sr=16000)
    session = MuseTalkSession(a.weights, a.video, mlx_dir=a.mlx_dir)
    session.push_audio(wav)

    t0 = time.monotonic()
    deadline = t0 + a.minutes * 60
    probe_bytes = a.probe_mb * 1024**2
    samples = []
    frames = 0
    last_push = time.monotonic()
    leak = -1
    ok = False
    reloads = []
    reload_gap_max = 0.0
    # Seed the reload schedule at t0 so the first --reload-interval boundary
    # triggers (the while-loop guard checks reloads[-1]["t"], empty list = never).
    if a.with_reload:
        reloads.append({"t": 0.0, "ok": True, "resume_gap_s": 0.0})
    # Backpressure: re-push only when the windower buffer is actually draining
    # (not on every transient window boundary), and rate-limit to wall-clock
    # cadence so consumed does not run ahead of real time and corrupt PTS
    # observations (audit fix). One chunk (sr*1s) per second keeps the stream
    # continuous without unbounded _pending growth.
    chunk = wav[: config.SR] if wav.size >= config.SR else wav
    while time.monotonic() < deadline:
        out = session.get_output_frame()
        if out is not None:
            frames += 1
        now = time.monotonic()
        if now - last_push >= 1.0:
            session.push_audio(chunk)
            last_push = now
        if a.with_reload and reloads and now - t0 >= reloads[-1]["t"] + a.reload_interval * 60:
            rt0 = time.monotonic()
            okr = session.reload(a.weights, a.mlx_dir)
            # Resume gap: time from reload start until the next frame appears.
            gap = 0.0
            while True:
                o = session.get_output_frame()
                if o is not None:
                    frames += 1
                    break
                if time.monotonic() - rt0 > 10.0:
                    break
                time.sleep(0.01)
            gap = time.monotonic() - rt0
            reload_gap_max = max(reload_gap_max, gap)
            reloads.append({"t": now - t0, "ok": okr, "resume_gap_s": round(gap, 2)})
            log.info("reload #%d ok=%s resume_gap=%.2fs", len(reloads), okr, gap)
        if now - t0 >= 60 * len(samples):
            mem = phys_footprint()
            frag_ok = max_contiguous_alloc(probe_bytes)
            mlx = mlx_memory_breakdown()
            # After a reload, mark the sample as a fresh leak baseline (reload
            # frees the old pipe + cache; comparing across reload inflates the
            # apparent leak with one-time reclaim). leak_mb is computed from
            # the post-reload baseline below.
            post_reload = bool(a.with_reload and reloads and now - t0 < reloads[-1]["t"] + 5)
            samples.append(
                {
                    "minute": len(samples) + 1,
                    "mem_mb": mem // 1024**2,
                    "frag_ok": frag_ok,
                    "mlx_active_mb": mlx["active"] // 1024**2,
                    "mlx_cache_mb": mlx["cache"] // 1024**2,
                    "post_reload": post_reload,
                }
            )
            log.info(
                "minute %d: phys=%dMB mlx(active=%dMB cache=%dMB) frag(%dMB)=%s frames=%d reloads=%d",
                len(samples),
                mem // 1024**2,
                mlx["active"] // 1024**2,
                mlx["cache"] // 1024**2,
                a.probe_mb,
                frag_ok,
                frames,
                len(reloads),
            )
    session.close()
    elapsed = time.monotonic() - t0
    leak = -1
    active_leak = -1
    ok = False
    if len(samples) >= 2:
        # Leak baseline: first sample AFTER the last reload (reload reclaims
        # the old pipe); if no reload, baseline is samples[1] (skip warmup).
        baseline_idx = 1
        if a.with_reload and reloads:
            last_r_t = reloads[-1]["t"]
            for i, s in enumerate(samples):
                if s["minute"] * 60 >= last_r_t + 60:
                    baseline_idx = i
                    break
        leak = samples[-1]["mem_mb"] - samples[baseline_idx]["mem_mb"]
        # MLX active leak: the true leak signal, immune to allocator-cache
        # bouncing and background-process contention that make phys noisy.
        # active = live tensors held by refs; stable active = no leak.
        active_leak = samples[-1]["mlx_active_mb"] - samples[baseline_idx]["mlx_active_mb"]
        # Pass on the tighter of the two signals: a phys bounce under
        # contention must not mask a real active leak, and an active leak
        # must not be hidden by phys reclaim. Both must clear the budget.
        ok = (
            leak * 1024**2 <= LEAK_BUDGET_BYTES
            and active_leak * 1024**2 <= LEAK_BUDGET_BYTES
            and all(s["frag_ok"] for s in samples)
        )
    report = {
        "duration_min": elapsed / 60,
        "frames": frames,
        "avg_fps": frames / elapsed if elapsed else 0,
        "mem_samples": samples,
        "leak_mb": leak,
        "mlx_active_leak_mb": active_leak,
        "leak_baseline_minute": samples[baseline_idx]["minute"] if len(samples) >= 2 else 0,
        "frag_probe_mb": a.probe_mb,
        "reloads": reloads,
        "reload_count": len(reloads),
        "reload_resume_gap_max_s": round(reload_gap_max, 2),
        "pass": ok,
    }
    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    log.info("stress done: %s", json.dumps(report))
    return 0 if ok else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
