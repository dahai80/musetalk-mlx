import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import librosa

from musetalk_mlx import config
from musetalk_mlx.utils.profiling import mlx_memory_breakdown, phys_footprint

log = logging.getLogger(__name__)


def _gpu_contamination() -> dict:
    # Detect background GPU consumers that invalidate benchmark numbers
    # (memory note: fusion-mlx-server + linguakids watchdog + other Claude
    # sessions' image_worker inflate render_eval several-fold). The benchmark
    # records what it found so a reader can judge whether the FPS is a code
    # limit or contention — a bare FPS number without this is misleading
    # (audit v3 cited contaminated artifacts as code regressions).
    procs = []
    try:
        out = subprocess.run(["ps", "-Ao", "pid,comm"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            comm = parts[1]
            if any(k in comm for k in ("fusion-mlx-server", "fusion-plugin-server", "image_worker")):
                procs.append({"pid": parts[0], "comm": comm.strip()})
    except Exception:
        pass
    util = None
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-k", "Device Utilization", "-d", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        for line in out.splitlines():
            if "Device Utilization" in line:
                util = line.split("=", 1)[-1].strip().strip("%")
                break
    except Exception:
        pass
    return {
        "gpu_procs": procs,
        "gpu_utilization_pct": util,
        "clean": not procs and (util is None or float(util) < 5),
    }


def _mlx_version() -> str:
    try:
        import mlx

        return getattr(mlx, "__version__", "unknown")
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Per-stage benchmark: STFT/Whisper/UNet/VAE/warp/frame-out, FPS, memory (PRD 9.5)"
    )
    p.add_argument("--weights", default=None)
    p.add_argument("--mlx-dir", default=None)
    p.add_argument("--audio", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--report", default="results/benchmark_report.json")
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession

    contamination = _gpu_contamination()
    mlx_ver = _mlx_version()
    if not contamination["clean"]:
        log.warning(
            "GPU contamination detected (%d procs, util=%s%%): FPS/render_eval "
            "numbers are NOT a code limit. Stop fusion-mlx-server + watchdog and "
            "rerun on a clean GPU for valid numbers.",
            len(contamination["gpu_procs"]),
            contamination["gpu_utilization_pct"],
        )
    wav, _ = librosa.load(a.audio, sr=16000)
    # Audio must cover at least one window or pop_window never returns and the
    # loop exits with frames=0 — fail loudly instead of reporting a false 0 FPS
    # (audit fix).
    if wav.size < config.SR * config.WINDOW_S:
        log.error("audio too short (%.1fs < one %.0fs window)", wav.size / config.SR, config.WINDOW_S)
        return 1
    session = MuseTalkSession(a.weights, a.video, mlx_dir=a.mlx_dir)
    try:
        session.push_audio(wav)

        import time

        t0 = time.monotonic()
        frames = 0
        while True:
            out = session.get_output_frame()
            if out is not None:
                frames += 1
                continue
            # The render thread produces asynchronously; a None with the
            # pipeline still busy is a producer hiccup, not end-of-stream.
            if session.idle:
                break
            time.sleep(0.002)
        elapsed = time.monotonic() - t0
        stats = session.profiler.summary()
        fps = frames / elapsed if elapsed else 0
        mlx_mem = mlx_memory_breakdown()
        report = {
            "frames": frames,
            "elapsed_s": elapsed,
            "fps": fps,
            "meets_30fps": fps >= 30,
            "mem_mb": phys_footprint() // 1024**2,
            "mlx_active_mb": mlx_mem["active"] // 1024**2,
            "mlx_cache_mb": mlx_mem["cache"] // 1024**2,
            "mlx_peak_mb": mlx_mem["peak"] // 1024**2,
            "stages": {
                k: {"n": v[0], "avg_ms": v[1] / v[0] * 1000, "max_ms": v[2] * 1000} for k, v in stats.items()
            },
            "budgets": {"fps": config.FPS, "mem_gb": config.MEM_BUDGET_GB},
            "mlx_version": mlx_ver,
            "gpu_contamination": contamination,
        }
        out_path = Path(a.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        log.info("benchmark: %s", json.dumps(report))
        # Exit non-zero on total failure so CI/release gates can detect it.
        return 0 if frames > 0 else 1
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
