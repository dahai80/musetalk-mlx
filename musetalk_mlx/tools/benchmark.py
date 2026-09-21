import argparse
import json
import logging
import sys
from pathlib import Path

import librosa

from musetalk_mlx import config
from musetalk_mlx.utils.profiling import phys_footprint

log = logging.getLogger(__name__)


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
        report = {
            "frames": frames,
            "elapsed_s": elapsed,
            "fps": fps,
            "meets_30fps": fps >= 30,
            "mem_mb": phys_footprint() // 1024**2,
            "stages": {
                k: {"n": v[0], "avg_ms": v[1] / v[0] * 1000, "max_ms": v[2] * 1000} for k, v in stats.items()
            },
            "budgets": {"fps": config.FPS, "mem_gb": config.MEM_BUDGET_GB},
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
