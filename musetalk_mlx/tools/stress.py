import argparse
import json
import logging
import sys
import time
from pathlib import Path

import librosa

from musetalk_mlx.utils.profiling import max_contiguous_alloc, phys_footprint

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
    while time.monotonic() < deadline:
        out = session.get_output_frame()
        if out is not None:
            frames += 1
        now = time.monotonic()
        if now - last_push >= 1.0:
            session.push_audio(wav)  # loop the audio for a continuous stream
            last_push = now
        if now - t0 >= 60 * len(samples):
            mem = phys_footprint()
            frag_ok = max_contiguous_alloc(probe_bytes)
            samples.append({"minute": len(samples) + 1, "mem_mb": mem // 1024**2, "frag_ok": frag_ok})
            log.info(
                "minute %d: mem=%dMB frag(%dMB)=%s frames=%d",
                len(samples),
                mem // 1024**2,
                a.probe_mb,
                frag_ok,
                frames,
            )
        if frames and not session._pending and session._windower.buf.size == 0 and frames > 0:
            session.push_audio(wav)
    elapsed = time.monotonic() - t0
    if len(samples) >= 2:
        leak = samples[-1]["mem_mb"] - samples[1]["mem_mb"]
        ok = leak * 1024**2 <= LEAK_BUDGET_BYTES and all(s["frag_ok"] for s in samples)
    report = {
        "duration_min": elapsed / 60,
        "frames": frames,
        "avg_fps": frames / elapsed if elapsed else 0,
        "mem_samples": samples,
        "leak_mb": leak,
        "frag_probe_mb": a.probe_mb,
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
