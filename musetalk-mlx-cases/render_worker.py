#!/usr/bin/env python3
"""Render worker: audio + base video -> silent video (called as subprocess by
inference.py).

MuseTalkSession's MLX render thread can trigger a C++-layer
"There is no Stream(gpu, 0)" abort at session teardown, killing the whole
process. Running in a dedicated subprocess isolates that crash so the parent's
ffmpeg audio mux is unaffected.

Usage: python render_worker.py --video V --audio A --out TMP.mp4 [--fps F]
"""

import argparse
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import librosa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render")

# musetalk-mlx-cases lives inside the musetalk-mlx repo, so weights are one
# level up (../weights-mlx). Relative keeps a public clone runnable; override
# via MT_MLX_DIR for non-standard layouts.
_PARENT = Path(__file__).resolve().parent.parent
DEFAULT_MLX_DIR = os.environ.get("MT_MLX_DIR", str(_PARENT / "weights-mlx"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=0, help="0 = auto-detect video fps")
    p.add_argument("--mlx-dir", default=DEFAULT_MLX_DIR)
    p.add_argument("--bbox-shift", type=int, default=0, help="face crop y-shift (sun2 case uses -7)")
    a = p.parse_args()

    import mlx.core as mx

    # Bind the GPU default device on the main thread before the render thread
    # spawns — mitigates the "There is no Stream(gpu, 0)" abort at teardown.
    mx.set_default_device(mx.gpu)

    from musetalk_mlx import MuseTalkSession, config
    from musetalk_mlx.face.crop import FaceCropper

    def probe_fps(path: str) -> int:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            num, den = out.split("/")
            return max(1, round(int(num) / int(den)))
        except Exception:
            return 30

    fps = a.fps or probe_fps(a.video)
    # bbox-shift must take effect before session __init__: __init__ builds the
    # default FaceCropper and runs bg precompute, so swapping session._cropper
    # afterwards has no effect. Patch the module attr the session imports from.
    if a.bbox_shift:
        import musetalk_mlx.pipeline.session as sess_mod

        _orig = FaceCropper

        def _patched(*args, **kw):
            return _orig(upperbondrange=a.bbox_shift)

        sess_mod.FaceCropper = _patched
        log.info("bbox_shift=%d patched into FaceCropper", a.bbox_shift)

    session = MuseTalkSession(None, a.video, fps=fps, mlx_dir=a.mlx_dir)
    wav, _ = librosa.load(a.audio, sr=16000)
    log.info("audio %.1fs, fps=%d, window=%.0fs", len(wav) / 16000, fps, config.WINDOW_S)

    # Chunked push: the session audio buffer caps at ~4 windows (~20s); pushing
    # a long clip all at once overflows and drops samples. Feed ahead of the
    # rendered frame count to keep the buffer at low watermark.
    SR = 16000
    CHUNK_SEC = 4.0
    pushed = 0
    out_path = Path(a.out)
    # ffmpeg rawvideo pipe instead of imageio: the teardown C++ abort kills
    # this process directly, and imageio would not finish writing the mp4 moov
    # index, leaving an invalid file. Pipe mode: the abort triggers EOF, ffmpeg
    # finalizes and writes a complete file regardless.
    writer = None
    ffmpeg_err = None

    def open_writer(frame) -> subprocess.Popen:
        h, w = frame.shape[:2]
        nonlocal ffmpeg_err
        ffmpeg_err = tempfile.NamedTemporaryFile(suffix="_ffmpeg_err.log", delete=False, mode="w")
        return subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "warning",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
            stderr=open(ffmpeg_err.name, "w"),
        )

    n = 0

    def _valid(p: Path) -> bool:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
            capture_output=True,
            text=True,
        )
        try:
            return r.returncode == 0 and float(r.stdout.strip()) > 0
        except ValueError:
            return False

    # Crash localization: rendering the tail zero-padding window AFTER
    # end_of_stream() reliably triggers the "Stream(gpu, 0)" abort. Workaround:
    # feed audio only up to the last full window (never call end_of_stream);
    # once the expected frame count is reached, tear down and os._exit past
    # session.close/atexit (they hit the same abort path).
    rendered_sec = len(wav) / SR - config.WINDOW_S
    if rendered_sec <= 0:
        log.error("audio too short: %.1fs < one %.0fs window", len(wav) / SR, config.WINDOW_S)
        return 1
    expected_frames = int(rendered_sec * fps)
    stall = 0
    try:
        while n < expected_frames:
            # Feed audio: keep ~8s of unconsumed headroom, under the buffer cap
            want_frames = int(n + 8 * fps)
            want_samples = min(len(wav), int(want_frames / fps * SR))
            if pushed < want_samples:
                end = min(len(wav), max(want_samples, pushed + int(CHUNK_SEC * SR)))
                session.push_audio(wav[pushed:end])
                pushed = end

            out = session.get_output_frame()
            if out is None:
                stall += 1
                if stall > 150:  # ~5s with no output: abandon this attempt
                    log.warning("producer stalled %d polls; bail out", stall)
                    break
                time.sleep(0.033)
                continue
            stall = 0
            frame, pts = out
            if writer is None:
                writer = open_writer(frame)
            writer.stdin.write(frame.tobytes())  # frame is already BGR
            n += 1
            if n % 100 == 0:
                log.info("frame %d pts=%.3fs", n, pts)
    finally:
        if writer is not None:
            try:
                writer.stdin.close()
                writer.wait(timeout=60)
            except Exception:
                pass
            # ffmpeg finalization (moov write) takes time; poll until ffprobe
            # can read the file.
            for _ in range(50):
                if out_path.exists() and _valid(out_path):
                    break
                time.sleep(0.2)
    log.info("wrote %d frames (%.1fs) -> %s", n, n / fps, out_path)
    # Tear down the paste subprocess explicitly BEFORE os._exit: os._exit skips
    # atexit handlers, so the PasteProcess child would be orphaned. Kill it
    # directly so no zombie lingers (audit: session.close hits the same abort,
    # so we bypass it).
    _hard_teardown(session)
    if not out_path.exists():
        return 1
    os._exit(0)


def _hard_teardown(session) -> None:
    # Kill the paste subprocess + worker directly without invoking the
    # session.close() path (which triggers the C++ stream abort). Best-effort:
    # any error here is non-fatal since we os._exit next.
    try:
        pp = getattr(session, "_paste_proc", None)
        if pp is not None and getattr(pp, "_proc", None) is not None:
            pp._proc.kill()
            pp._proc.wait(timeout=5)
    except Exception as e:
        log.warning("paste proc kill failed (%s); may orphan child", e)


if __name__ == "__main__":
    raise SystemExit(main())
