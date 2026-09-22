#!/usr/bin/env python3
"""Eye-motion amplification (simplified Eulerian video magnification): bandpass
the eye ROI temporally and amplify, making blinks more visible. Used for
MuseTalk base videos where the blink amplitude is too small.

Streams frame-by-frame through ffmpeg instead of loading the whole video into
memory (long clips would OOM).

Usage: python amplify_eyes.py --in sun.mp4 --out sun_amp.mp4 [--alpha 2.5]
"""

import argparse
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def amplify(in_path: Path, out_path: Path, alpha: float) -> None:
    cap = cv2.VideoCapture(str(in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"streaming {total} frames @{fps:.1f}fps {w}x{h}", file=sys.stderr)

    # Eye ROI (matches the quant script): height 25%-45%, width 20%-80%
    y0, y1, x0, x1 = int(h * 0.25), int(h * 0.45), int(w * 0.2), int(w * 0.8)

    # Rolling temporal low-pass of the ROI (Gaussian over the last few frames).
    # A full temporal stack would OOM on long clips; a deque of blurred ROIs
    # approximates the low-frequency component that blinks deviate from.
    LPF_SIGMA = 4.0
    LPF_WINDOW = 8  # frames; blink ~3-7Hz @25fps, low-pass cutoff ~8 frames

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
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
    )

    roi_buf = deque(maxlen=LPF_WINDOW)  # blurred ROIs for the running low-pass
    n = 0
    try:
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            fr = fr.astype(np.float32)
            roi = fr[y0:y1, x0:x1, :]
            blurred = cv2.GaussianBlur(roi, (0, 0), LPF_SIGMA)
            roi_buf.append(blurred)
            if len(roi_buf) >= 2:
                # Running mean of the buffered low-pass = the low-frequency bg.
                lpf = np.mean(np.stack(roi_buf), axis=0)
                motion = roi - lpf  # high-frequency motion (blink edges)
                fr[y0:y1, x0:x1, :] = roi + alpha * motion
            proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
            n += 1
    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()
    print(f"wrote {n} frames -> {out_path}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=2.5, help="amplification factor")
    a = p.parse_args()
    amplify(Path(a.inp), Path(a.out), a.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
