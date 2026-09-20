import argparse
import logging
import sys
from pathlib import Path

import imageio
import librosa

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Offline MuseTalk inference: audio + base video -> output video")
    p.add_argument(
        "--weights", default=None, help="weights root (from_pretrained layout; omit with --mlx-dir)"
    )
    p.add_argument("--mlx-dir", default=None, help="MLX dist dir (musetalk-mlx-convert output); torch-free")
    p.add_argument("--audio", required=True, help="input wav (any sr, resampled to 16k)")
    p.add_argument("--video", required=True, help="base video (looped)")
    p.add_argument("--out", default="results/output.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--bbox-shift", type=int, default=0, help="upperbondrange passthrough (face crop y-shift)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all audio)")
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession
    from musetalk_mlx.face.crop import FaceCropper

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = MuseTalkSession(a.weights, a.video, fps=a.fps, mlx_dir=a.mlx_dir)
    session._cropper = FaceCropper(upperbondrange=a.bbox_shift)
    wav, _ = librosa.load(a.audio, sr=16000)
    log.info("audio %.1fs pushed into session", len(wav) / 16000)
    session.push_audio(wav)

    writer = imageio.get_writer(str(out_path), fps=a.fps)
    n = 0
    flushed = False
    try:
        while True:
            if a.max_frames and n >= a.max_frames:
                break
            out = session.get_output_frame()
            if out is None:
                if flushed:
                    break
                # first drain complete: zero-pad + render the sub-window tail
                session.end_of_stream()
                flushed = True
                continue
            frame, pts = out
            if n % 100 == 0:
                log.info("frame %d pts=%.3fs", n, pts)
            writer.append_data(frame[..., ::-1])  # BGR -> RGB
            n += 1
    finally:
        writer.close()
    tail = session._windower.buf.size / session._windower.sr
    if tail > 0:
        log.warning("%.2fs trailing audio left unprocessed (< 1 window)", tail)
    log.info("wrote %d frames (%.1fs) to %s", n, n / a.fps, out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
