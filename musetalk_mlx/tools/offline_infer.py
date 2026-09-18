import argparse
import logging
from pathlib import Path

import imageio
import librosa

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline MuseTalk inference: audio + base video -> output video")
    p.add_argument("--weights", required=True, help="weights root (from_pretrained layout)")
    p.add_argument("--mlx-dir", default=None, help="MLX dist dir (musetalk-mlx-convert output); torch-free")
    p.add_argument("--audio", required=True, help="input wav (any sr, resampled to 16k)")
    p.add_argument("--video", required=True, help="base video (looped)")
    p.add_argument("--out", default="results/output.mp4")
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = MuseTalkSession(a.weights, a.video, fps=a.fps, mlx_dir=a.mlx_dir)
    wav, _ = librosa.load(a.audio, sr=16000)
    log.info("audio %.1fs pushed into session", len(wav) / 16000)
    session.push_audio(wav)

    writer = imageio.get_writer(str(out_path), fps=a.fps)
    n = 0
    while True:
        out = session.get_output_frame()
        if out is None:
            break
        frame, pts = out
        if n % 100 == 0:
            log.info("frame %d pts=%.3fs", n, pts)
        writer.append_data(frame[..., ::-1])  # BGR -> RGB
        n += 1
    writer.close()
    tail = session._windower.buf.size / session._windower.sr
    if tail > 0:
        log.warning("%.2fs trailing audio left unprocessed (< 1 window)", tail)
    log.info("wrote %d frames (%.1fs) to %s", n, n / a.fps, out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
