import argparse
import logging
import subprocess
import sys
from pathlib import Path

import imageio
import librosa

log = logging.getLogger(__name__)


def _mux_audio(video_path: Path, audio_path: str, fps: int) -> None:
    # imageio writes a video-only stream; a lip-sync product without audio is
    # useless (audit B1). Mux the source audio in via ffmpeg (list-form, no
    # shell). Mirrors eval._ensure_audio. -shortest trims audio to video length.
    if not Path(audio_path).exists():
        log.warning("source audio %s missing; output stays video-only", audio_path)
        return
    has_audio = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        != ""
    )
    if has_audio:
        log.info("output already has audio; skip mux")
        return
    tmp = video_path.with_suffix(".muxed.mp4")
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        str(tmp),
    ]
    log.info("muxing source audio %s into %s", audio_path, video_path)
    subprocess.run(cmd, check=True)
    tmp.replace(video_path)
    log.info("audio muxed -> %s", video_path)


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
    p.add_argument(
        "--sink",
        default="zerocopy",
        choices=["zerocopy", "numpy"],
        help="egress sink: zerocopy (MetalZeroCopyBridge #913, FR-LK-001) or numpy",
    )
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession
    from musetalk_mlx.face.crop import FaceCropper
    from musetalk_mlx.pipeline.frame_sink import make_sink

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = MuseTalkSession(a.weights, a.video, fps=a.fps, mlx_dir=a.mlx_dir)
    session._cropper = FaceCropper(upperbondrange=a.bbox_shift)
    wav, _ = librosa.load(a.audio, sr=16000)
    log.info("audio %.1fs pushed into session", len(wav) / 16000)
    session.push_audio(wav)

    # Attach the egress sink (FR-LK-001). ZeroCopySink routes each emitted
    # frame through the #913 MetalZeroCopyBridge (IOSurface-backed CVPixelBuffer)
    # — the production egress abstraction. The imageio writer below still
    # consumes via get_output_frame for the mp4 mux; the sink is the
    # zero-copy path for future native VideoToolbox direct-encoding. numpy sink
    # is the torch-free fallback when the bridge is unavailable.
    first_bg = session._bg_frame()
    sink = make_sink(a.sink, width=first_bg.shape[1], height=first_bg.shape[0])
    session.set_sink(sink)

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
        sink.close()
        session.set_sink(None)
    tail = session._windower.buf.size / session._windower.sr
    if tail > 0:
        log.warning("%.2fs trailing audio left unprocessed (< 1 window)", tail)
    log.info("wrote %d frames (%.1fs) to %s", n, n / a.fps, out_path)
    _mux_audio(out_path, a.audio, a.fps)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
