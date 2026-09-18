import argparse
import logging
import sys
import time

import librosa

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description="Realtime MuseTalk -> LiveKit (FR-LK-001/002)")
    p.add_argument("--weights", required=True)
    p.add_argument("--mlx-dir", default=None)
    p.add_argument("--video", required=True, help="base video (looped)")
    p.add_argument("--livekit-url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--audio", default=None, help="file fallback when no inbound LiveKit audio")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    a = p.parse_args()

    from musetalk_mlx import MuseTalkSession
    from musetalk_mlx.livekit.adapter import LiveKitAdapter

    session = MuseTalkSession(a.weights, a.video, fps=a.fps, mlx_dir=a.mlx_dir)
    adapter = LiveKitAdapter(a.livekit_url, a.token, width=a.width, height=a.height, fps=a.fps)
    adapter.set_audio_callback(lambda pcm, pts: session.push_audio(pcm))
    adapter.connect()
    log.info("realtime session up; feeding audio -> LiveKit")

    if a.audio:
        wav, _ = librosa.load(a.audio, sr=16000)
        session.push_audio(wav)

    period = 1.0 / a.fps
    try:
        while True:
            out = session.get_output_frame()
            if out is None:
                time.sleep(period * 0.5)
                continue
            frame, pts = out
            adapter.publish_frame(frame, pts)
            log.debug("published pts=%.3fs", pts)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
