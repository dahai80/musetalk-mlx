import logging
import threading
import time

import librosa

from musetalk_mlx import MuseTalkSession
from musetalk_mlx.pipeline.pacing import PacedPublisher

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("nolk")

session = MuseTalkSession(
    "weights-mlx",
    "/Users/dahai/migration/MuseTalk/data/video/sun.mp4",
    fps=30,
    mlx_dir="weights-mlx",
)
wav, _ = librosa.load("/Users/dahai/migration/MuseTalk/data/audio/eng.wav", sr=16000)
chunk = session.sr * 100 // 1000


def _feeder():
    step = chunk / session.sr
    t0 = time.monotonic()
    for i, off in enumerate(range(0, len(wav), chunk)):
        session.push_audio(wav[off : off + chunk])
        target = t0 + (i + 1) * step
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
    log.info("feeder done")


threading.Thread(target=_feeder, name="audio-feeder", daemon=True).start()

pacer = PacedPublisher(30)
pacer.start()
published = 0


def _pub(frame, pts):
    global published
    published += 1
    buckets.append(time.monotonic())


buckets = []


def _report_buckets():
    if not buckets:
        return
    t0 = buckets[0]
    out = []
    start = t0
    for i in range(1, len(buckets)):
        if buckets[i] - start >= 5.0:
            out.append(i - _bucket_idx[0])
            _bucket_idx[0] = i
            start = buckets[i]
    log.info("publish per 5s bucket: %s out_q_now=%d", out[:20], 0)


_bucket_idx = [0]


try:
    while published < 1500:
        pacer.tick(session.get_output_frame, _pub)
        if published % 150 == 0:
            log.info(
                "pub=%d out_q=%d pending=%d inflight=%d elapsed=%.1fs",
                published,
                len(session._out_q),
                len(session._pending),
                len(session._inflight),
                time.monotonic() - pacer._start_t,
            )
except BaseException:
    log.exception("loop terminated")
    raise
finally:
    pacer.write_report()
    session.close()
log.info("nolk done: published=%d", published)
