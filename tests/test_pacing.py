import time
from itertools import pairwise

from musetalk_mlx.pipeline.pacing import PacedPublisher


def test_tick_publishes_and_paces():
    frames = iter([(f"f{i}", i / 30.0) for i in range(30)])
    pub = []
    p = PacedPublisher(30)
    p.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5 and len(pub) < 30:
        p.tick(lambda: next(frames, None), lambda f, pts: pub.append((f, pts)))
    assert len(pub) == 30
    # 30 frames at 30fps ~ 1s; allow generous CI margin (GitHub runners are
    # shared and can stall >1s on a single tick — 1.5s flaked on 35697242110).
    assert time.monotonic() - t0 < 2.5
    assert p.published == 30
    assert p.summary()["fps"] > 20


def test_no_frame_still_paces_no_burst():
    pub = []
    p = PacedPublisher(30)
    p.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.3:
        p.tick(lambda: None, lambda f, pts: pub.append((f, pts)))
    assert pub == []
    assert p.overruns <= 1  # None frames pace normally; <=1 tolerates CI scheduler jitter


def test_resync_after_slow_render_no_burst():
    # get_frame blocks 100ms (slower than the 33ms period): the pacer must
    # resync (deadline := now+period) and NOT emit a burst of back-to-back
    # publishes afterwards.
    pub = []

    def slow_get():
        time.sleep(0.1)
        return ("f", 0.0)

    p = PacedPublisher(30)
    p.start()
    for _ in range(3):
        p.tick(slow_get, lambda f, pts: pub.append(time.monotonic()))
    assert p.overruns >= 2
    assert p.overrun_ratio > 0
    gaps = [b - a for a, b in pairwise(pub)]
    assert all(g >= p.period * 0.5 for g in gaps)  # no zero-gap burst


def test_report_written(tmp_path):
    from musetalk_mlx.pipeline.pacing import PacedPublisher as P

    report = tmp_path / "pacing.json"
    p = P(30, report_path=str(report))
    p.start()
    p.tick(lambda: ("f", 0.0), lambda f, pts: None)
    p.write_report()
    assert report.exists()
    s = __import__("json").loads(report.read_text())
    assert s["published"] == 1
