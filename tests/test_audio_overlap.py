import numpy as np

from musetalk_mlx import config
from musetalk_mlx.utils.audio import AudioWindower


def test_windower_hop_lt_window():
    w = AudioWindower()
    assert w.window == w.step * round(config.WINDOW_S * config.FPS)
    assert w.hop < w.window
    assert w.hop == w.window - round(config.OVERLAP_S * config.SR)


def test_windower_no_samples_dropped():
    w = AudioWindower()
    total = config.SR * 12  # 12s
    w.push(np.ones(total, dtype=np.float32))
    emitted_new = 0
    pts_seen = []
    while True:
        win, pts = w.pop_window()
        if win is None:
            break
        emitted_new += win.size - w.prefix_samples
        pts_seen.append(pts)
    # no samples dropped or double-counted: emitted + trailing buffer == total
    assert emitted_new + w.buf.size == total
    assert pts_seen == sorted(pts_seen)


def test_windower_pts_monotonic_and_tail():
    w = AudioWindower()
    w.push(np.ones(config.SR * 11, dtype=np.float32))
    prev_pts = -1.0
    tails = 0
    while True:
        win, pts = w.pop_window()
        if win is None:
            break
        assert pts > prev_pts
        prev_pts = pts
        if w.prefix_samples > 0:
            tails += 1
    assert tails >= 1  # at least one overlapping window
    assert w.last_window_tail().size == round(config.OVERLAP_S * config.SR)


def test_windower_first_window_no_prefix():
    w = AudioWindower()
    w.push(np.ones(w.window, dtype=np.float32))
    win, pts = w.pop_window()
    assert win is not None
    assert w.prefix_samples == 0  # first window has no tail
    assert pts == 0.0
    assert w.pop_window() == (None, None)
