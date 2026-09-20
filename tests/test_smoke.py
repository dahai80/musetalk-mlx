import numpy as np

from musetalk_mlx import config
from musetalk_mlx.utils.audio import AudioWindower
from musetalk_mlx.utils.thermal import thermal_tier


def test_windower_alignment():
    w = AudioWindower()
    assert w.step == round(config.SR / config.FPS)  # 533 samples = 33.3ms
    # window sample-exact so window*fps/sr is an integer (no per-window drift)
    assert w.window == round(config.SR * config.WINDOW_S)
    assert (w.window * config.FPS) % config.SR == 0


def test_windower_emit():
    w = AudioWindower()
    assert w.pop_window() == (None, None)
    w.push(np.zeros(w.window, dtype=np.float32))
    win, pts = w.pop_window()
    assert win.shape == (w.window,)
    assert pts == 0.0
    assert w.pop_window() == (None, None)


def test_thermal_tier_bounded():
    assert thermal_tier() in (config.THERMAL_NORMAL, config.THERMAL_SERIOUS, config.THERMAL_CRITICAL)


def test_session_import():
    from musetalk_mlx.pipeline.session import MuseTalkSession

    assert MuseTalkSession is not None
