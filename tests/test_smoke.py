import numpy as np

from musetalk_mlx import config
from musetalk_mlx.utils.audio import AudioWindower
from musetalk_mlx.utils.thermal import ThermalController, thermal_tier


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


def test_thermal_controller_hysteresis_no_flap(monkeypatch):
    # audit A1: the stateless thermal_tier() chatters fair<->serious; the
    # controller must hold tier stable across an oscillating raw signal and
    # NEVER jump normal->critical in one step (FR-END-003 invariant).
    ctrl = ThermalController()
    seq = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  # fair/serious chatter
    held = [ctrl.step() for _ in seq]
    # chatter around normal/serious must not promote past serious
    assert all(t <= config.THERMAL_SERIOUS for t in held)
    # and must not flap every reading (some consecutive equal)
    assert any(held[i] == held[i + 1] for i in range(len(held) - 1))


def test_thermal_controller_no_tier_skip(monkeypatch):
    # audit A1/A4: a raw critical reading must not jump straight to critical
    # from normal — the controller steps one tier at a time.
    ctrl = ThermalController()
    # hold normal long enough to satisfy MIN_HOLD_STEPS
    for _ in range(ctrl.MIN_HOLD_STEPS + 1):
        assert ctrl.step() == config.THERMAL_NORMAL
    # now force a sustained critical raw signal
    monkeypatch.setattr("musetalk_mlx.utils.thermal.thermal_tier", lambda: config.THERMAL_CRITICAL)
    for _ in range(ctrl.PROMOTE_CONFIRM):
        t = ctrl.step()
    # after confirm, tier is normal+1 = serious, NOT critical (one-tier step)
    assert t == config.THERMAL_SERIOUS


def test_session_import():
    from musetalk_mlx.pipeline.session import MuseTalkSession

    assert MuseTalkSession is not None
