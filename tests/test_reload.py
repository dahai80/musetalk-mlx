from collections import deque

import numpy as np

from musetalk_mlx.pipeline.session import MuseTalkSession


class _MockPipe:
    def __init__(self, tag="init"):
        self.tag = tag

    def generate_faces(self, latent, chunk):
        return np.zeros((1, 256, 256, 3), dtype=np.uint8)


def _make_session(tag="init"):
    # Build a session shell without fusion-mlx/weights (bypasses __init__).
    s = MuseTalkSession.__new__(MuseTalkSession)
    s.pipe = _MockPipe(tag)
    s._weights_dir = "w"
    s._mlx_dir = None
    s._pending = deque()
    s._build_calls = []
    return s


def _patch_build(s, fail_on=None):
    def _build(wd, md):
        s._build_calls.append((wd, md))
        if wd == fail_on:
            raise RuntimeError("boom")
        return _MockPipe(wd)

    s._build_pipe = staticmethod(_build)


def test_reload_swaps_pipe():
    s = _make_session("init")
    _patch_build(s)
    ok = s.reload(weights_dir="new")
    assert ok
    assert s.pipe.tag == "new"
    assert s._build_calls == [("new", None)]


def test_reload_drains_pending():
    s = _make_session("init")
    _patch_build(s)
    s._pending.extend([("c1", 0.0), ("c2", 0.03), ("c3", 0.06)])
    s.reload(weights_dir="new")
    assert len(s._pending) == 0


def test_reload_failure_keeps_old_pipe():
    s = _make_session("init")
    old = s.pipe
    _patch_build(s, fail_on="BAD")
    ok = s.reload(weights_dir="BAD")
    assert ok is False
    assert s.pipe is old  # no black screen: old pipe retained
    assert len(s._pending) == 0  # still drained


def test_reload_persists_dirs():
    s = _make_session("init")
    _patch_build(s)
    s.reload(mlx_dir="/tmp/mlx")
    assert s._mlx_dir == "/tmp/mlx"


def test_reload_defaults_to_current_dirs():
    s = _make_session("init")
    _patch_build(s)
    s.reload()  # no args -> reuse current
    assert s._build_calls == [("w", None)]
