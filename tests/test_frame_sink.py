import numpy as np

from musetalk_mlx.pipeline.frame_sink import NumpyFrameSink, ZeroCopySink, make_sink


def test_numpy_sink_preserves_pts():
    sink = NumpyFrameSink()
    sink.emit(np.zeros((4, 4, 3), dtype=np.uint8), 1.25)
    assert sink.last_pts == 1.25
    assert sink.frames[0][1] == 1.25


def test_numpy_sink_copies_frame():
    sink = NumpyFrameSink()
    f = np.full((4, 4, 3), 7, dtype=np.uint8)
    sink.emit(f, 0.0)
    f[:] = 0
    assert sink.frames[0][0][0, 0, 0] == 7


def test_zerocopy_sink_factory():
    # ZeroCopySink is the PRD FR-LK-001 zero-copy egress via the #913
    # MetalZeroCopyBridge. It must be wired (not deleted) — restored after the
    # audit-0921 DC1 deletion was reverted per user direction.
    sink = make_sink("zerocopy", width=64, height=48)
    assert isinstance(sink, ZeroCopySink)
    assert (sink.width, sink.height) == (64, 48)
    sink.emit(np.zeros((48, 64, 3), dtype=np.uint8), 0.5)
    assert sink.last_pts == 0.5
    sink.close()


def test_zerocopy_sink_fallback_on_bridge_missing(monkeypatch):
    # If the #913 bridge import fails, make_sink("zerocopy") falls back to
    # NumpyFrameSink (graceful degradation, not a hard crash).
    import builtins

    real_import = builtins.__import__

    def _fail(name, *a, **k):
        if name.startswith("fusion_mlx.metal.zero_copy"):
            raise ImportError("simulated bridge missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fail)
    sink = make_sink("zerocopy", width=64, height=48)
    assert isinstance(sink, NumpyFrameSink)


def test_make_sink_factory():
    assert isinstance(make_sink("numpy"), NumpyFrameSink)
