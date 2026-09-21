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


def test_set_sink_feeds_both_out_q_and_sink():
    # DC1 fix: ZeroCopySink/make_sink must have a production consumer.
    # session.set_sink attaches a sink; _emit feeds both _out_q (for
    # get_output_frame) and the sink. Verify the dual-path with a lightweight
    # recording sink (no model/bridge needed).
    from musetalk_mlx.pipeline.frame_sink import FrameSink

    class _Recorder(FrameSink):
        def __init__(self):
            self.emitted = []

        def emit(self, frame_bgr, pts):
            self.emitted.append((pts, frame_bgr[0, 0, 0]))

    rec = _Recorder()
    # Stub a minimal session-like object: just _emit + _out_q + _emit_lock
    import threading
    from collections import deque

    import numpy as np

    class _Stub:
        pass

    s = _Stub()
    s._sink = None
    s._out_q = deque()
    s._emit_lock = threading.RLock()
    s._out_q_cap = 8
    s._expected_pts = None
    # Bind the real _emit (read it from the module source to avoid full init)
    import musetalk_mlx.pipeline.session as sess_mod

    # _emit references self.profiler.tick — stub it
    class _Prof:
        def tick(self):
            pass

    s.profiler = _Prof()
    s.step = 1
    s.sr = 30
    # Copy the _emit method from MuseTalkSession (unbound)
    _emit = sess_mod.MuseTalkSession._emit
    s.set_sink = lambda sink: setattr(s, "_sink", sink)
    s.set_sink(rec)
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    _emit(s, f, 1.0)
    _emit(s, f, 2.0)
    assert len(s._out_q) == 2, "out_q must still get frames (dual-path)"
    assert len(rec.emitted) == 2, "sink must get frames"
    assert rec.emitted[0][0] == 1.0 and rec.emitted[1][0] == 2.0
    # get_output_frame still works from out_q
    assert s._out_q[0][1] == 1.0


def test_set_sink_isolation_on_emit_failure():
    # A broken sink must not drop the frame from _out_q (audit P2-9 isolation).
    import threading
    from collections import deque

    import numpy as np

    import musetalk_mlx.pipeline.session as sess_mod

    class _Broken:
        def emit(self, *a):
            raise RuntimeError("sink exploded")

    class _Prof:
        def tick(self):
            pass

    class _Stub:
        pass

    s = _Stub()
    s._sink = None
    s._out_q = deque()
    s._emit_lock = threading.RLock()
    s._out_q_cap = 8
    s._expected_pts = None
    s.profiler = _Prof()
    s.step = 1
    s.sr = 30
    s._sink = _Broken()
    _emit = sess_mod.MuseTalkSession._emit
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    _emit(s, f, 5.0)  # must not raise
    assert len(s._out_q) == 1, "out_q unaffected by sink failure"
