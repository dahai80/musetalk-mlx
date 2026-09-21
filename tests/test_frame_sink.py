import numpy as np

from musetalk_mlx.pipeline.frame_sink import NumpyFrameSink, make_sink


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


def test_zerocopy_kind_falls_back_to_numpy():
    # ZeroCopySink was removed (audit 0921 DC1: dead code, #913 bridge not on
    # the Python production path). "zerocopy" now falls back to NumpyFrameSink.
    sink = make_sink("zerocopy", width=64, height=48)
    assert isinstance(sink, NumpyFrameSink)
    sink.emit(np.zeros((48, 64, 3), dtype=np.uint8), 0.5)
    assert sink.last_pts == 0.5


def test_make_sink_factory():
    assert isinstance(make_sink("numpy"), NumpyFrameSink)
