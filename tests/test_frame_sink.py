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


def test_zerocopy_sink_factory():
    # fusion-mlx #913 bridge; one-copy fallback when the native ext is absent.
    try:
        sink = make_sink("zerocopy", width=64, height=48)
    except Exception as e:
        assert False, f"ZeroCopySink init failed: {e}"
    assert sink.width == 64 and sink.height == 48
    sink.emit(np.zeros((48, 64, 3), dtype=np.uint8), 0.5)
    assert sink.last_pts == 0.5


def test_make_sink_factory():
    assert isinstance(make_sink("numpy"), NumpyFrameSink)
