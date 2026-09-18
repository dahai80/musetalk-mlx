import numpy as np

from musetalk_mlx.pipeline.frame_sink import CVPixelBufferSink, NumpyFrameSink, make_sink


def test_numpy_sink_records_pts():
    sink = NumpyFrameSink()
    f = np.zeros((1080, 1920, 3), dtype=np.uint8)
    sink.emit(f, 1.23)
    sink.emit(f, 2.34)
    assert len(sink.frames) == 2
    assert sink.frames[0][1] == 1.23
    assert sink.last_pts == 2.34


def test_numpy_sink_copies():
    sink = NumpyFrameSink()
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    sink.emit(f, 0.0)
    f[0, 0] = 255  # mutate after emit
    assert sink.frames[0][0][0, 0, 0] == 0  # stored copy unaffected


def test_cvpixel_sink_pts_preserved():
    sink = CVPixelBufferSink(1920, 1080)
    sink.emit(np.zeros((1080, 1920, 3), dtype=np.uint8), 5.5)
    assert sink.last_pts == 5.5


def test_make_sink_factory():
    assert isinstance(make_sink("numpy"), NumpyFrameSink)
    assert isinstance(make_sink("cvpixel", width=100, height=100), CVPixelBufferSink)
