import threading

import numpy as np

from musetalk_mlx.utils.audio import AudioWindower


def test_windower_reset_discards_buffer_keeps_consumed():
    w = AudioWindower(sr=16000, fps=30)
    w.push(np.zeros(16000 * 6, dtype=np.float32))  # > 1 window
    win, pts0 = w.pop_window()
    assert win is not None
    consumed0 = w.consumed
    w.push(np.ones(1000, dtype=np.float32))
    w.reset()
    assert w._pending == 0
    assert w._tail.size == 0
    assert w.prefix_samples == 0
    assert w.consumed == consumed0  # monotonic, never rewound
    # next window starts only after a FULL window (tail was dropped)
    w.push(np.zeros(16000 * 5, dtype=np.float32))
    win2, pts1 = w.pop_window()
    assert win2 is not None
    assert pts1 > pts0


def test_windower_reset_empty_noop():
    w = AudioWindower(sr=16000, fps=30)
    w.reset()
    assert w._pending == 0


def test_session_interrupt_clears_queues():
    import queue
    from collections import deque

    from musetalk_mlx.pipeline.session import MuseTalkSession

    s = MuseTalkSession.__new__(MuseTalkSession)
    s._closed = False
    s._pending = deque([("c1", 0.0), ("c2", 0.03)])
    s._inflight = deque([object()])
    s._out_q = deque([(object(), 0.1), (object(), 0.13)])
    s._expected_pts = 0.2
    s._emit_lock = threading.RLock()
    s._paste_q = queue.Queue(maxsize=8)
    s._paste_q.put(("job",))
    s._paste_q.put(("job",))
    s._audio_prefix = object()
    s._windower = AudioWindower(sr=16000, fps=30)
    s._windower.push(np.ones(1000, dtype=np.float32))
    s._last_frame = object()
    s._paste_worker_alive = True

    dropped = s.interrupt()

    assert dropped == 2
    assert len(s._pending) == 0
    assert len(s._inflight) == 0
    assert len(s._out_q) == 0
    assert s._expected_pts is None
    assert s._audio_prefix is None
    assert s._windower._pending == 0
    assert s._last_frame is not None  # standby frame kept
    assert s._paste_q.qsize() == 0  # lightly drained
    assert s._paste_worker_alive  # worker survives (generation unchanged)


def test_session_interrupt_closed_noop():
    from collections import deque

    from musetalk_mlx.pipeline.session import MuseTalkSession

    s = MuseTalkSession.__new__(MuseTalkSession)
    s._closed = True
    s._pending = deque([("c1", 0.0)])
    assert s.interrupt() == 0
    assert len(s._pending) == 1  # untouched
