import logging

import numpy as np

from .. import config

log = logging.getLogger(__name__)


class AudioWindower:
    # 16k mono PCM -> overlapping 5s windows, consumed in 33ms steps (FR-END-001).
    # hop = window - overlap so the boundary slice is re-encoded, smoothing the
    # window-edge mouth jump at the sample level. The embedding-level prefix
    # cache (fusion-mlx #914) splices the prior window's tail embedding; until
    # then, last_window_tail() exposes the raw tail for the session to hand off.

    def __init__(self, sr=config.SR, window_s=config.WINDOW_S, fps=config.FPS, overlap_s=config.OVERLAP_S):
        self.sr = sr
        self.fps = fps
        self.step = max(1, round(sr / fps))
        n_steps = max(1, round(window_s * fps))
        self.window = self.step * n_steps
        overlap = max(0, round(overlap_s * sr))
        overlap = min(overlap, self.window - self.step)
        self.hop = self.window - overlap
        self.overlap = overlap
        self.buf = np.zeros(0, dtype=np.float32)
        self.consumed = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self.prefix_samples = 0  # tail prepended on the last pop (for session chunk-skip)
        log.debug(
            "AudioWindower window=%d hop=%d overlap=%d step=%d",
            self.window,
            self.hop,
            self.overlap,
            self.step,
        )

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        self.buf = pcm if self.buf.size == 0 else np.concatenate([self.buf, pcm])

    def pop_window(self):
        """Return (window, pts_start) once a full window is buffered, else (None, None).

        Prepends the previous window's overlap tail so consecutive windows share
        boundary audio (FR-END-001). pts_start is the PTS of the *new* audio in
        the window (the tail is re-emitted only for smoothing, not re-counted)."""
        # First window (no tail) consumes a full window; later windows consume a
        # hop and prepend the prior overlap tail for boundary smoothing.
        need = self.window if self._tail.size == 0 else self.hop
        if self.buf.size < need:
            return None, None
        take = need
        new = self.buf[:take]
        self.buf = self.buf[take:]
        prefix = self._tail.size
        if prefix:
            w = np.concatenate([self._tail, new])
        else:
            w = new.copy()
        if w.size > self.window:
            w = w[-self.window :]
            prefix = max(0, prefix - (w.size - self.window))
        pts = self.consumed / self.sr
        self.consumed += take
        self._tail = w[-self.overlap :] if self.overlap else np.zeros(0, dtype=np.float32)
        self.prefix_samples = prefix
        log.debug("audio window pts=%.3fs len=%d (overlap=%d)", pts, w.size, self._tail.size)
        return w, pts

    def last_window_tail(self) -> np.ndarray:
        """Overlap tail of the most recent window (raw PCM) — prefix handoff to fusion-mlx #914."""
        return self._tail.copy()

    def step_pts(self) -> float:
        """PTS of the next 33ms step, derived from consumed audio samples."""
        return self.consumed / self.sr
