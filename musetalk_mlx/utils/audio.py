import logging

import numpy as np

from .. import config

log = logging.getLogger(__name__)


class AudioWindower:
    # 16k mono PCM -> 5s windows, consumed in 33ms steps (PRD FR-END-001).
    # TODO(phase2): prefix context cache — splice the previous window's tail
    # audio embedding as the next window's prefix to smooth the Transformer
    # slice boundary (lip teleport / rollback at window edges).

    def __init__(self, sr: int = config.SR, window_s: float = config.WINDOW_S, fps: int = config.FPS):
        self.sr = sr
        self.fps = fps
        # Step must divide the sample clock by fps exactly (533 @ 16k/30fps =
        # 33.3ms). A 5s window is then an integer number of steps (150).
        self.step = max(1, round(sr / fps))
        n_steps = max(1, round(window_s * fps))
        self.window = self.step * n_steps
        self.buf = np.zeros(0, dtype=np.float32)
        self.consumed = 0

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        self.buf = pcm if self.buf.size == 0 else np.concatenate([self.buf, pcm])

    def pop_window(self):
        """Return (window, pts_start) once a full window is buffered, else (None, None)."""
        if self.buf.size < self.window:
            return None, None
        w = self.buf[: self.window].copy()
        self.buf = self.buf[self.window :]
        pts = self.consumed / self.sr
        self.consumed += self.window
        log.debug("audio window pts=%.3fs len=%d", pts, w.size)
        return w, pts

    def step_pts(self) -> float:
        """PTS of the next 33ms step, derived from consumed audio samples."""
        return self.consumed / self.sr
