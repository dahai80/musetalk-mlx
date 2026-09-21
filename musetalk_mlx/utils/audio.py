import logging
import threading

import numpy as np

from .. import config

log = logging.getLogger(__name__)

# Backpressure cap: if buffered audio exceeds this many samples, drop the
# oldest (streaming must not OOM when inference pauses on thermal throttle).
_MAX_BUF_SAMPLES = int(config.SR * config.WINDOW_S * 4)


class AudioWindower:
    # 16k mono PCM -> overlapping 5s windows, consumed in 33ms steps (FR-END-001).
    # hop = window - overlap so the boundary slice is re-encoded, smoothing the
    # window-edge mouth jump at the sample level. The embedding-level prefix
    # cache (fusion-mlx #914) splices the prior window's tail embedding; until
    # then, last_window_tail() exposes the raw tail for the session to hand off.
    #
    # Thread safety: push() runs on the LiveKit audio-callback thread while
    # pop_window()/flush() run on the render thread. All state is guarded by
    # _lock (push_audio and get_output_frame cross threads in realtime mode).

    def __init__(self, sr=config.SR, window_s=config.WINDOW_S, fps=config.FPS, overlap_s=config.OVERLAP_S):
        if sr <= 0:
            raise ValueError(f"sr must be > 0, got {sr}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.sr = sr
        self.fps = fps
        self.step = max(1, round(sr / fps))
        self.window = max(1, round(sr * window_s))
        overlap = max(0, round(overlap_s * sr))
        overlap = max(0, min(overlap, self.window - self.step))
        self.hop = self.window - overlap
        if self.hop <= 0:
            raise ValueError(f"hop must be > 0 (window={self.window} overlap={overlap} step={self.step})")
        self.overlap = overlap
        # Chunked buffer (audit 0921 M-7): inbound audio arrives in 20-40ms
        # packets; the old `np.concatenate([buf, pcm])` on every push copied
        # the whole ~160KB buffer per packet (O(n^2) across a 5s window).
        # Chunks accumulate and join once per pop_window/flush instead.
        self._chunks: list = []
        self._pending = 0  # total samples across _chunks
        self.consumed = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self.prefix_samples = 0
        self._lock = threading.Lock()
        log.debug(
            "AudioWindower window=%d hop=%d overlap=%d step=%d",
            self.window,
            self.hop,
            self.overlap,
            self.step,
        )

    @property
    def buf(self) -> np.ndarray:
        # Joined view of pending samples (kept for introspection/tests; the hot
        # path never materializes the whole buffer).
        with self._lock:
            return self._join_locked()

    def _join_locked(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        flat = self._chunks[0] if len(self._chunks) == 1 else np.concatenate(self._chunks)
        self._chunks = []
        self._pending = 0
        return flat

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if not np.isfinite(pcm).all():
            log.error("push received non-finite PCM (NaN/Inf); sanitizing to zero")
            pcm = np.nan_to_num(pcm)
        with self._lock:
            self._chunks.append(pcm)
            self._pending += pcm.size
            # Backpressure: cap buffered audio, drop oldest overflow.
            if self._pending > _MAX_BUF_SAMPLES:
                drop = self._pending - _MAX_BUF_SAMPLES
                dropped = 0
                while drop > 0 and self._chunks:
                    c = self._chunks[0]
                    if c.size <= drop:
                        drop -= c.size
                        dropped += c.size
                        self._chunks.pop(0)
                    else:
                        self._chunks[0] = c[drop:]
                        dropped += drop
                        drop = 0
                self._pending -= dropped
                self.consumed += dropped
                log.warning("audio buffer overflow %d samples, dropped oldest (thermal stall?)", dropped)

    def pop_window(self):
        """Return (window, pts_start) once a full window is buffered, else (None, None).

        Prepends the previous window's overlap tail so consecutive windows share
        boundary audio (FR-END-001). pts_start is the PTS of the *new* audio in
        the window (the tail is re-emitted only for smoothing, not re-counted)."""
        with self._lock:
            need = self.window if self._tail.size == 0 else self.hop
            if self._pending < need:
                return None, None
            buf = self._join_locked()
            take = need
            new = buf[:take]
            buf = buf[take:]
            if buf.size:
                self._chunks = [buf]
                self._pending = buf.size
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
            # .copy(): caller may mutate w in-place; a view here would corrupt the
            # next window's prefix tail (silent boundary-smoothing breakage).
            self._tail = w[-self.overlap :].copy() if self.overlap else np.zeros(0, dtype=np.float32)
            self.prefix_samples = prefix
            log.debug("audio window pts=%.3fs len=%d (overlap=%d)", pts, w.size, self._tail.size)
            return w, pts

    def last_window_tail(self) -> np.ndarray:
        """Overlap tail of the most recent window (raw PCM) — prefix handoff to fusion-mlx #914."""
        with self._lock:
            return self._tail.copy()

    def reset(self) -> None:
        """Barge-in (audit 0921 P3): discard all buffered/unconsumed audio.

        ``consumed`` is deliberately NOT rewound — output PTS must stay
        monotonic (LiveKit capture_frame timestamps and _expected_pts both
        assume it); the session's PTS-deviation observer re-syncs from the new
        window. _tail is dropped too: the interrupted utterance's overlap
        would otherwise smear into the next turn's first window."""
        with self._lock:
            self._chunks = []
            self._pending = 0
            self._tail = np.zeros(0, dtype=np.float32)
            self.prefix_samples = 0
            log.info("AudioWindower reset (barge-in); consumed stays monotonic at %d", self.consumed)

    def flush(self) -> bool:
        """Zero-pad the buffer to the next full window (offline end-of-stream).

        Streaming never flushes (wait for more audio); offline must render the
        tail or the last <5s of audio silently produces no frames. Returns
        True if padding was applied."""
        with self._lock:
            need = self.window if self._tail.size == 0 else self.hop
            if self._pending == 0 or self._pending >= need:
                return False
            buf = self._join_locked()
            pad = need - buf.size
            self._chunks = [np.concatenate([buf, np.zeros(pad, dtype=np.float32)])]
            self._pending = need
            log.info("flush: zero-padded %.2fs tail to a full window", pad / self.sr)
            return True

    def step_pts(self) -> float:
        """PTS of the next 33ms step, derived from consumed audio samples."""
        with self._lock:
            return self.consumed / self.sr
