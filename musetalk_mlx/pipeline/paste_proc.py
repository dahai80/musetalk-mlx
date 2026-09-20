import logging
import pickle
import subprocess as sp
import sys

from ..face.mask import _expand_crop_box
from .blending import _paste_alpha

log = logging.getLogger(__name__)


def _send(w, obj) -> None:
    # 4-byte big-endian length + pickled payload over a binary pipe.
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    w.write(len(data).to_bytes(4, "big"))
    w.write(data)
    w.flush()


def _recv(r):
    # None on clean EOF (child gone) — caller treats as proc-down.
    hdr = r.read(4)
    if not hdr:
        return None
    if len(hdr) < 4:
        raise OSError("short header from paste proc")
    n = int.from_bytes(hdr, "big")
    data = r.read(n)
    if len(data) < n:
        raise OSError("short body from paste proc")
    return pickle.loads(data)


def _child_main(r, w):
    # Launched as `python -m ...paste_proc --child` via subprocess — immune
    # to the multiprocessing spawn __main__ re-import trap (python -m pytest
    # and other unguarded -m runners re-run the parent script in spawn
    # children, killing them). Own GIL, no MLX import chain: the parent
    # render thread spends the sync window inside MLX's GIL-holding
    # busy-wait, which starves in-process paste work; this child is immune.
    # Logging goes to stderr only — stdout carries the protocol frames.
    while True:
        try:
            msg = _recv(r)
        except (OSError, EOFError):
            return
        if msg is None:
            log.info("paste proc exiting")
            return
        tag, crop, face, bbox_local, alpha = msg
        if tag != "job":
            continue
        try:
            out = _paste_alpha(crop, face, bbox_local, alpha)
            _send(w, ("ok", out))
        except Exception as e:
            log.warning("paste proc job failed: %s: %s", type(e).__name__, e)
            _send(w, ("err", f"{type(e).__name__}: {e}"))


class PasteProcess:
    # Parent-side handle. paste() ships only the expanded crop region
    # (~1.7MB round trip per frame) — never the 1080p frame on the wire.
    # The 6MB frame copy on the worker thread (~1ms) is far cheaper than
    # the GIL-starved ~20ms/round of paste CPU.
    def __init__(self):
        self._w = None
        self._r = None
        self._proc = None
        self._dead = False

    def start(self):
        self._proc = sp.Popen(
            [sys.executable, "-m", "musetalk_mlx.pipeline.paste_proc", "--child"],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
        )
        self._w = self._proc.stdin
        self._r = self._proc.stdout
        log.info("paste proc started pid=%d", self._proc.pid)

    def paste(self, frame, face, bbox, alpha):
        # Returns a NEW blended frame (input untouched) or None when the
        # child is unavailable — caller falls back to in-process paste.
        if self._dead or self._w is None:
            return None
        x, y, x1, y1 = bbox
        crop_box = _expand_crop_box(bbox, frame.shape)
        x_s, y_s, x_e, y_e = crop_box
        if x_e <= x_s or y_e <= y_s:
            return None
        crop = frame[y_s:y_e, x_s:x_e].copy()
        bbox_local = (int(x - x_s), int(y - y_s), int(x1 - x_s), int(y1 - y_s))
        try:
            _send(self._w, ("job", crop, face, bbox_local, alpha))
            kind, payload = _recv(self._r)
        except (OSError, EOFError, BrokenPipeError, ValueError) as e:
            log.warning("paste proc down (%s); thread-only paste", e)
            self._dead = True
            return None
        if kind != "ok":
            log.warning("paste proc job error (%s); in-process fallback", payload)
            return None
        out = frame.copy()
        out[y_s:y_e, x_s:x_e] = payload
        return out

    def stop(self):
        if self._w is not None:
            try:
                _send(self._w, None)
                self._w.close()
            except (OSError, ValueError):
                pass
        if self._r is not None:
            self._r.close()
        if self._proc is not None:
            try:
                self._proc.wait(timeout=2)
            except sp.TimeoutExpired:
                self._proc.kill()
        self._w = None
        self._r = None
        self._proc = None


if __name__ == "__main__":
    if "--child" in sys.argv:
        _child_main(sys.stdin.buffer, sys.stdout.buffer)
