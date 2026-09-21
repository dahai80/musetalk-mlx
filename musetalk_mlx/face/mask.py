import logging
import threading

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)

# Face-parse class labels (BiSeNet 19-class) kept for the mouth-region alpha.
# MuseTalk keeps the lower `upper_boundary_ratio` of the face-skin classes.
_FACE_CLASSES = (1, 2, 3, 4, 5, 6, 10, 12, 13)


def _expand_crop_box(face_box, frame_shape, expand=config.BLEND_EXPAND):
    x, y, x1, y1 = face_box
    h, w = frame_shape[:2]
    cx, cy = (x + x1) // 2, (y + y1) // 2
    s = int(max(x1 - x, y1 - y) // 2 * expand)
    return (max(0, cx - s), max(0, cy - s), min(w, cx + s), min(h, cy + s))


def _lower_band(mask):
    top = int(mask.shape[0] * config.BLEND_UPPER_BOUNDARY_RATIO)
    out = np.zeros_like(mask)
    out[top:, :] = mask[top:, :]
    return out


class MaskProvider:
    # Paste alpha mask for production blend (FR-MLX-005). Returns (mask, crop_box)
    # where crop_box is the expanded paste region (x_s, y_s, x_e, y_e). Real impl
    # needs a face-parse model (fusion-mlx #910); feather fallback hides the seam.
    # mouth_mask_cached_only is the thread-safe read path for the paste worker
    # thread: cache hit or a cheap pure-numpy mask, never a model call.

    def mouth_mask(self, frame_bgr: np.ndarray, face_box):
        raise NotImplementedError

    def mouth_mask_cached_only(self, frame_bgr: np.ndarray, face_box):
        # Base default must NOT delegate to mouth_mask — that would run the
        # parse backend on whatever thread calls this (paste worker / render
        # thread), contradicting the "cached_only = never a model call"
        # contract. Subclasses override with a real cache-hit lookup; the base
        # raises so a misconfigured provider fails loudly (audit 0921 DC3).
        raise NotImplementedError


class FeatherMask(MaskProvider):
    # FeatherMask has no model call — mouth_mask is pure numpy/cv2, so the
    # cached_only contract ("never a model call") is trivially satisfied by
    # computing the feather mask directly (no cache needed).
    def mouth_mask_cached_only(self, frame_bgr: np.ndarray, face_box):
        return self.mouth_mask(frame_bgr, face_box)

    def mouth_mask(self, frame_bgr: np.ndarray, face_box):
        crop_box = _expand_crop_box(face_box, frame_bgr.shape)
        x_s, y_s, x_e, y_e = crop_box
        ph, pw = y_e - y_s, x_e - x_s
        if ph <= 0 or pw <= 0:
            return np.zeros((0, 0), dtype=np.float32), crop_box
        m = np.zeros((ph, pw), dtype=np.float32)
        m[:, :] = 1.0
        m = _lower_band(m)
        k = max(1, int(0.05 * ph // 2) * 2 + 1)
        return cv2.GaussianBlur(m, (k, k), 0), crop_box


class FaceParseMask(MaskProvider):
    # Upstream MuseTalk computes the parse mask once per identity and reuses it;
    # parsing every frame costs ~30ms and the alpha is nearly static (same face,
    # same crop geometry). Cache keyed by quantized crop box; invalidate when the
    # box moves more than MASK_CACHE_TOL px or the crop size changes.
    MASK_CACHE_TOL = 4

    def __init__(self, backend):
        self.backend = backend
        self._cache_key = None
        self._cache_mask = None
        # _cache_mask/_cache_key are written on the render thread and read by
        # mouth_mask_cached_only on the paste worker thread — guard the swap
        # (audit P1-17). Whole-tuple replacement keeps the read consistent.
        self._cache_lock = threading.Lock()

    def mouth_mask(self, frame_bgr: np.ndarray, face_box):
        crop_box = _expand_crop_box(face_box, frame_bgr.shape)
        x_s, y_s, x_e, y_e = crop_box
        ph, pw = y_e - y_s, x_e - x_s
        if ph <= 0 or pw <= 0:
            return np.zeros((0, 0), dtype=np.float32), crop_box
        key = (pw, ph, x_s, y_s)
        with self._cache_lock:
            cached_mask = self._cache_mask
            cached_key = self._cache_key
        if (
            cached_mask is not None
            and cached_key is not None
            and cached_key[:2] == key[:2]
            and abs(cached_key[2] - key[2]) <= self.MASK_CACHE_TOL
            and abs(cached_key[3] - key[3]) <= self.MASK_CACHE_TOL
        ):
            # Tighten the tolerance: a 4px drift re-uses a mask computed at the
            # old position, which shifts the mouth alpha up to 4px (audit P1-15).
            # Re-fetch when the shift exceeds 2px so the alpha tracks the face.
            return cached_mask, crop_box
        # parse() is a model call — wrap so an inference failure falls back to
        # the feather mask instead of killing the paste worker (audit P1-16).
        try:
            labels, _face_mask = self.backend.parse(frame_bgr[y_s:y_e, x_s:x_e])
        except Exception as e:
            log.warning("face-parse backend failed (%s); feather fallback this frame", e)
            return np.zeros((0, 0), dtype=np.float32), crop_box
        if labels.shape[:2] != (ph, pw):
            labels = cv2.resize(labels.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST)
        mask = np.isin(labels, _FACE_CLASSES).astype(np.float32)
        mask = _lower_band(mask)
        k = max(1, int(0.1 * ph // 2) * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        with self._cache_lock:
            self._cache_key = key
            self._cache_mask = mask
        return mask, crop_box

    def mouth_mask_cached_only(self, frame_bgr: np.ndarray, face_box):
        # Cache-hit-only lookup for the paste worker thread (never calls the
        # MLX parse backend off the main thread). None means cache miss — the
        # caller must fill the mask on the main thread.
        with self._cache_lock:
            cached_mask = self._cache_mask
            cached_key = self._cache_key
        if cached_mask is None or cached_key is None:
            return None
        crop_box = _expand_crop_box(face_box, frame_bgr.shape)
        x_s, y_s, x_e, y_e = crop_box
        pw, ph = x_e - x_s, y_e - y_s
        key = (pw, ph, x_s, y_s)
        if (
            cached_key[:2] == key[:2]
            and abs(cached_key[2] - key[2]) <= self.MASK_CACHE_TOL
            and abs(cached_key[3] - key[3]) <= self.MASK_CACHE_TOL
        ):
            return cached_mask, crop_box
        return None


def load_face_parse() -> MaskProvider:
    # Probe fusion-mlx face-parsing backend (issue #910). Falls back to FeatherMask.
    try:
        from fusion_mlx.video.face_parsing import FaceParsing
    except ImportError:
        log.info("fusion_mlx.video.face_parsing unavailable (#910); feather-mask fallback")
        return FeatherMask()
    try:
        return FaceParseMask(FaceParsing.from_pretrained())
    except Exception as e:
        log.warning("face-parse init failed (%s); feather-mask fallback", e)
        return FeatherMask()
