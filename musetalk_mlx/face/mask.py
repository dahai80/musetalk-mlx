import logging

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

    def mouth_mask(self, frame_bgr: np.ndarray, face_box):
        raise NotImplementedError


class FeatherMask(MaskProvider):
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
    def __init__(self, backend):
        self.backend = backend

    def mouth_mask(self, frame_bgr: np.ndarray, face_box):
        crop_box = _expand_crop_box(face_box, frame_bgr.shape)
        x_s, y_s, x_e, y_e = crop_box
        ph, pw = y_e - y_s, x_e - x_s
        if ph <= 0 or pw <= 0:
            return np.zeros((0, 0), dtype=np.float32), crop_box
        labels, _face_mask = self.backend.parse(frame_bgr[y_s:y_e, x_s:x_e])
        mask = np.isin(labels, _FACE_CLASSES).astype(np.float32)
        mask = _lower_band(mask)
        k = max(1, int(0.1 * ph // 2) * 2 + 1)
        return cv2.GaussianBlur(mask, (k, k), 0), crop_box


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
