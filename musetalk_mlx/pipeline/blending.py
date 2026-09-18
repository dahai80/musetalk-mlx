import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MASK_CACHE = {}


def _feather_mask(h: int, w: int, edge: int = 16) -> np.ndarray:
    key = (h, w, edge)
    m = _MASK_CACHE.get(key)
    if m is None:
        edge = max(1, min(edge, h // 2, w // 2))
        vy = np.ones(h, dtype=np.float32)
        vy[:edge] = np.linspace(1.0, 0.0, edge)
        vy[-edge:] = np.linspace(0.0, 1.0, edge)
        vx = np.ones(w, dtype=np.float32)
        vx[:edge] = np.linspace(1.0, 0.0, edge)
        vx[-edge:] = np.linspace(0.0, 1.0, edge)
        m = vy[:, None] * vx[None, :]
        _MASK_CACHE[key] = m
    return m


def paste_back(frame_bgr: np.ndarray, patch: np.ndarray, bbox) -> np.ndarray:
    # Paste the generated face patch back onto the base frame with a feathered
    # alpha edge to hide the 256x256 seam.
    # TODO(phase2): inverse warpAffine paste + upstream jaw-parse mask
    # (musetalk/utils/blending.get_image) for the production blend.
    x, y = bbox[0], bbox[1]
    ph, pw = patch.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + pw, frame_bgr.shape[1]), min(y + ph, frame_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return frame_bgr
    sub = patch[x0 - x : x1 - x, y0 - y : y1 - y]
    sub = cv2.resize(sub, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    m = _feather_mask(sub.shape[0], sub.shape[1])[..., None]
    roi = frame_bgr[y0:y1, x0:x1].astype(np.float32)
    frame_bgr[y0:y1, x0:x1] = (sub.astype(np.float32) * m + roi * (1 - m)).astype(np.uint8)
    return frame_bgr
