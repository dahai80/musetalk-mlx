import logging

import cv2
import numpy as np

from ..face.mask import MaskProvider, _expand_crop_box

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


def paste_back(
    frame_bgr: np.ndarray, face: np.ndarray, bbox, mask_provider: MaskProvider | None = None
) -> np.ndarray:
    # Paste the generated 256^2 face patch back onto the base frame (FR-MLX-005).
    # Production path (MuseTalk blending.get_image): expand-crop the face region,
    # paste the generated face at its (x,y) offset within the crop, composite with
    # the face-parse mouth alpha (lower-band, Gaussian-blurred). Feather fallback
    # when no mask provider. bbox is (x1, y1, x2, y2) in frame coords.
    x, y, x1, y1 = bbox
    if x1 <= x or y1 <= y:
        return frame_bgr
    # Batched decode returns MLX arrays; OpenCV path needs host memory.
    face = np.asarray(face)
    if mask_provider is not None:
        return _paste_masked(frame_bgr, face, bbox, mask_provider)
    return _paste_feather(frame_bgr, face, bbox)


def _paste_feather(frame_bgr: np.ndarray, face: np.ndarray, bbox) -> np.ndarray:
    x, y, x1, y1 = bbox
    ph, pw = face.shape[:2]
    fx = x1 - x
    fy = y1 - y
    resized = cv2.resize(face, (fx, fy), interpolation=cv2.INTER_LINEAR) if (fx, fy) != (pw, ph) else face
    x0, y0 = max(x, 0), max(y, 0)
    x1c, y1c = min(x1, frame_bgr.shape[1]), min(y1, frame_bgr.shape[0])
    sub = resized[y0 - y : y1c - y, x0 - x : x1c - x]
    m = _feather_mask(sub.shape[0], sub.shape[1])[..., None]
    roi = frame_bgr[y0:y1c, x0:x1c].astype(np.float32)
    frame_bgr[y0:y1c, x0:x1c] = (sub.astype(np.float32) * m + roi * (1 - m)).astype(np.uint8)
    return frame_bgr


def _paste_masked(frame_bgr: np.ndarray, face: np.ndarray, bbox, mp: MaskProvider) -> np.ndarray:
    x, y, x1, y1 = bbox
    crop_box = _expand_crop_box(bbox, frame_bgr.shape)
    x_s, y_s, x_e, y_e = crop_box
    if x_e <= x_s or y_e <= y_s:
        return _paste_feather(frame_bgr, face, bbox)
    alpha, _ = mp.mouth_mask(frame_bgr, bbox)
    if alpha.size == 0:
        return _paste_feather(frame_bgr, face, bbox)
    # resize generated face to the face-box size, paste into the crop at offset
    fx, fy = x1 - x, y1 - y
    face_resized = cv2.resize(face, (fx, fy), interpolation=cv2.INTER_LINEAR)
    crop = frame_bgr[y_s:y_e, x_s:x_e].copy()
    ox, oy = x - x_s, y - y_s
    # clamp paste into crop bounds
    p_x0, p_y0 = max(0, ox), max(0, oy)
    p_x1, p_y1 = min(crop.shape[1], ox + fx), min(crop.shape[0], oy + fy)
    if p_x1 <= p_x0 or p_y1 <= p_y0:
        return _paste_feather(frame_bgr, face, bbox)
    src = face_resized[p_y0 - oy : p_y1 - oy, p_x0 - ox : p_x1 - ox]
    dst = crop[p_y0:p_y1, p_x0:p_x1]
    a = alpha[p_y0:p_y1, p_x0:p_x1][..., None]
    if a.shape != src.shape:
        a = cv2.resize(a, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)[..., None]
    blended = src.astype(np.float32) * a + dst.astype(np.float32) * (1 - a)
    crop[p_y0:p_y1, p_x0:p_x1] = blended.astype(np.uint8)
    frame_bgr[y_s:y_e, x_s:x_e] = crop
    return frame_bgr


def crop_bbox_to_xyxy(bbox) -> tuple[int, int, int, int]:
    # FaceCropper.crop emits (x, y, w, h); paste_back expects (x1, y1, x2, y2).
    # Session converts at the boundary — this helper keeps the two contracts
    # in one place so a producer change shows up in tests.
    x, y, w, h = bbox
    return (int(x), int(y), int(x) + int(w), int(y) + int(h))
