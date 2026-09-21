import logging

import cv2
import numpy as np

from .. import config
from .dwpose import derive_face_bbox

log = logging.getLogger(__name__)


class FaceCropper:
    # Axis-aligned face crop -> PATCH x PATCH (PRD FR-MLX-005).
    # Bbox derived from 68 DWPose landmarks (MuseTalk preprocessing convention);
    # center-crop fallback keeps the pipeline runnable pre-DWPose / on a miss.

    MIN_CROP = 32  # below this the upscaled patch is severely undersampled

    def __init__(self, size: int = config.PATCH, upperbondrange: int = 0):
        self.size = size
        self.upperbondrange = upperbondrange

    def _center_crop(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty frame passed to FaceCropper")
        h, w = frame_bgr.shape[:2]
        s = min(h, w)
        x0, y0 = (w - s) // 2, (h - s) // 2
        patch = cv2.resize(
            frame_bgr[y0 : y0 + s, x0 : x0 + s],
            (self.size, self.size),
            interpolation=cv2.INTER_AREA,
        )
        return patch, (x0, y0, s, s)

    def crop(self, frame_bgr: np.ndarray, landmarks=None):
        # Returns (patch, paste_bbox). paste_bbox is (x, y, w, h) in frame coords.
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("empty frame passed to FaceCropper")
        if landmarks is None:
            return self._center_crop(frame_bgr)
        bbox = derive_face_bbox(landmarks, self.upperbondrange)
        if bbox is None:
            return self._center_crop(frame_bgr)
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]
        x0, y0 = max(0, x1), max(0, y1)
        x1c, y1c = min(w, x2), min(h, y2)
        if x1c <= x0 or y1c <= y0:
            log.debug("clamped bbox empty (%d,%d,%d,%d); center-crop", x0, y0, x1c, y1c)
            return self._center_crop(frame_bgr)
        cw, ch = x1c - x0, y1c - y0
        # A tiny crop (distant face) upscaled to PATCH produces heavy interpolation
        # artifacts and crashes UNet input quality — fall back to center crop and
        # log so the degradation is observable (audit P3-7).
        if cw < self.MIN_CROP or ch < self.MIN_CROP:
            log.warning("face crop too small %dx%d (<%d); center-crop fallback", cw, ch, self.MIN_CROP)
            return self._center_crop(frame_bgr)
        crop = frame_bgr[y0:y1c, x0:x1c]
        patch = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return patch, (x0, y0, cw, ch)
