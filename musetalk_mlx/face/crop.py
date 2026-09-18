import logging

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)


class FaceCropper:
    # 5-point affine alignment -> PATCH x PATCH crop (PRD FR-MLX-005).
    # TODO(phase1): similarity transform from 5-point landmarks against the
    # upstream reference template (MuseTalk preprocessing.get_landmark_and_bbox).

    def __init__(self, size: int = config.PATCH):
        self.size = size
        self.affine = None  # 2x3 float32, set once 5-point alignment is wired

    def crop(self, frame_bgr: np.ndarray, landmarks=None):
        """Return (patch, paste_bbox). bbox is (x, y, w, h) in frame coords."""
        h, w = frame_bgr.shape[:2]
        if landmarks is None or self.affine is None:
            # Placeholder: center crop. Keeps the pipeline runnable pre-phase1.
            s = min(h, w)
            x0, y0 = (w - s) // 2, (h - s) // 2
            patch = cv2.resize(
                frame_bgr[y0 : y0 + s, x0 : x0 + s],
                (self.size, self.size),
                interpolation=cv2.INTER_AREA,
            )
            return patch, (x0, y0, s, s)
        patch = cv2.warpAffine(frame_bgr, self.affine, (self.size, self.size))
        return patch, self._bbox
