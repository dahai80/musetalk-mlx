import logging

import numpy as np

log = logging.getLogger(__name__)

# COCO-WholeBody face keypoint subset: indices 23..90 of the 133-pt body give
# 68 face landmarks. Same convention as MuseTalk
# musetalk/utils/preprocessing.py::get_landmark_and_bbox (keypoints[0][23:91]).
FACE_KPT_SLICE = slice(23, 91)
NUM_FACE_KPTS = 68
COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def derive_face_bbox(face_lm: np.ndarray, upperbondrange: int = 0):
    # Derive the lip-sync crop bbox from 68 DWPose face landmarks.
    # Ported from MuseTalk preprocessing.get_landmark_and_bbox: axis-aligned
    # bbox; x from landmark x-extent, y from a half-face line (lm[29]) minus
    # half_face_dist, shifted by upperbondrange. Returns (x1,y1,x2,y2) int32
    # or None when the derived bbox is degenerate.
    if face_lm is None or len(face_lm) <= 29:
        return None
    lm = np.asarray(face_lm, dtype=np.int32)
    half_y = int(lm[29, 1])
    if upperbondrange != 0:
        half_y = upperbondrange + half_y
    y_max = int(np.max(lm[:, 1]))
    half_face_dist = y_max - half_y
    upper_bond = max(0, half_y - half_face_dist)
    x1 = int(np.min(lm[:, 0]))
    x2 = int(np.max(lm[:, 0]))
    y1 = upper_bond
    y2 = y_max
    if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
        log.debug("degenerate landmark bbox x=%d-%d y=%d-%d; fallback", x1, x2, y1, y2)
        return None
    return (x1, y1, x2, y2)


def load_dwpose_backend():
    # Load the DWPose MLX backend from fusion-mlx if available.
    # fusion-mlx does not yet ship DWPose/RTMPose (tracked as an issue on
    # dahai80/fusion-mlx). Returns None until then -> LandmarkTracker falls
    # back to idle-blink, keeping the pipeline runnable.
    try:
        from fusion_mlx.video.dwpose import DWPose
    except ImportError:
        log.info(
            "fusion_mlx.video.dwpose unavailable; DWPose backend offline "
            "(see fusion-mlx DWPose issue). LandmarkTracker -> idle-blink fallback."
        )
        return None
    # API shape is settled by the fusion-mlx DWPose issue; instantiate lazily.
    try:
        return DWPose.from_pretrained()
    except Exception as e:  # API drift guard
        log.warning("DWPose backend init failed (%s); falling back to idle-blink", e)
        return None
