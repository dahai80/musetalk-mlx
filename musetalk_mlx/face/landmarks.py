import logging

import numpy as np

from .. import config
from .dwpose import NUM_FACE_KPTS, derive_face_bbox, load_dwpose_backend

log = logging.getLogger(__name__)


class _KalmanSmoother:
    # Constant-velocity 1D Kalman per coordinate, vectorized over (68,2).
    # Smooths DWPose jitter and carries a position prediction across a
    # single-frame detection miss (PRD FR-MLX-001 / FR-END-006).

    def __init__(self, n_pts=NUM_FACE_KPTS, process_var=1.0, meas_var=4.0):
        n = n_pts * 2
        self.x = np.zeros((n, 2), dtype=np.float32)  # [pos, vel]
        self.p = np.full((n, 2, 2), 1e3, dtype=np.float32)
        self.q = float(process_var)
        self.r = float(meas_var)
        self._init = False

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if not self._init:
            self.x[:, 0] = z
            self.x[:, 1] = 0.0
            self._init = True
            return self.x[:, 0].copy()
        # predict: x_pos += x_vel; P = F P F^T + Q  (F=[[1,1],[0,1]])
        self.x[:, 0] += self.x[:, 1]
        p00 = self.p[:, 0, 0] + self.p[:, 1, 1] + self.q
        p01 = self.p[:, 0, 1] + self.p[:, 1, 1]
        p11 = self.p[:, 1, 1] + self.q
        # update: H=[1,0], R=r
        y = z - self.x[:, 0]
        s = p00 + self.r
        k0 = p00 / s
        k1 = p01 / s
        self.x[:, 0] += k0 * y
        self.x[:, 1] += k1 * y
        self.p[:, 0, 0] = (1.0 - k0) * p00
        self.p[:, 0, 1] = (1.0 - k0) * p01
        self.p[:, 1, 0] = p01 - k1 * p00
        self.p[:, 1, 1] = p11 - k1 * p01
        return self.x[:, 0].copy()


class LandmarkTracker:
    # 68-pt DWPose face landmarks with miss handling (PRD FR-MLX-001/FR-END-006):
    # single miss -> hold last smoothed pose, N consecutive misses -> idle-blink.
    # A few decoded landmarks can fly to the frame border (e.g. jaw points at
    # x=0 with high SimCC score); they wreck the derived crop bbox, so they are
    # replaced with the face-cluster median before smoothing.

    _OUTLIER_K = 4.0

    def __init__(self, pose_backend=None, upperbondrange: int = 0):
        self.backend = pose_backend if pose_backend is not None else load_dwpose_backend()
        self.upperbondrange = upperbondrange
        self.kf = _KalmanSmoother()
        self.last = None  # smoothed (68,2) float32
        self.fails = 0
        self.idle = False

    def _reject_outliers(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        med = np.median(pts, axis=0)
        d = np.abs(pts - med).max(axis=1)
        med_d = float(np.median(d))
        if med_d <= 1e-3:
            return pts
        bad = d > max(self._OUTLIER_K * med_d, 8.0)
        if bad.any():
            log.debug("rejected %d landmark outliers at %s", int(bad.sum()), np.where(bad)[0][:8])
            pts = pts.copy()
            pts[bad] = med
        return pts

    def detect(self, frame_bgr: np.ndarray):
        if self.backend is None:
            return None
        return self.backend.face_landmarks(frame_bgr)

    def update(self, frame_bgr: np.ndarray):
        pts = self.detect(frame_bgr)
        if pts is not None:
            smooth = self.kf.update(self._reject_outliers(pts))
            cand = smooth.reshape(NUM_FACE_KPTS, 2)
            if self._plausible_bbox(cand, frame_bgr.shape):
                self.fails = 0
                if self.idle:
                    log.info("face re-appearing, resuming lip drive")
                self.idle = False
                self.last = cand
                return self.last
            # Numerically present but geometrically absurd (e.g. a corrupted
            # backend emitting border-pinned coords) — same miss path as None.
            log.warning("implausible face bbox from landmarks; treating frame as miss")
            pts = None
        self.fails += 1
        if self.fails == config.KEYPOINT_FAIL_IDLE and not self.idle:
            self.idle = True
            log.warning("landmarks lost/implausible for %d frames, entering idle-blink state", self.fails)
        if self.idle:
            return None
        log.debug("landmark miss #%d, holding last known pose", self.fails)
        return self.last

    def _plausible_bbox(self, lm: np.ndarray, frame_shape) -> bool:
        bbox = derive_face_bbox(lm, self.upperbondrange)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        fh, fw = frame_shape[:2]
        area = (x2 - x1) * (y2 - y1)
        if area > 0.6 * fw * fh:
            log.debug("bbox area %.0f > 60%% of frame", area)
            return False
        return True
