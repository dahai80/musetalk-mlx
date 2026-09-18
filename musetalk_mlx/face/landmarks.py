import logging

import numpy as np

from .. import config

log = logging.getLogger(__name__)


class LandmarkTracker:
    # 5-point face landmarks with miss handling (PRD FR-MLX-001 / FR-END-006):
    # single miss -> hold last known pose, N consecutive misses -> idle-blink.
    # TODO(phase1): DWPose/RTMPose 5-point inference + Kalman smoothing over
    # KALMAN_HISTORY frames for sub-frame-miss prediction.

    def __init__(self):
        self.last = None
        self.fails = 0
        self.idle = False

    def detect(self, frame_bgr: np.ndarray):
        # TODO(phase1): wire DWPose/RTMPose backend (weights via fusion-mlx).
        return None

    def update(self, frame_bgr: np.ndarray):
        pts = self.detect(frame_bgr)
        if pts is None:
            self.fails += 1
            if self.fails == config.KEYPOINT_FAIL_IDLE and not self.idle:
                self.idle = True
                log.warning("landmarks lost for %d frames, entering idle-blink state", self.fails)
            if self.idle:
                return None
            log.debug("landmark miss #%d, holding last known pose", self.fails)
            return self.last
        self.fails = 0
        if self.idle:
            log.info("face re-appearing, resuming lip drive")
        self.idle = False
        self.last = pts
        return pts
