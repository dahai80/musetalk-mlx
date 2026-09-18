import numpy as np

from musetalk_mlx import config
from musetalk_mlx.face import FaceCropper, LandmarkTracker, derive_face_bbox
from musetalk_mlx.face.landmarks import _KalmanSmoother


def _synth_face():
    # 68 landmarks: x in [100,200], y in [60,160]; lm[29]=(150,80); y_max=160.
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[:, 0] = np.linspace(100, 200, 68)
    lm[:, 1] = np.linspace(60, 160, 68)
    lm[29] = (150, 80)
    return lm


def test_derive_face_bbox():
    lm = _synth_face()
    # half_face_dist = 160-80 = 80; upper_bond = max(0, 80-80) = 0
    bbox = derive_face_bbox(lm)
    assert bbox == (100, 0, 200, 160)


def test_derive_face_bbox_upperbond():
    lm = _synth_face()
    # upperbondrange=+20 -> half_y=100, dist=60, upper_bond=40
    bbox = derive_face_bbox(lm, upperbondrange=20)
    assert bbox == (100, 40, 200, 160)


def test_derive_face_bbox_invalid():
    # all landmarks on a line -> x2-x1>0 but y2-y1==0 -> None
    lm = np.zeros((68, 2), dtype=np.float32)
    lm[:, 0] = np.linspace(100, 200, 68)
    assert derive_face_bbox(lm) is None
    assert derive_face_bbox(None) is None


def test_kalman_converges_to_static():
    kf = _KalmanSmoother()
    lm = _synth_face()
    out = None
    for _ in range(20):
        out = kf.update(lm)
    assert out is not None
    assert np.allclose(out.reshape(68, 2), lm, atol=2.0)


def test_kalman_reduces_jitter():
    rng = np.random.default_rng(0)
    base = _synth_face()
    kf = _KalmanSmoother(meas_var=9.0)
    noisy_var = []
    smoothed_var = []
    for _ in range(40):
        z = base + rng.normal(0, 5.0, size=base.shape)
        out = kf.update(z).reshape(68, 2)
        noisy_var.append(np.var(z - base))
        smoothed_var.append(np.var(out - base))
    assert np.mean(smoothed_var[20:]) < np.mean(noisy_var[20:])


class _NullBackend:
    def face_landmarks(self, frame_bgr):
        return None


class _FixedBackend:
    def __init__(self):
        self.n = 0

    def face_landmarks(self, frame_bgr):
        self.n += 1
        return _synth_face()


def test_tracker_idle_after_n_misses():
    t = LandmarkTracker(pose_backend=_NullBackend())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(config.KEYPOINT_FAIL_IDLE - 1):
        assert t.update(frame) is None  # holding (last is None here)
        assert not t.idle
    t.update(frame)
    assert t.idle


def test_tracker_resumes_after_reappear():
    t = LandmarkTracker(pose_backend=_FixedBackend())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t.update(frame)  # real detection
    assert t.last is not None
    assert not t.idle


def test_cropper_landmark_bbox():
    lm = _synth_face()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropper = FaceCropper()
    patch, bbox = cropper.crop(frame, landmarks=lm)
    assert patch.shape == (config.PATCH, config.PATCH, 3)
    assert bbox == (100, 0, 100, 160)


def test_cropper_center_fallback():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cropper = FaceCropper()
    patch, bbox = cropper.crop(frame, landmarks=None)
    assert patch.shape == (config.PATCH, config.PATCH, 3)
    _, _, w, h = bbox
    assert w == h == 480
