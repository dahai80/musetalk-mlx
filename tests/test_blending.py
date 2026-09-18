import numpy as np

from musetalk_mlx.face.mask import FeatherMask, MaskProvider
from musetalk_mlx.pipeline.blending import paste_back


def _frame():
    return np.full((400, 400, 3), 50, dtype=np.uint8)


def test_paste_back_feather_no_seam_outside_roi():
    frame = _frame()
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    bbox = (100, 100, 200, 200)
    out = paste_back(frame, face, bbox, mask_provider=None)
    assert out.shape == frame.shape
    assert np.all(out[0:50, 0:50] == 50)  # untouched region outside ROI


def test_paste_back_feather_inside_roi():
    frame = _frame()
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    bbox = (100, 100, 200, 200)
    out = paste_back(frame, face, bbox, mask_provider=None)
    # center of the pasted face should be brighter than the base
    assert out[150, 150].mean() > 100


def test_paste_back_masked_uses_mask():
    class _FullMask(MaskProvider):
        def mouth_mask(self, frame_bgr, face_box):
            ph = face_box[3] - face_box[1]
            pw = face_box[2] - face_box[0]
            return np.ones((ph, pw), dtype=np.float32), (0, 0, 400, 400)

    frame = _frame()
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    bbox = (100, 100, 200, 200)
    out = paste_back(frame, face, bbox, mask_provider=_FullMask())
    assert out[150, 150].mean() > 150  # mask=1 -> face fully applied


def test_paste_back_degenerate_bbox_unchanged():
    frame = _frame()
    face = np.zeros((256, 256, 3), dtype=np.uint8)
    out = paste_back(frame, face, (200, 200, 100, 100), mask_provider=None)
    assert np.array_equal(out, frame)


def test_feather_mask_shape():
    mp = FeatherMask()
    m, crop = mp.mouth_mask(_frame(), (100, 100, 200, 200))
    assert m.ndim == 2
    assert m.shape[0] > 0 and m.shape[1] > 0
    assert len(crop) == 4
