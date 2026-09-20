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


def test_paste_back_does_not_mutate_input():
    # Regression: paste_back used to write the blended region into the input
    # frame — session bg-pool frames are reused every loop and accumulated
    # the blended face region run over run.
    frame = _frame()
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    paste_back(frame, face, (100, 100, 200, 200), mask_provider=None)
    assert np.all(frame == 50)


def test_crop_bbox_xyxy_conversion():
    # FaceCropper.crop emits (x, y, w, h); regression: session used to feed the
    # raw xywh into paste_back whose contract is (x1, y1, x2, y2) — x2=150 < x=217
    # read as degenerate and the generated face was silently dropped (raw
    # base-video passthrough, audio never affected output).
    from musetalk_mlx.pipeline.blending import crop_bbox_to_xyxy

    assert crop_bbox_to_xyxy((217, 159, 116, 150)) == (217, 159, 333, 309)


def test_paste_back_with_cropper_style_bbox_changes_mouth():
    from musetalk_mlx.pipeline.blending import crop_bbox_to_xyxy

    frame = _frame()
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    xywh = (100, 100, 150, 150)  # what FaceCropper.crop actually returns
    out = paste_back(frame, face, crop_bbox_to_xyxy(xywh), mask_provider=None)
    assert out[200, 150].mean() > 100  # inside the pasted face
    assert np.array_equal(out[0:80, :], frame[0:80, :])  # outside ROI untouched
