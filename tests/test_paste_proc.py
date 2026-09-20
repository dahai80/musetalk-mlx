import numpy as np

from musetalk_mlx.face.mask import FeatherMask
from musetalk_mlx.pipeline.blending import paste_back
from musetalk_mlx.pipeline.paste_proc import PasteProcess


def _fixture():
    frame = np.full((400, 400, 3), 50, dtype=np.uint8)
    face = np.full((256, 256, 3), 200, dtype=np.uint8)
    alpha, _ = FeatherMask().mouth_mask(frame, (100, 100, 300, 300))
    return frame, face, (100, 100, 300, 300), alpha


def test_paste_proc_matches_inprocess():
    # End-to-end spawn round trip: child-process blend must equal the
    # in-process paste_back result bit-for-bit (same code path, own GIL).
    frame, face, bbox, alpha = _fixture()
    mp_paste = PasteProcess()
    try:
        mp_paste.start()
        out = mp_paste.paste(frame, face, bbox, alpha)
    finally:
        mp_paste.stop()
    assert out is not None
    ref = paste_back(frame, face, bbox, alpha=alpha)
    np.testing.assert_array_equal(out, ref)


def test_paste_proc_input_untouched():
    frame, face, bbox, alpha = _fixture()
    mp_paste = PasteProcess()
    try:
        mp_paste.start()
        mp_paste.paste(frame, face, bbox, alpha)
    finally:
        mp_paste.stop()
    assert np.all(frame == 50)  # bg pool frames must stay pristine


def test_paste_proc_degenerate_bbox_returns_none():
    frame, face, _alpha, _bbox = _fixture()
    mp_paste = PasteProcess()
    try:
        mp_paste.start()
        assert mp_paste.paste(frame, face, (200, 200, 100, 100), np.zeros((0, 0), dtype=np.float32)) is None
    finally:
        mp_paste.stop()
