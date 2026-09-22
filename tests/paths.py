import os
from pathlib import Path

# Central test-asset resolution. MT_MLX_DIST / MT_TEST_VIDEO / MT_TEST_AUDIO
# override defaults; weight-gated tests skip (importorskip) when assets absent.

REPO = Path(__file__).resolve().parent.parent

DIST = Path(os.environ.get("MT_MLX_DIST", REPO / "weights-mlx"))
# Test video/audio default to the MuseTalk demo assets; set MT_MUSETALK to the
# MuseTalk clone root (default: sibling of this repo's parent workspace).
_MUSETALK = Path(os.environ.get("MT_MUSETALK", REPO.parent / "MuseTalk"))
TEST_VIDEO = Path(os.environ.get("MT_TEST_VIDEO", _MUSETALK / "data/video/sun.mp4"))
TEST_AUDIO = Path(os.environ.get("MT_TEST_AUDIO", _MUSETALK / "data/audio/eng.wav"))
