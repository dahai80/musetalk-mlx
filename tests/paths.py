import os
from pathlib import Path

# Central test-asset resolution (Phase 2/3/4 wrap-up). /tmp/mtlk_mlx_dist was a
# hand-built temp dir that no longer exists — every weight-gated test skipped.
# MT_MLX_DIST overrides; default = repo weights-mlx/.

REPO = Path(__file__).resolve().parent.parent

DIST = Path(os.environ.get("MT_MLX_DIST", REPO / "weights-mlx"))
TEST_VIDEO = Path(os.environ.get("MT_TEST_VIDEO", "/Users/dahai/migration/MuseTalk/data/video/sun.mp4"))
TEST_AUDIO = Path(os.environ.get("MT_TEST_AUDIO", "/Users/dahai/migration/MuseTalk/data/audio/eng.wav"))
