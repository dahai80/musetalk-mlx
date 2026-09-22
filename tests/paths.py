import json
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


def weights_ok(dist: Path | None = None) -> bool:
    # Real weight gate (audit A1): DIST.exists() is too coarse — weights-mlx/
    # ships config.json + manifest.json + manifest.sig in git (tracked), so the
    # dir exists on every checkout including the CI runner, but the 3.5GB of
    # *.safetensors is gitignored and absent. That made weight-gated tests ERROR
    # on from_pretrained instead of skip (5 consecutive CI FAILs). Gate on the
    # actual weight files: manifest-declared safetensors present and non-empty.
    d = dist or DIST
    manifest = d / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        declared = json.loads(manifest.read_text()).get("files", {})
    except Exception:
        return False
    needed = {n for n in declared if n.endswith(".safetensors")}
    if not needed:
        needed = {"unet.safetensors", "vae.safetensors", "whisper_encoder.safetensors"}
    return all((d / n).is_file() and (d / n).stat().st_size > 0 for n in needed)
