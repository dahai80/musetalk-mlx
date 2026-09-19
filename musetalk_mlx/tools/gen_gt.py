import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# CUDA ground-truth generator (GATED). Runs original MuseTalk PyTorch on the
# test inputs to produce GT videos for PSNR/SSIM/CSIM regression (PRD §10.5.3).
#
# Requires: MuseTalk weights (download_weights.sh), torch + mmpose/mmdet stack,
# MPS backend (1-3 FPS — slow, one-time per sample). Not wired into CI until
# the torch env + weights are provisioned. See tests/eval/EVAL_README.md.


def main() -> int:
    p = argparse.ArgumentParser(description="Generate CUDA GT videos via MuseTalk PyTorch (GATED)")
    p.add_argument("--musetalk-dir", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="v1.5")
    a = p.parse_args()

    weights = Path(a.musetalk_dir) / "models"
    if not (weights / "musetalkV15" / "unet.pth").exists():
        log.error("MuseTalk weights not found at %s; run download_weights.sh first", weights)
        return 2
    try:
        import mmdet  # noqa: F401
        import mmpose  # noqa: F401
    except ImportError:
        log.error("mmpose/mmdet not installed; GT generation env not provisioned (see EVAL_README.md)")
        return 3

    log.error("GT generation not yet implemented — gated pending torch env provisioning")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
