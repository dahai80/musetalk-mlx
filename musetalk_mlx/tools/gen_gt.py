import argparse
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Ground-truth generator: runs original MuseTalk PyTorch (MPS backend) on the
# test inputs to produce GT videos for PSNR/SSIM/CSIM regression (PRD §10.5.3).
#
# Implementation (audit 0921 P0-2): subprocess into the MuseTalk checkout and
# drive its scripts.inference via a generated OmegaConf config. MPS is 1-3 FPS
# (one-time per sample); the output mp4 is the GT for the eval CLI.
#
# Requires: MuseTalk checkout with weights (download_weights.sh), torch +
# mmpose/mmdet stack in the MuseTalk venv/python. This tool does NOT run in the
# musetalk-mlx venv — it shells out to MuseTalk's own python (default: python3
# on PATH, override with --musetalk-python).


def _write_config(video: str, audio: str, result_name: str) -> Path:
    lines = [
        "task_0:",
        f'  video_path: "{video}"',
        f'  audio_path: "{audio}"',
        f'  result_name: "{result_name}"',
    ]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return Path(f.name)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate GT videos via MuseTalk PyTorch (MPS) for eval regression"
    )
    p.add_argument("--musetalk-dir", required=True, help="MuseTalk checkout root")
    p.add_argument("--musetalk-python", default="python3", help="python with torch/mmpose/mmdet")
    p.add_argument("--audio", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="output GT mp4 path")
    p.add_argument("--version", default="v1.5", choices=["v1.0", "v1.5"])
    a = p.parse_args()

    mt = Path(a.musetalk_dir).resolve()
    if a.version == "v1.5":
        unet = mt / "models" / "musetalkV15" / "unet.pth"
        cfg = mt / "models" / "musetalkV15" / "musetalk.json"
        varg = "v15"
    else:
        unet = mt / "models" / "musetalk" / "pytorch_model.bin"
        cfg = mt / "models" / "musetalk" / "musetalk.json"
        varg = "v1"
    if not unet.exists():
        log.error("MuseTalk weights not found at %s; run download_weights.sh first", unet)
        return 2

    out_path = Path(a.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_name = out_path.stem
    result_dir = out_path.parent

    conf = _write_config(str(Path(a.video).resolve()), str(Path(a.audio).resolve()), result_name)
    import subprocess

    cmd = [
        a.musetalk_python,
        "-m",
        "scripts.inference",
        "--inference_config",
        str(conf),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        str(unet),
        "--unet_config",
        str(cfg),
        "--version",
        varg,
    ]
    log.info("running MuseTalk GT: cwd=%s cmd=%s", mt, " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=str(mt), check=False)
    except FileNotFoundError:
        log.error("%s not found; point --musetalk-python at the MuseTalk venv python", a.musetalk_python)
        return 3
    if proc.returncode != 0:
        log.error("MuseTalk inference failed (exit %d); GT not generated", proc.returncode)
        return 4

    gt = result_dir / f"{result_name}.mp4"
    if not gt.exists():
        log.error("MuseTalk exited 0 but %s not found; check result_dir naming", gt)
        return 5
    log.info("GT written: %s", gt)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
