import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

from ..utils.weights_verify import verify_weights, write_manifest

log = logging.getLogger(__name__)

# Default HF mirror (CLAUDE.md: download models via https://hf-mirror.com).
# MT_WEIGHTS_HF_REPO selects the repo (e.g. "musetalk-mlx/musetalk-v15-mlx").
# MT_WEIGHTS_MIRROR overrides the host for air-gapped / CN mirror deploys.
DEFAULT_MIRROR = os.environ.get("MT_WEIGHTS_MIRROR", "https://hf-mirror.com")
FILES = ("config.json", "unet.safetensors", "vae.safetensors", "whisper_encoder.safetensors")


def _url(repo: str, mirror: str, name: str, rev: str) -> str:
    return f"{mirror}/{repo}/resolve/{rev}/{name}"


def _download(url: str, dst: Path, timeout: float = 300.0) -> None:
    # Stream to a .part file then atomic rename so a partial download never
    # masquerades as a complete weight file (manifest verify would catch it,
    # but a half-written safetensors could crash the loader earlier).
    tmp = dst.with_suffix(dst.suffix + ".part")
    log.info("downloading %s -> %s", url, dst)
    req = urllib.request.Request(url, headers={"User-Agent": "musetalk-mlx-download/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if total:
                pct = done * 100 // total
                if pct % 10 == 0:
                    log.info("  %s: %d%% (%d/%d MB)", dst.name, pct, done >> 20, total >> 20)
    tmp.replace(dst)
    log.info("  done: %s (%d MB)", dst.name, dst.stat().st_size >> 20)


def download(repo: str, out_dir: Path, mirror: str, rev: str = "main") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = out_dir / name
        if dst.exists() and dst.stat().st_size > 0:
            log.info("exists, skip: %s", dst)
            continue
        _download(_url(repo, mirror, name, rev), dst)
    manifest = write_manifest(out_dir)
    ok, reason = verify_weights(out_dir)
    if not ok:
        log.error("post-download verify FAILED: %s", reason)
        sys.exit(2)
    log.info("weights ready at %s (%s)", out_dir, reason)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Download pre-converted MLX weights + verify integrity")
    p.add_argument("--out", required=True, help="output dir (MLX weights root)")
    p.add_argument("--repo", default=os.environ.get("MT_WEIGHTS_HF_REPO", "musetalk-mlx/musetalk-v15-mlx"))
    p.add_argument("--mirror", default=DEFAULT_MIRROR)
    p.add_argument("--rev", default="main")
    a = p.parse_args()
    download(a.repo, Path(a.out), a.mirror, a.rev)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
