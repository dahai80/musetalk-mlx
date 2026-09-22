#!/usr/bin/env python3
"""MuseTalk demo (MLX 版): 复刻 MuseTalk scripts/inference.py 的 yaml 多任务推理,
底层调用 musetalk-mlx (Apple Silicon, 无 PyTorch)。

架构: 渲染跑在独立子进程 (render_worker.py) — musetalk-mlx 的渲染线程在会话
收尾时偶发 C++ 层 "There is no Stream(gpu, 0)" abort, 子进程隔离后主进程的
ffmpeg 音频合成不受影响。

用法:
    ../.venv/bin/python inference.py \
        --inference-config configs/inference.yaml [--only task_0]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference")

ROOT = Path(__file__).resolve().parent
# musetalk-mlx-cases lives inside the musetalk-mlx repo, so the venv + weights
# are one level up. Relative paths keep a public clone runnable out-of-the-box
# after `git clone` + venv setup (no hardcoded absolute paths). Override via env
# for non-standard layouts.
_PARENT = ROOT.parent
MLX_DIR = Path(os.environ.get("MT_MLX_DIR", str(_PARENT / "weights-mlx")))
MLX_PYTHON = Path(os.environ.get("MT_MLX_PYTHON", str(_PARENT / ".venv" / "bin" / "python")))


def get_video_fps(video_path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        num, den = out.split("/")
        return max(1, round(int(num) / int(den)))
    except Exception:
        return 30


def mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> None:
    """将源音频合入输出视频 (imageio 只写视频流)。"""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(out_path),
        ],
        check=True,
    )


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def valid_video(path: Path) -> bool:
    """ffprobe 校验视频完整可读 (moov box 完好)。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return r.returncode == 0 and float(r.stdout.strip()) > 0
    except ValueError:
        return False


def run_task(task_id: str, cfg: dict) -> bool:
    video_path = resolve(cfg["video_path"])
    audio_path = resolve(cfg["audio_path"])
    result_name = cfg.get("result_name", f"{video_path.stem}_{audio_path.stem}.mp4")
    out_path = ROOT / "results" / result_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.mp4")

    log.info("[%s] video=%s audio=%s -> %s", task_id, video_path, audio_path, out_path)

    # 1) 渲染 (子进程隔离; musetalk-mlx 渲染线程偶发 C++ "Stream(gpu, 0)"
    #    abort, 且可能发生在渲染中途 — 失败重试至多 3 次)
    ok = False
    # MT_BATCH=1: 批深度 2 时 60s 长渲染中途 abort 概率高; 单批实测稳定
    env = {**os.environ, "MT_BATCH": "1"}
    for attempt in range(1, 4):
        cmd = [
            str(MLX_PYTHON),
            str(ROOT / "render_worker.py"),
            "--video",
            str(video_path),
            "--audio",
            str(audio_path),
            "--out",
            str(tmp_path),
            "--mlx-dir",
            str(MLX_DIR),
        ]
        if cfg.get("bbox_shift"):
            cmd += ["--bbox-shift", str(cfg["bbox_shift"])]
        ret = subprocess.run(cmd, env=env).returncode
        if tmp_path.exists() and valid_video(tmp_path):
            if ret != 0:
                log.warning("[%s] renderer exited %d but output valid, continuing", task_id, ret)
            ok = True
            break
        log.warning(
            "[%s] render attempt %d failed (exit=%d, valid=%s)",
            task_id,
            attempt,
            ret,
            tmp_path.exists() and valid_video(tmp_path),
        )
        tmp_path.unlink(missing_ok=True)
    if not ok:
        log.error("[%s] render failed after 3 attempts", task_id)
        return False

    # 2) 音频合成 (主进程)
    try:
        mux_audio(tmp_path, audio_path, out_path)
    except subprocess.CalledProcessError as e:
        log.error("[%s] audio mux failed: %s", task_id, e)
        return False
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("[%s] done -> %s", task_id, out_path)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="MuseTalk MLX demo inference")
    p.add_argument("--inference-config", default=str(ROOT / "configs" / "inference.yaml"))
    p.add_argument("--only", default=None, help="只跑指定 task id")
    args = p.parse_args()

    if not MLX_PYTHON.exists():
        log.error("未找到 musetalk-mlx venv: %s", MLX_PYTHON)
        return 1

    with open(args.inference_config) as f:
        config = yaml.safe_load(f)

    ok = True
    for task_id, cfg in config.items():
        if args.only and task_id != args.only:
            continue
        try:
            ok = run_task(task_id, cfg) and ok
        except Exception:
            log.exception("[%s] task failed", task_id)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
