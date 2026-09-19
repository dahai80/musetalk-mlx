import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from musetalk_mlx.eval.face_crop import crop_face_video
from musetalk_mlx.eval.metrics import (
    THRESHOLDS,
    EvalReport,
    csim,
    eval_video_sync,
    video_psnr_ssim,
)

log = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(
        description="End-to-end quality regression (PRD §10.5): PSNR/SSIM/CSIM vs GT, LSE-C/LSE-D on output."
    )
    p.add_argument("--pred", required=True, help="musetalk-mlx output video (or dir of .mp4)")
    p.add_argument("--gt", default=None, help="CUDA GT video/dir. Omit to gate PSNR/SSIM/CSIM.")
    p.add_argument(
        "--audio",
        default=None,
        help="source audio (16kHz wav) to mux into pred if it has no audio track",
    )
    p.add_argument("--syncnet", default="weights/eval/auxiliary/syncnet_v2.model")
    p.add_argument("--device", default="cpu")
    p.add_argument("--report", default="results/eval_report.json")
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args()

    preds = _expand(a.pred)
    gts = _expand(a.gt) if a.gt else []
    audios = _expand(a.audio) if a.audio else []
    reports = []
    app = None
    for i, pred in enumerate(preds):
        gt = gts[i] if i < len(gts) else None
        pred_audio = _ensure_audio(pred, audios[i] if i < len(audios) else None)
        r = EvalReport(sample=Path(pred).stem)
        if gt:
            ps, ss, n = video_psnr_ssim(gt, pred, max_frames=a.max_frames)
            r.psnr = ps
            r.ssim = ss
            r.frames = n
            try:
                from insightface.app import FaceAnalysis

                if app is None:
                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                r.csim = csim(gt, pred, app=app)
            except Exception as e:
                log.warning("CSIM skipped (%s)", e)
                r.gated.append("csim")
        else:
            r.gated.extend(["psnr", "ssim", "csim"])

        if Path(a.syncnet).exists():
            try:
                from insightface.app import FaceAnalysis

                if app is None:
                    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                    app.prepare(ctx_id=-1, det_size=(640, 640))
                cropped = Path(pred).with_suffix(".facecrop.mp4")
                if crop_face_video(pred_audio, cropped, app=app):
                    sync = eval_video_sync(str(cropped), a.syncnet, device=a.device)
                else:
                    sync = eval_video_sync(pred_audio, a.syncnet, device=a.device)
                r.lse_c = sync["lse_c"]
                r.lse_d = sync["lse_d"]
            except Exception as e:
                log.warning("LSE skipped (%s)", e)
                r.gated.extend(["lse_c", "lse_d"])
        else:
            log.info("syncnet weights missing (%s); LSE gated", a.syncnet)
            r.gated.extend(["lse_c", "lse_d"])

        reports.append(r.to_dict())
        print(json.dumps(r.to_dict(), indent=2))

    out = Path(a.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples": reports,
        "thresholds": THRESHOLDS,
        "gated_metrics": sorted({g for r in reports for g in r.get("gated", [])}),
    }
    out.write_text(json.dumps(summary, indent=2))
    log.info("eval report: %s", out)
    return 0


def _expand(path):
    p = Path(path)
    if p.is_dir():
        return sorted(str(x) for x in p.glob("*.mp4"))
    return [str(p)]


def _ensure_audio(pred, audio):
    """SyncNet needs audio track. If pred lacks one, mux source audio via ffmpeg."""
    if audio is None:
        return pred
    has_audio = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                pred,
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        != ""
    )
    if has_audio:
        return pred
    muxed = str(Path(pred).with_suffix(".muxed.mp4"))
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        pred,
        "-i",
        audio,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        muxed,
    ]
    log.info("muxing audio %s into %s -> %s", audio, pred, muxed)
    subprocess.run(cmd, check=True)
    return muxed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
