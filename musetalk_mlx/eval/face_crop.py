import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Face crop for SyncNet input. The canonical joonson/LatentSync eval pipeline
# (eval/syncnet_detect.py::crop_video) uses S3FD + scene-detect + per-frame face
# tracking + a mouth-biased asymmetric crop (crop_scale=0.4) + medfilt(k=13)
# smoothing. We replicate the *crop geometry* exactly but swap the detector to
# insightface (already a CSIM dep) — bbox parity is close for single-face
# talking heads; the syncnet input distribution is governed by the crop math
# (scale + mouth-bias), not the detector. Documented deviation from joonson.

CROP_SCALE = 0.4
MEDFILT_KERNEL = 13
SYNCNET_FPS = 25  # MFCC 100fps / 4 video frames; canonical eval assumes 25fps


def _detect_per_frame(frames_rgb, app):
    boxes = []
    for f in frames_rgb:
        faces = app.get(f)
        if faces:
            boxes.append(faces[0].bbox)  # (x1,y1,x2,y2)
        else:
            boxes.append(None)
    return boxes


def _interpolate_missing(boxes):
    # Fill None gaps by linear interpolation between nearest valid neighbors so
    # medfilt has a continuous signal (joonson track_face does the same).
    n = len(boxes)
    valid = [(i, b) for i, b in enumerate(boxes) if b is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return [valid[0][1] for _ in range(n)]
    idx = [v[0] for v in valid]
    arr = np.array([v[1] for v in valid], dtype=float)
    out = np.empty((n, 4))
    for j in range(4):
        out[:, j] = np.interp(np.arange(n), idx, arr[:, j])
    return out


def crop_face_video(video_path, out_path, app=None, expand=CROP_SCALE, size=224, target_fps=SYNCNET_FPS):
    # Replicate joonson crop_video geometry: per-frame face center (cx,cy) and
    # half-size bs=max(w,h)/2, medfilt-smoothed, then mouth-biased square crop
    # of side bs*(2+2*cs) resized to 224x224. Resamples to 25fps (canonical).
    import cv2
    from insightface.app import FaceAnalysis
    from scipy import signal

    if app is None:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {video_path}")

    resampled = _resample_frames(frames, src_fps, target_fps)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resampled]
    boxes = _detect_per_frame(rgb, app)
    interp = _interpolate_missing(boxes)
    det = sum(1 for b in boxes if b is not None)
    if interp is None:
        log.warning("no face detected in %s; using full frame", video_path)
        return None

    bs_arr = np.maximum(interp[:, 2] - interp[:, 0], interp[:, 3] - interp[:, 1]) / 2
    cx_arr = (interp[:, 0] + interp[:, 2]) / 2
    cy_arr = (interp[:, 1] + interp[:, 3]) / 2
    k = min(MEDFILT_KERNEL, len(bs_arr) if len(bs_arr) % 2 else len(bs_arr) - 1) or 1
    if k % 2 == 0:
        k += 1
    bs_s = signal.medfilt(bs_arr, kernel_size=k)
    cx_s = signal.medfilt(cx_arr, kernel_size=k)
    cy_s = signal.medfilt(cy_arr, kernel_size=k)

    cs = expand
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, target_fps, (size, size))
    for fidx, image in enumerate(resampled):
        bs = bs_s[fidx]
        bsi = int(bs * (1 + 2 * cs))
        padded = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = cy_s[fidx] + bsi
        mx = cx_s[fidx] + bsi
        y0 = int(my - bs)
        y1 = int(my + bs * (1 + 2 * cs))
        x0 = int(mx - bs * (1 + cs))
        x1 = int(mx + bs * (1 + cs))
        face = padded[max(0, y0) : y1, max(0, x0) : x1]
        if face.size == 0:
            face = image
        writer.write(cv2.resize(face, (size, size)))
    writer.release()
    log.info(
        "face crop %s: %d/%d detected, src_fps=%.1f->%d, mean_bs=%.0f crop_scale=%.1f",
        video_path,
        det,
        len(resampled),
        src_fps,
        target_fps,
        float(bs_s.mean()),
        cs,
    )

    # passthrough audio from source onto the cropped video
    import subprocess

    tmp = str(out_path) + ".tmp.mp4"
    Path(tmp).write_bytes(b"")
    subprocess.run(
        f"ffmpeg -loglevel error -nostdin -y -i {out_path} -i {video_path} "
        f"-c:v copy -c:a aac -map 0:v:0 -map 1:a:0? -shortest {tmp}",
        shell=True,
        check=False,
    )
    if Path(tmp).stat().st_size > 0:
        Path(tmp).replace(out_path)
    else:
        Path(tmp).unlink(missing_ok=True)
    return out_path


def _resample_frames(frames, src_fps, dst_fps):
    # Nearest-index resample so frame count matches audio duration * dst_fps.
    if abs(src_fps - dst_fps) < 0.5 or not frames:
        return frames
    n = max(1, round(len(frames) * dst_fps / src_fps))
    idx = np.clip(np.linspace(0, len(frames) - 1, n), 0, len(frames) - 1).astype(int)
    return [frames[i] for i in idx]
