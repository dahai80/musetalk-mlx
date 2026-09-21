import logging
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# PRD §10.5.3 regression thresholds (Normal thermal mode).
THRESHOLDS = {
    "psnr": 38.0,
    "ssim": 0.95,
    "csim": 0.98,
    "lse_c": 5.5,
    "lse_d": 9.0,
    "fps": 30.0,
    "rtt_ms": 80.0,
    "mem_gb": 4.0,
    "max_contiguous_gb": 1.0,
    "leak_mb": 50.0,
}


# --------------------------------------------------------------------------- #
# image quality (torch-free; numpy + scikit-image)
# --------------------------------------------------------------------------- #
def psnr(a, b, data_range=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if data_range is None:
        # Auto-deriving data_range from the sample extrema inflates PSNR for
        # uint8 frames whose actual range is <255 (e.g. 180 -> +3dB), which can
        # cross the 38dB gate falsely. Default to 255 for uint8 (audit P1-19).
        if a.dtype == np.uint8 or b.dtype == np.uint8 or np.max([a.max(), b.max()]) <= 255:
            data_range = 255.0
        else:
            data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 999.0
    return float(10.0 * np.log10((data_range**2) / mse))


def ssim(a, b, data_range=None, channel_axis=-1):
    from skimage.metrics import structural_similarity

    a = np.asarray(a)
    b = np.asarray(b)
    if data_range is None:
        if a.dtype == np.uint8 or b.dtype == np.uint8 or np.max([a.max(), b.max()]) <= 255:
            data_range = 255.0
        else:
            data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    return float(structural_similarity(a, b, data_range=data_range, channel_axis=channel_axis))


def _video_frames(path, max_frames=None):
    # Materializing every frame OOMs on long eval videos. Callers that only need
    # a per-frame metric should pass max_frames; csim path streams via
    # _iter_video_frames instead (audit P1-18).
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def _iter_video_frames(path, max_frames=None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        yield f
        i += 1
        if max_frames and i >= max_frames:
            break
    cap.release()


def video_psnr_ssim(gt_path, pred_path, max_frames=None):
    # Stream pairwise so two long videos don't both sit fully in RAM. Per-frame
    # PSNR/SSIM only needs the current pair (audit P1-18).
    import cv2

    cap_gt = cv2.VideoCapture(str(gt_path))
    cap_pr = cv2.VideoCapture(str(pred_path))
    ps, ss, n = [], [], 0
    try:
        while True:
            rg, fg = cap_gt.read()
            rp, fp = cap_pr.read()
            if not rg or not rp:
                break
            ps.append(psnr(fg, fp, data_range=255.0))
            ss.append(ssim(fg, fp, data_range=255.0))
            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        cap_gt.release()
        cap_pr.release()
    if n == 0:
        return 999.0, 1.0, 0
    return float(np.mean(ps)), float(np.mean(ss)), n


# --------------------------------------------------------------------------- #
# face identity cosine similarity (arcface via insightface; torch env)
# --------------------------------------------------------------------------- #
def _arcface_embed(video_path, app=None, max_frames=None):
    # Returns mean identity embedding over detected faces. Streams frames so a
    # long eval video doesn't OOM before reaching the mean (audit P1-18).
    import cv2

    if app is None:
        app = _default_arcface_app()
    embeds = []
    for f in _iter_video_frames(video_path, max_frames=max_frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)
        if faces:
            embeds.append(faces[0].normed_embedding)
    if not embeds:
        return None
    return np.mean(embeds, axis=0)


def _default_arcface_app():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def csim(gt_path, pred_path, app=None, max_frames=None):
    g = _arcface_embed(gt_path, app, max_frames=max_frames)
    p = _arcface_embed(pred_path, app, max_frames=max_frames)
    if g is None or p is None:
        return 0.0
    cos = float(np.dot(g, p) / (np.linalg.norm(g) * np.linalg.norm(p)))
    return cos


# --------------------------------------------------------------------------- #
# lip-sync (LSE-C / LSE-D) — no GT needed; measures the output's own A/V sync.
# --------------------------------------------------------------------------- #
def lse_d(video_path, model_path, device="cpu"):
    from .syncnet import evaluate_sync

    _, dist, _ = evaluate_sync(video_path, model_path, device=device)
    return dist


def lse_c(video_path, model_path, device="cpu"):
    from .syncnet import evaluate_sync

    _, _, conf = evaluate_sync(video_path, model_path, device=device)
    return conf


def eval_video_sync(video_path, model_path, device="cpu"):
    from .syncnet import evaluate_sync

    _, dist, conf = evaluate_sync(video_path, model_path, device=device)
    return {"lse_d": dist, "lse_c": conf}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
@dataclass
class EvalReport:
    sample: str = ""
    psnr: float | None = None
    ssim: float | None = None
    csim: float | None = None
    lse_c: float | None = None
    lse_d: float | None = None
    frames: int = 0
    gated: list = field(default_factory=list)

    def meets(self, key):
        v = getattr(self, key, None)
        if v is None:
            return None
        thr = THRESHOLDS[key]
        if key in ("lse_d", "rtt_ms", "mem_gb", "leak_mb"):
            return v <= thr
        return v >= thr

    def to_dict(self):
        d = asdict(self)
        d["thresholds"] = THRESHOLDS
        d["meets"] = {k: self.meets(k) for k in THRESHOLDS if getattr(self, k, None) is not None}
        return d
