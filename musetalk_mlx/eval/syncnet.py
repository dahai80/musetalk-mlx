import logging
import math
import os
import subprocess
import tempfile

import numpy as np

log = logging.getLogger(__name__)

# Faithful port of the joonson/LatentSync SyncNet (eval/syncnet/syncnet.py +
# syncnet_eval.py). Defines the canonical LSE-C / LSE-D lip-sync metrics cited
# in PRD §10.5.3 (MuseTalk 6.53 / Wav2Lip 7.42 / LatentSync 7.90).
#
# Weights: joonson syncnet_v2.model, hosted at ByteDance/LatentSync-1.5
# (path auxiliary/syncnet_v2.model). Download via hf-mirror:
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
#       ByteDance/LatentSync-1.5 auxiliary/syncnet_v2.model --local-dir weights/eval
#
# torch is imported lazily — this module is part of the torch-free musetalk-mlx
# package; eval runs in a separate torch env (openclaw conda).

SYNCNET_INPUT_SIZE = 224  # HARD CODED in upstream eval
SYNCNET_VSHIFT = 15
SYNCNET_AUDIO_WIN = 20  # MFCC frames per window (5 video frames * 4)
SYNCNET_VIDEO_WIN = 5  # consecutive frames per window


def _torch():
    import torch

    return torch


class SyncNetS:
    # joonson syncnet_v2 architecture (eval/syncnet/syncnet.py class S).
    # Reproduced verbatim; loads the upstream .model checkpoint directly.

    def __init__(self, num_layers_in_fc_layers=1024):
        torch = _torch()
        nn = torch.nn
        self.torch = torch

        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        return self.netfcaud(mid)

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        return self.netfclip(mid)

    def load_state(self, model_path):
        torch = self.torch
        # weights_only=True: syncnet_v2.model is a third-party pickle —
        # weights_only=False executes arbitrary __reduce__ payloads (RCE)
        # (audit P0-7). Fall back to False only if the checkpoint uses legacy
        # pickled modules AND the operator has explicitly opted in via env.
        import os

        weights_only = os.environ.get("MT_SYNCNET_UNSAFE_LOAD", "0") != "1"
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=weights_only)
        except Exception as e:
            if not weights_only:
                raise
            log.warning(
                "safe torch.load failed (%s); set MT_SYNCNET_UNSAFE_LOAD=1 only on a "
                "trusted checkpoint to allow legacy unpickle",
                e,
            )
            raise
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # joonson .model saves the whole module; keys may be prefixed __S__.
        clean = {}
        for k, v in state.items():
            nk = k.replace("__S__.", "") if k.startswith("__S__") else k
            clean[nk] = v
        own = dict(self._named_params())
        loaded = 0
        for k, v in clean.items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v)
                loaded += 1
        # A near-zero partial load produces metrics on a mostly-random network
        # — fail loudly instead of logging info and proceeding (audit fix).
        if loaded < max(1, int(0.5 * len(own))):
            raise RuntimeError(
                f"syncnet: only {loaded}/{len(own)} params loaded from {model_path} "
                f"(<50%); checkpoint may be incompatible"
            )
        log.info("syncnet loaded %d/%d params from %s", loaded, len(own), model_path)
        return loaded, len(own)

    def _named_params(self):
        # flatten netcnnaud/netfcaud/netfclip/netcnnlip into a named dict
        out = {}
        for prefix, mod in [
            ("netcnnaud", self.netcnnaud),
            ("netfcaud", self.netfcaud),
            ("netfclip", self.netfclip),
            ("netcnnlip", self.netcnnlip),
        ]:
            for k, v in mod.state_dict().items():
                out[f"{prefix}.{k}"] = v
        return out

    def eval(self):
        for mod in [self.netcnnaud, self.netfcaud, self.netfclip, self.netcnnlip]:
            mod.eval()


def _calc_pdist(feat1, feat2, vshift):
    torch = _torch()
    F = torch.nn.functional
    win = vshift * 2 + 1
    feat2p = F.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(F.pairwise_distance(feat1[[i], :].repeat(win, 1), feat2p[i : i + win, :]))
    return dists


def _extract_mfcc(audio_path, sr):
    import python_speech_features as psf
    from scipy.io import wavfile

    _, audio = wavfile.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    # psf.mfcc expects float in [-1,1]; int16 magnitudes (~32767) produce
    # scale-dependent mel coeffs that only match upstream's exact (buggy) path.
    # Normalize — keeps parity with any librosa-based path elsewhere (audit fix).
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / 32768.0
    mfcc = list(zip(*psf.mfcc(audio, sr)))
    mfcc = np.stack([np.array(i) for i in mfcc])
    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    return cc


def _load_video_frames(video_path, size=SYNCNET_INPUT_SIZE):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    images = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        images.append(cv2.resize(img, (size, size)))
    cap.release()
    if not images:
        return None, fps
    im = np.stack(images, axis=3)
    im = np.expand_dims(im, axis=0)
    im = np.transpose(im, (0, 3, 4, 1, 2))
    return im, fps


def evaluate_sync(video_path, model_path, device="cpu", vshift=SYNCNET_VSHIFT, batch_size=20):
    # Returns (av_offset, lse_d, lse_c). lse_d = min mean dist over offsets;
    # lse_c = median(mean_dists) - min_dist (confidence, higher = better).
    torch = _torch()
    net = SyncNetS()
    loaded, total = net.load_state(model_path)
    if loaded == 0:
        raise RuntimeError(f"syncnet: 0 params loaded from {model_path} (total {total})")
    net.eval()
    for mod in [net.netcnnaud, net.netfcaud, net.netfclip, net.netcnnlip]:
        mod.to(device)

    with tempfile.TemporaryDirectory() as tmp:
        img_jpg = os.path.join(tmp, "%06d.jpg")
        # List-form subprocess: shell=True with unquoted user paths is a command
        # injection / breakage vector (audit P0-8).
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-f",
                "image2",
                img_jpg,
            ],
            check=True,
        )
        wav = os.path.join(tmp, "audio.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-async",
                "1",
                "-ac",
                "1",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                wav,
            ],
            check=True,
        )
        im, _fps = _load_video_frames(video_path)
        cc = _extract_mfcc(wav, 16000)

    if im is None:
        raise RuntimeError("no frames decoded from video")

    imtv = torch.from_numpy(im.astype(float)).float()
    cct = torch.from_numpy(cc.astype(float)).float()

    min_length = min(imtv.shape[2], math.floor(cct.shape[3] / 4))
    lastframe = min_length - SYNCNET_VIDEO_WIN
    if lastframe <= 0:
        raise RuntimeError(f"video too short: {imtv.shape[2]} frames, need > {SYNCNET_VIDEO_WIN}")

    im_feat, cc_feat = [], []
    # no_grad: eval builds a graph per forward otherwise — leaks CPU/GPU memory
    # and slows each batch over a long video (audit fix).
    with torch.no_grad():
        for i in range(0, lastframe, batch_size):
            upper = min(lastframe, i + batch_size)
            im_batch = [imtv[:, :, v : v + SYNCNET_VIDEO_WIN, :, :] for v in range(i, upper)]
            im_in = torch.cat(im_batch, 0).to(device)
            im_out = net.forward_lip(im_in)
            im_feat.append(im_out.data.cpu())
            cc_batch = [cct[:, :, :, v * 4 : v * 4 + SYNCNET_AUDIO_WIN] for v in range(i, upper)]
            cc_in = torch.cat(cc_batch, 0).to(device)
            cc_out = net.forward_aud(cc_in)
            cc_feat.append(cc_out.data.cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)

    dists = _calc_pdist(im_feat, cc_feat, vshift)
    mean_dists = torch.mean(torch.stack(dists, 1), 1)
    min_dist, minidx = torch.min(mean_dists, 0)
    conf = torch.median(mean_dists) - min_dist

    av_offset = vshift - int(minidx)
    log.info(
        "syncnet %s: LSE-D=%.4f LSE-C=%.4f offset=%d",
        video_path,
        min_dist.item(),
        conf.item(),
        av_offset,
    )
    return av_offset, float(min_dist.item()), float(conf.item())
