import logging
from collections import deque

import cv2
import imageio
import mlx.core as mx
import numpy as np
from fusion_mlx.video.musetalk_mlx import MuseTalkPipeline
from fusion_mlx.video.musetalk_mlx.whisper.log_mel import log_mel_spectrogram

from .. import config
from ..face.crop import FaceCropper
from ..face.landmarks import LandmarkTracker
from ..face.mask import load_face_parse
from ..pipeline.blending import paste_back
from ..pipeline.lcm import LCMFastSession
from ..utils.audio import AudioWindower
from ..utils.thermal import ladder_for_tier, thermal_tier

log = logging.getLogger(__name__)


class MuseTalkSession:
    # PRD section 7.2 business API (Python/MLX edition).
    # push_audio(16k mono PCM float32 [-1,1]) -> get_output_frame()
    #   -> (BGR uint8 HxWx3, pts seconds) or None when no audio is ready yet.
    # Video PTS inherits the input audio PTS (FR-LK-001); never the system clock.

    def __init__(self, weights_dir, bg_video_path, fps=config.FPS, mlx_dir=None, mask_provider=None):
        self.fps = fps
        self.sr = config.SR
        self.step = self.sr // fps
        self._mlx_dir = mlx_dir
        self._weights_dir = weights_dir
        self.pipe = self._build_pipe(weights_dir, mlx_dir)
        self.lcm = LCMFastSession(self.pipe)
        self._bg = imageio.get_reader(str(bg_video_path))
        self._bg_n = int(self._bg.count_frames())
        self._bg_idx = 0
        self._windower = AudioWindower(sr=self.sr, fps=self.fps)
        self._tracker = LandmarkTracker()
        self._cropper = FaceCropper()
        self._mask = mask_provider if mask_provider is not None else load_face_parse()
        self._pending = deque()  # (chunk (50,384), pts seconds)
        self._last_frame = None
        self._reuse = 0
        log.info(
            "MuseTalkSession ready: bg=%s frames=%d step=%d (%.1fms)",
            bg_video_path,
            self._bg_n,
            self.step,
            self.step / self.sr * 1000,
        )

    @staticmethod
    def _build_pipe(weights_dir, mlx_dir):
        if mlx_dir is not None:
            return MuseTalkPipeline.from_pretrained_mlx(mlx_dir)
        return MuseTalkPipeline.from_pretrained(weights_dir)

    def reload(self, weights_dir=None, mlx_dir=None) -> bool:
        # V2 FR-MLX-006 / FR-END-005: hot-reload weights without dropping audio.
        # Drain pending to a consistent boundary, build the new pipeline, swap
        # atomically. During the swap get_output_frame() emits standby base frames
        # (no black screen). Returns True on success.
        wd = weights_dir or self._weights_dir
        md = mlx_dir if mlx_dir is not None else self._mlx_dir
        log.info("ReloadModel: draining %d pending chunks, rebuilding pipe", len(self._pending))
        self._pending.clear()
        try:
            new_pipe = self._build_pipe(wd, md)
        except Exception as e:
            log.error("ReloadModel failed (%s); keeping old pipe, no black screen", e)
            return False
        self.pipe = new_pipe
        self.lcm = LCMFastSession(new_pipe)
        self._weights_dir = wd
        self._mlx_dir = md
        log.info("ReloadModel done")
        return True

    def push_audio(self, pcm: np.ndarray) -> None:
        self._windower.push(pcm)

    def get_output_frame(self):
        # Render the next 33ms step. Returns (frame_bgr, pts) or None.
        self._encode_windows()
        if not self._pending:
            return None
        chunk, pts = self._pending.popleft()
        frame = self._render(chunk)
        return frame, pts

    def _encode_windows(self) -> None:
        # Drain every fully-buffered overlapping 5s window into per-step chunks.
        # The windower prepends the prior window's overlap tail for boundary
        # smoothing; those prefix chunks were already emitted, so skip them.
        # fusion-mlx #914 will add the embedding-level prefix cache.
        while True:
            w, pts0 = self._windower.pop_window()
            if w is None:
                break
            mel = log_mel_spectrogram(mx.array(w))  # (1,80,3000), 30s pad
            chunks = self.pipe.encode_audio(mel, len(w), fps=self.fps)
            skip = round(self._windower.prefix_samples / self.step) if self._windower.prefix_samples else 0
            n = chunks.shape[0]
            for i in range(skip, n):
                self._pending.append((chunks[i], pts0 + (i - skip) * self.step / self.sr))
            log.debug("encoded window pts=%.3fs -> %d chunks (skipped %d prefix)", pts0, n - skip, skip)

    def _render(self, chunk) -> np.ndarray:
        tier = thermal_tier()
        ladder = ladder_for_tier(tier)
        if ladder["frame_reuse"] > 1:
            self._reuse = (self._reuse + 1) % ladder["frame_reuse"]
            if self._reuse and self._last_frame is not None:
                return self._last_frame
        else:
            self._reuse = 0
        # step-count / ICB need fusion-mlx #911/#912; apply if the pipe exposes it.
        self._apply_ddim_steps(ladder["ddim_steps"])
        frame = self._bg_frame(downscale=ladder["bg_downscale"])
        landmarks = self._tracker.update(frame)
        if landmarks is None or self._tracker.idle:
            return frame
        crop, bbox = self._cropper.crop(frame, landmarks)
        latent = self.pipe.get_latents_for_unet(crop)
        face = self.lcm.generate(latent, chunk)
        if face is None:
            dtype = getattr(self.pipe, "_dtype", mx.float32)
            face = self.pipe.generate_faces(latent, mx.array(chunk[None]).astype(dtype))[0]
        out = paste_back(frame, face, bbox, mask_provider=self._mask)
        self._last_frame = out
        return out

    def _apply_ddim_steps(self, steps: int) -> None:
        setter = getattr(self.pipe, "set_ddim_steps", None)
        if setter is not None:
            setter(steps)
            log.debug("ddim steps set to %d", steps)

    def _bg_frame(self, downscale: int = 1) -> np.ndarray:
        idx = self._bg_idx % self._bg_n if self._bg_n > 0 else self._bg_idx
        try:
            rgb = self._bg.get_data(idx)
        except (IndexError, ValueError):
            self._bg.seek(0)
            rgb = next(self._bg)
        self._bg_idx += 1
        if downscale > 1:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(rgb, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
