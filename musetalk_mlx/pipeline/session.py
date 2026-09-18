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
from ..pipeline.blending import paste_back
from ..utils.audio import AudioWindower
from ..utils.thermal import thermal_tier

log = logging.getLogger(__name__)


class MuseTalkSession:
    # PRD section 7.2 business API (Python/MLX edition).
    # push_audio(16k mono PCM float32 [-1,1]) -> get_output_frame()
    #   -> (BGR uint8 HxWx3, pts seconds) or None when no audio is ready yet.
    # Video PTS inherits the input audio PTS (FR-LK-001); never the system clock.

    def __init__(self, weights_dir, bg_video_path, fps=config.FPS, mlx_dir=None):
        self.fps = fps
        self.sr = config.SR
        self.step = self.sr // fps
        if mlx_dir is not None:
            self.pipe = MuseTalkPipeline.from_pretrained_mlx(mlx_dir)
        else:
            self.pipe = MuseTalkPipeline.from_pretrained(weights_dir)
        self._bg = imageio.get_reader(str(bg_video_path))
        self._bg_n = int(self._bg.count_frames())
        self._bg_idx = 0
        self._windower = AudioWindower(sr=self.sr, fps=self.fps)
        self._tracker = LandmarkTracker()
        self._cropper = FaceCropper()
        self._pending = deque()  # (chunk (50,384), pts seconds)
        self._last_frame = None
        self._reuse = 0
        log.info(
            "MuseTalkSession ready: bg=%s frames=%d step=%d (%.1fms)",
            bg_video_path, self._bg_n, self.step, self.step / self.sr * 1000,
        )

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
        # Drain every fully-buffered 5s window into per-step audio chunks.
        # TODO(phase2): prefix context cache to smooth the window boundary.
        while True:
            w, pts0 = self._windower.pop_window()
            if w is None:
                break
            mel = log_mel_spectrogram(mx.array(w))  # (1,80,3000), 30s pad
            chunks = self.pipe.encode_audio(mel, len(w), fps=self.fps)
            n = chunks.shape[0]
            for i in range(n):
                self._pending.append((chunks[i], pts0 + i * self.step / self.sr))
            log.debug("encoded window pts=%.3fs -> %d chunks", pts0, n)

    def _render(self, chunk) -> np.ndarray:
        tier = thermal_tier()
        if tier >= config.THERMAL_CRITICAL:
            # Frame reuse: infer 1 of FRAME_REUSE steps, repeat the rest (FR-END-003).
            self._reuse = (self._reuse + 1) % config.FRAME_REUSE
            if self._reuse and self._last_frame is not None:
                return self._last_frame
        else:
            self._reuse = 0
        frame = self._bg_frame()
        landmarks = self._tracker.update(frame)
        if landmarks is None or self._tracker.idle:
            # Idle-blink fallback: emit the base frame unchanged, no lip drive.
            return frame
        crop, bbox = self._cropper.crop(frame, landmarks)
        latent = self.pipe.get_latents_for_unet(crop)
        dtype = getattr(self.pipe, "_dtype", mx.float32)
        face = self.pipe.generate_faces(latent, chunk[None].astype(dtype))[0]
        out = paste_back(frame, face, bbox)
        self._last_frame = out
        return out

    def _bg_frame(self) -> np.ndarray:
        idx = self._bg_idx % self._bg_n if self._bg_n > 0 else self._bg_idx
        try:
            rgb = self._bg.get_data(idx)
        except (IndexError, ValueError):
            self._bg.seek(0)
            rgb = next(self._bg)
        self._bg_idx += 1
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
