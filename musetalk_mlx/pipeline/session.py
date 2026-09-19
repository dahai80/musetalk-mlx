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
from ..utils.profiling import StageProfiler
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
        self._audio_prefix = None  # fusion-mlx #914: prior window's tail embedding
        self.profiler = StageProfiler()
        self._croppers = {}  # patch size -> FaceCropper (thermal ladder 256/128)
        self._bg_pool = None  # FR-END-002: preloaded base-video frame pool
        self._expected_pts = None  # observability: PTS sync-deviation logging
        self._setup_graph_pass()
        self._preload_bg()
        log.info(
            "MuseTalkSession ready: bg=%s frames=%d step=%d (%.1fms)",
            bg_video_path,
            len(self._bg_pool or []),
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
        self._audio_prefix = None  # reset prefix cache on model swap
        try:
            new_pipe = self._build_pipe(wd, md)
        except Exception as e:
            log.error("ReloadModel failed (%s); keeping old pipe, no black screen", e)
            return False
        self.pipe = new_pipe
        self.lcm = LCMFastSession(new_pipe)
        self._setup_graph_pass()
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
        self.profiler.tick()
        if self._expected_pts is not None:
            dev = abs(pts - self._expected_pts)
            if dev > 2 * self.step / self.sr:
                log.warning("PTS sync deviation %.1fms at pts=%.3fs", dev * 1000, pts)
        self._expected_pts = pts + self.step / self.sr
        return frame, pts

    def _encode_windows(self) -> None:
        # Drain every fully-buffered overlapping 5s window into per-step chunks.
        # The windower prepends the prior window's overlap tail for boundary
        # smoothing; those prefix chunks were already emitted, so skip them.
        # fusion-mlx #914: encode_audio returns (chunks, tail); the tail is the
        # embedding-level prefix spliced into the next window to smooth the
        # Transformer slice boundary (lip teleport at 5s edges).
        pf = self.profiler
        while True:
            pf.begin("stft")
            w, pts0 = self._windower.pop_window()
            if w is None:
                pf.end()
                break
            mel = log_mel_spectrogram(mx.array(w))  # (1,80,3000), 30s pad
            pf.end()
            pf.begin("whisper")
            chunks, tail = self.pipe.encode_audio(mel, len(w), fps=self.fps, prefix=self._audio_prefix)
            pf.end()
            self._audio_prefix = tail
            skip = round(self._windower.prefix_samples / self.step) if self._windower.prefix_samples else 0
            n = chunks.shape[0]
            for i in range(skip, n):
                self._pending.append((chunks[i], pts0 + (i - skip) * self.step / self.sr))
            log.debug("encoded window pts=%.3fs -> %d chunks (skipped %d prefix)", pts0, n - skip, skip)

    def _render(self, chunk) -> np.ndarray:
        pf = self.profiler
        pf.begin("frame_out")
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
            pf.end()
            return frame
        # FR-END-003 strategy D (last resort): face patch 256 -> 128; croppers
        # cached per size so no 256->128 jump path skips the earlier rungs.
        patch = ladder["patch"]
        if patch not in self._croppers:
            self._croppers[patch] = FaceCropper(size=patch, upperbondrange=self._cropper.upperbondrange)
        crop, bbox = self._croppers[patch].crop(frame, landmarks)
        pf.end()
        pf.begin("vae")
        latent = self.pipe.get_latents_for_unet(crop)
        pf.end()
        pf.begin("unet")
        face = None
        if self._compiled_generate is not None:
            try:
                face = self._compiled_generate(latent, chunk)[0]
            except Exception as e:
                log.warning("compiled generate failed (%s); plain path", e)
                self._compiled_generate = None
        if face is None:
            face = self.lcm.generate(latent, chunk)
        if face is None:
            dtype = getattr(self.pipe, "_dtype", mx.float32)
            face = self.pipe.generate_faces(latent, mx.array(chunk[None]).astype(dtype))[0]
        pf.end()
        pf.begin("warp")
        out = paste_back(frame, face, bbox, mask_provider=self._mask)
        pf.end()
        self._last_frame = out
        return out

    def _apply_ddim_steps(self, steps: int) -> None:
        setter = getattr(self.pipe, "set_ddim_steps", None)
        if setter is not None:
            setter(steps)
            log.debug("ddim steps set to %d", steps)

    def _setup_graph_pass(self) -> None:
        # FR-MLX-003: consume the fusion-mlx #911 graph pass (Conv+GN+SiLU
        # fusion) when available; mx.compile-wraps the UNet entry point.
        self._compiled_generate = None
        if not config.GRAPH_OPT:
            log.info("graph pass disabled by config flag")
            return
        try:
            from fusion_mlx.graph_opt import compile_with_custom_pass

            fn = self.pipe.generate_faces
            self._compiled_generate = compile_with_custom_pass(fn)
            log.info("fusion-mlx graph pass applied to generate_faces (#911)")
        except Exception as e:
            self._compiled_generate = None
            log.info("graph pass unavailable (%s); plain generate_faces", e)

    def _preload_bg(self) -> None:
        # FR-END-002: preload the base-video frames into a memory pool (looped
        # serving from RAM, no per-frame decode cost). Logs pool memory cost.
        frames = []
        try:
            for rgb in self._bg:
                frames.append(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
        except Exception as e:
            # imageio v2 reader raises at stream end; keep collected frames.
            log.debug("bg preload stopped at %d frames (%s)", len(frames), e)
        self._bg_pool = frames
        self._bg_n = len(frames)
        self._bg_idx = 0
        mb = sum(f.nbytes for f in frames) / (1024 * 1024)
        log.info("bg pool preloaded %d frames (pool %.0fMB)", len(frames), mb)

    def _bg_frame(self, downscale: int = 1) -> np.ndarray:
        # Serve from the preloaded pool (FR-END-002); looped. downscale > 1
        # (FR-END-003 strategy C) renders the frame at reduced res then
        # upsamples — compute saved outside the face ROI, face patch unaffected.
        if self._bg_pool:
            rgb = self._bg_pool[self._bg_idx % len(self._bg_pool)]
            self._bg_idx += 1
            if downscale > 1:
                h, w = rgb.shape[:2]
                small = cv2.resize(rgb, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
                rgb = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            return rgb
        idx = self._bg_idx % self._bg_n if self._bg_n > 0 else self._bg_idx
        try:
            rgb = self._bg.get_data(idx)
        except (IndexError, ValueError):
            self._bg.seek(0)
            rgb = next(self._bg)
        self._bg_idx += 1
        if downscale > 1:
            h, w = rgb.shape[:2]
            small = cv2.resize(rgb, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return rgb
