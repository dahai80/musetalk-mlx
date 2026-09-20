import logging
import queue
import threading
import time
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
from ..pipeline.blending import crop_bbox_to_xyxy, paste_back
from ..pipeline.lcm import LCMFastSession
from ..utils.audio import AudioWindower
from ..utils.profiling import StageProfiler
from ..utils.thermal import ladder_for_tier, thermal_tier

log = logging.getLogger(__name__)


def _unet_forward(pipe, latent, chunk):
    # Plain (uncompiled) single-step UNet forward; PE applied inside,
    # mirroring pipe.generate_faces ordering.
    from fusion_mlx.video.musetalk_mlx.config import UNET_TIMESTEP
    from fusion_mlx.video.musetalk_mlx.whisper.audio2feature import apply_pe

    dtype = getattr(pipe, "_dtype", None) or mx.float32
    audio = mx.array(chunk[None]).astype(dtype)
    # fp32 VAE-encode latents must not leak into the UNet call in fp16 mode:
    # mixed dtype forces a separate (slow) fp32 kernel specialization.
    if latent.dtype != dtype:
        latent = latent.astype(dtype)
    return pipe.unet(latent, mx.array([UNET_TIMESTEP]), apply_pe(audio))


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
        if config.FP16 and hasattr(self.pipe, "astype"):
            # PRD 30FPS + <=4GB: fp16 halves weight memory and keeps the
            # single-frame UNet call inside the 33ms budget on M-series GPU.
            self.pipe.astype(mx.float16)
            log.info("pipeline cast to fp16")
        self.lcm = LCMFastSession(self.pipe)
        self._bg = imageio.get_reader(str(bg_video_path))
        self._bg_n = int(self._bg.count_frames())
        self._bg_idx = 0
        self._windower = AudioWindower(sr=self.sr, fps=self.fps)
        self._tracker = LandmarkTracker()
        self._cropper = FaceCropper()
        self._mask = mask_provider if mask_provider is not None else load_face_parse()
        self._pending = deque()  # (chunk (50,384), pts seconds)
        self._out_q = deque()  # rendered (frame, pts) awaiting get_output_frame
        self._inflight = deque()  # lazily dispatched rounds (img, metas, frames, items)
        self._last_frame = None
        self._reuse = 0
        self._audio_prefix = None  # fusion-mlx #914: prior window's tail embedding
        self.profiler = StageProfiler()
        self._croppers = {}  # patch size -> FaceCropper (thermal ladder 256/128)
        self._bg_pool = None  # FR-END-002: preloaded base-video frame pool
        self._bg_cache = []  # per-frame (landmarks, bbox, latent) precompute
        self._expected_pts = None  # observability: PTS sync-deviation logging
        self._tune_mlx_memory()
        self._setup_graph_pass()
        self._preload_bg()
        self._set_render_cache()
        # Paste worker thread: warp/blend/emit are pure cv2/numpy and run
        # OFF the render thread so they overlap the GPU rounds (measured
        # ~20ms/round of CPU post-processing that depth-2 render-ahead
        # could not fully hide). The worker never calls MLX — mask cache
        # misses are handed back to the main thread.
        self._paste_q = queue.Queue()
        self._paste_fallback = deque()
        self._emit_lock = threading.Lock()
        self._paste_worker_t = threading.Thread(target=self._paste_worker, name="musetalk-paste", daemon=True)
        self._paste_worker_t.start()
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

    @staticmethod
    def _tune_mlx_memory() -> None:
        # Pre-load: only a loose safety cap. Setting a TIGHT budget or cache
        # limit here poisons the allocator watermark for the whole process:
        # limits active during load/precompute made every later render round
        # re-allocate (84-92ms vs 62-67ms per b2 round measured on MLX
        # 0.32.0). Real caps land post-warmup in _set_render_cache.
        limiter = getattr(mx, "set_memory_limit", None) or getattr(
            getattr(mx, "metal", None), "set_memory_limit", None
        )
        try:
            if limiter is not None:
                limiter(32 * 1024 * 1024 * 1024)
            log.info("MLX pre-load memory safety cap 32GB")
        except Exception as e:
            log.info("MLX memory tuning unavailable (%s); defaults kept", e)

    @staticmethod
    def _set_render_cache() -> None:
        # Post-warmup caps (sweep on MLX 0.32.0, compiled joint graph, b2:
        # 61.8/58.9/59.1/57.0/56.9/58.4 ms per round at 2.5/3/3.5/4/5/6GB
        # cache) — 4GB keeps every decode intermediate resident without
        # hoarding. Memory budget must also clear the real working set: live
        # peak is ~4.6GB, and a 3GB budget made the allocator reclaim
        # mid-round, costing +27ms/round.
        setter = getattr(mx, "set_cache_limit", None) or getattr(
            getattr(mx, "metal", None), "set_cache_limit", None
        )
        limiter = getattr(mx, "set_memory_limit", None) or getattr(
            getattr(mx, "metal", None), "set_memory_limit", None
        )
        try:
            if setter is not None:
                setter(4 * 1024 * 1024 * 1024)
            if limiter is not None:
                limiter(8 * 1024 * 1024 * 1024)
            log.info("MLX render memory tuned: cache<=4GB, limit 8GB (post-warmup)")
        except Exception as e:
            log.info("MLX cache tuning unavailable (%s); defaults kept", e)

    def reload(self, weights_dir=None, mlx_dir=None) -> bool:
        # V2 FR-MLX-006 / FR-END-005: hot-reload weights without dropping audio.
        # Drain pending to a consistent boundary, build the new pipeline, swap
        # atomically. During the swap get_output_frame() emits standby base frames
        # (no black screen). Returns True on success.
        wd = weights_dir or self._weights_dir
        md = mlx_dir if mlx_dir is not None else self._mlx_dir
        log.info("ReloadModel: draining %d pending chunks, rebuilding pipe", len(self._pending))
        self._pending.clear()
        self._inflight = deque()  # stale graph output belongs to the old pipe
        self._audio_prefix = None  # reset prefix cache on model swap
        try:
            new_pipe = self._build_pipe(wd, md)
        except Exception as e:
            log.error("ReloadModel failed (%s); keeping old pipe, no black screen", e)
            return False
        self.pipe = new_pipe
        self.lcm = LCMFastSession(new_pipe)
        self._setup_graph_pass()
        if config.FP16 and hasattr(self.pipe, "astype"):
            new_pipe.astype(mx.float16)
        if getattr(self, "_bg_pool", None):
            # cached latents belong to the old pipe dtype/weights; rebuild
            self._precompute_cache(self._bg_pool)
        self._set_render_cache()
        self._weights_dir = wd
        self._mlx_dir = md
        log.info("ReloadModel done")
        return True

    def push_audio(self, pcm: np.ndarray) -> None:
        self._windower.push(pcm)

    def end_of_stream(self) -> None:
        # Offline boundary: zero-pad the sub-window audio tail so the final
        # <5s renders instead of being silently dropped. No-op mid-stream.
        if self._windower.flush():
            self._encode_windows()

    def get_output_frame(self):
        # Render the next 33ms step. Returns (frame_bgr, pts) or None.
        self._encode_windows()
        if self._out_q:
            return self._out_q.popleft()
        if not self._pending and not self._inflight:
            if self._paste_fallback:
                # mask-cache miss deferred from the worker: fill + paste here
                item = self._paste_fallback.popleft()
                out = paste_back(item[0], item[1], item[2], mask_provider=self._mask)
                with self._emit_lock:
                    self._last_frame = out
                    self._emit(out, item[4])
            elif not self._paste_q.empty():
                time.sleep(0.002)  # worker draining the tail
            if self._out_q:
                return self._out_q.popleft()
            return None
        tier = thermal_tier()
        ladder = ladder_for_tier(tier)
        normal = ladder["frame_reuse"] == 1 and ladder["patch"] == 256 and ladder["bg_downscale"] == 1
        if config.BATCH > 1 and normal and self._compiled_generate is not None:
            # Render-ahead depth 2: keep TWO lazy rounds queued so the GPU
            # stays busy across the paste/emit CPU window (single stream runs
            # rounds serially, but the queue must never drain while the CPU
            # does numpy/cv2 work). Depth 1 left ~20ms/round of GPU idle
            # (measured 89.5ms/round wall vs 72 ideal).
            while len(self._inflight) < 2:
                if not self._submit_round():
                    break
            if not self._inflight:
                # cache miss / compiled failure: whole round falls back singly
                self._render_all_singly()
                if self._out_q:
                    return self._out_q.popleft()
                return None
            state = self._inflight.popleft()
            self._finish_round(state)
            if self._out_q:
                return self._out_q.popleft()
            return None
        chunk, pts = self._pending.popleft()
        frame = self._render(chunk)
        self._emit(frame, pts)
        return self._out_q.popleft()

    def _emit(self, frame, pts) -> None:
        # Single egress point: PTS sync-deviation observability (FR-LK-001).
        self.profiler.tick()
        if self._expected_pts is not None:
            dev = abs(pts - self._expected_pts)
            if dev > 2 * self.step / self.sr:
                log.warning("PTS sync deviation %.1fms at pts=%.3fs", dev * 1000, pts)
        self._expected_pts = pts + self.step / self.sr
        self._out_q.append((frame, pts))

    def _submit_round(self) -> bool:
        # Batched hot path (PRD 30FPS): one UNet + one VAE decode per BATCH
        # steps, dispatched LAZILY (no sync). RTT-safe: BATCH=2 adds one step
        # (66ms) — within the <=80ms audio-to-video budget. Cache miss (idle
        # frame / thermal) requeues and returns False; caller falls back singly.
        pf = self.profiler
        n = min(config.BATCH, len(self._pending))
        items = [self._pending.popleft() for _ in range(n)]
        frames, latents, chunks, metas = [], [], [], []
        pf.begin("frame_out")
        for chunk, pts in items:
            frame, bg_idx = self._bg_frame()
            cached = None
            if self._bg_cache and bg_idx < len(self._bg_cache):
                cached = self._bg_cache[bg_idx]
            if cached is None:
                # cache miss (idle/thermal frame): render this round singly
                pf.end()
                self._pending.extendleft(reversed(items))
                return False
            frames.append(frame)
            latents.append(cached[2])
            chunks.append(chunk)
            metas.append(cached)
        pf.end()
        if not latents:
            return False
        dtype = getattr(self.pipe, "_dtype", None) or mx.float32
        pf.begin("unet")
        # Both args must be dtype-exact: VAE-encode latents come back fp32
        # (SafeGroupNorm fp32 protect), and a mixed fp32/fp16 call makes
        # mx.compile specialize a slow fp32 graph (~50x slower measured).
        lat = mx.concatenate(latents, 0).astype(dtype)
        ch = mx.concatenate([mx.array(c[None]) for c in chunks], 0).astype(dtype)
        try:
            # joint UNet+VAE-decode graph; output is RGB [0,1] (B,3,256,256)
            img = self._compiled_generate(lat, ch)
        except Exception as e:
            log.warning("compiled batched render failed (%s); plain path", e)
            self._compiled_generate = None
            self._pending.extendleft(reversed(items))
            return False
        pf.end()
        self._inflight.append((img, metas, frames, items))
        return True

    def _paste_worker(self) -> None:
        # Render-thread offload: paste + emit for rounds whose parse-mask was
        # a cache hit (the common case — mask is static per identity). Items
        # are (frame, face, bbox_xyxy, alpha, pts); alpha None means the main
        # thread must fill the mask first (worker must not touch MLX).
        while True:
            item = self._paste_q.get()
            if item is None:
                return
            frame, face, bbox, alpha, pts = item
            if alpha is None:
                self._paste_fallback.append(item)
                continue
            out = paste_back(frame, face, bbox, alpha=alpha)
            with self._emit_lock:
                self._last_frame = out
                self._emit(out, pts)

    def _paste_item(self, frame, face, bbox, pts) -> None:
        # Main-thread entry: fetch alpha (fills the parse cache on miss) and
        # hand the item to the worker. Empty alpha (degenerate crop) pastes
        # inline via the provider's feather fallback.
        alpha, _ = self._mask.mouth_mask(frame, bbox)
        if alpha.size != 0:
            self._paste_q.put((frame, face, bbox, alpha, pts))
        else:
            out = paste_back(frame, face, bbox, mask_provider=self._mask)
            with self._emit_lock:
                self._last_frame = out
                self._emit(out, pts)

    def _finish_round(self, state) -> None:
        # Sync + materialize a previously submitted round. By now the NEXT
        # round is already dispatched, so this sync rides on a deep GPU queue.
        img, metas, frames, items = state
        pf = self.profiler
        pf.begin("vae_dec")
        faces = self._decode_faces(img)
        pf.end()
        for i, meta in enumerate(metas):
            pf.begin("warp")
            self._paste_item(frames[i], faces[i], crop_bbox_to_xyxy(meta[1]), items[i][1])
            pf.end()

    def _render_all_singly(self) -> None:
        while self._pending:
            chunk, pts = self._pending.popleft()
            self._emit(self._render(chunk), pts)

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
                # Chunk i is frame (i - skip) of this window: PTS advances 1/fps
                # per chunk (step/sr is 533/16000 — off by 0.67% and drifts).
                self._pending.append((chunks[i], pts0 + (i - skip) / self.fps))
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
        frame, bg_idx = self._bg_frame(downscale=ladder["bg_downscale"])
        # FR-END-003 strategy D (last resort): face patch 256 -> 128; croppers
        # cached per size so no 256->128 jump path skips the earlier rungs.
        patch = ladder["patch"]
        cached = None
        if self._bg_cache and patch == 256 and bg_idx < len(self._bg_cache):
            cached = self._bg_cache[bg_idx]
        if cached is not None:
            landmarks, bbox, latent = cached
            pf.end()
        else:
            landmarks = self._tracker.update(frame)
            if landmarks is None or self._tracker.idle:
                pf.end()
                return frame
            if patch not in self._croppers:
                self._croppers[patch] = FaceCropper(size=patch, upperbondrange=self._cropper.upperbondrange)
            crop, bbox = self._croppers[patch].crop(frame, landmarks)
            pf.end()
            pf.begin("vae")
            latent = self.pipe.get_latents_for_unet(crop)
            if self.pipe._dtype is not None:
                latent = latent.astype(self.pipe._dtype)
            pf.end()
        pf.begin("unet")
        face = None
        if self._compiled_generate is not None:
            try:
                dtype = getattr(self.pipe, "_dtype", None) or mx.float32
                img = self._compiled_generate(latent, mx.array(chunk[None]).astype(dtype))
            except Exception as e:
                log.warning("compiled render failed (%s); plain path", e)
                self._compiled_generate = None
            else:
                pf.end()
                pf.begin("vae_dec")
                face = self._decode_faces(img)[0]
                pf.end()
        if face is None:
            pred = _unet_forward(self.pipe, latent, chunk)
            pf.end()
            pf.begin("vae_dec")
            face = self.pipe.decode_latents(pred)[0]
            pf.end()
        pf.begin("warp")
        out = paste_back(frame, face, crop_bbox_to_xyxy(bbox), mask_provider=self._mask)
        pf.end()
        self._last_frame = out
        return out

    def _apply_ddim_steps(self, steps: int) -> None:
        setter = getattr(self.pipe, "set_ddim_steps", None)
        if setter is not None:
            setter(steps)
            log.debug("ddim steps set to %d", steps)

    def _decode_faces(self, img) -> np.ndarray:
        # Compiled-joint path output: RGB float [0,1] (B,3,256,256) -> BGR
        # uint8, same contract as pipe.decode_latents. Transpose + scale run
        # in numpy: the MLX-side transpose/fp32 cast forced two extra GPU
        # copies before a 2x-bytes readback for zero visual gain.
        arr = np.array(img)
        return ((arr.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8))[..., ::-1]

    def _setup_graph_pass(self) -> None:
        # FR-MLX-003: consume the fusion-mlx #911/#918 graph passes when
        # available: structural rewrite (Conv+GN+SiLU fusion) + SmartConv2d
        # shape-dispatched conv (#919) on the module tree, then compile the
        # joint UNet+VAE-decode forward (PE applied inside, mirroring
        # generate_faces; generate_faces itself calls mx.eval, illegal under
        # mx.compile). One graph for unet->decode is ~2x faster than two
        # compiled calls (no compiled-input boundary).
        self._compiled_generate = None
        if not config.GRAPH_OPT:
            log.info("graph pass disabled by config flag")
            return
        try:
            from fusion_mlx.graph_opt import apply_patterns, apply_smart_conv, compile_with_custom_pass
            from fusion_mlx.video.musetalk_mlx.config import UNET_TIMESTEP
            from fusion_mlx.video.musetalk_mlx.whisper.audio2feature import apply_pe

            n_pat = apply_patterns(self.pipe.unet) + apply_patterns(self.pipe.vae)
            n_sc = 0
            if config.SMART_CONV:
                n_sc = apply_smart_conv(self.pipe.unet) + apply_smart_conv(self.pipe.vae)
            log.info("graph passes: %d pattern rewrites, %d smart convs", n_pat, n_sc)
            pipe = self.pipe

            def render_fn(latent, audio):
                pred = pipe.unet(latent, mx.array([UNET_TIMESTEP]), apply_pe(audio))
                return mx.clip(pipe.vae.decode(pred / pipe.scaling_factor) / 2 + 0.5, 0, 1)

            self._compiled_generate = compile_with_custom_pass(render_fn)
            log.info("fusion-mlx graph pass applied to joint UNet+decode (#911/#918)")
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
        if config.PRECOMPUTE:
            self._precompute_cache(frames)

    def _precompute_cache(self, frames) -> None:
        # MuseTalk-realtime-style offline pass: per base frame precompute
        # landmarks, crop bbox and VAE latent once, so the hot loop skips
        # DWPose (~100ms/frame) and VAE encode (~50ms/frame). Latents are
        # tiny ((1,8,32,32) fp16 ≈ 16KB/frame). Live fallback kept for
        # cache misses (idle frames, thermal patch 128).
        cache = []
        t0 = time.monotonic()
        for i, fr in enumerate(frames):
            lm = self._tracker.update(fr)
            if lm is None:
                cache.append(None)
                continue
            crop, bbox = self._cropper.crop(fr, lm)
            lat = self.pipe.get_latents_for_unet(crop)
            mx.eval(lat)
            cache.append((lm, bbox, lat))
            if (i + 1) % 100 == 0:
                log.info("bg cache precompute %d/%d (%.1fs)", i + 1, len(frames), time.monotonic() - t0)
        self._bg_cache = cache
        self._tracker.fails = 0
        self._tracker.idle = False
        log.info(
            "bg cache precomputed %d/%d frames in %.1fs",
            sum(1 for c in cache if c is not None),
            len(cache),
            time.monotonic() - t0,
        )

    def _bg_frame(self, downscale: int = 1):
        # Serve from the preloaded pool (FR-END-002); looped. downscale > 1
        # (FR-END-003 strategy C) renders the frame at reduced res then
        # upsamples — compute saved outside the face ROI, face patch unaffected.
        # Returns (frame_bgr, pool_index).
        if self._bg_pool:
            rgb = self._bg_pool[self._bg_idx % len(self._bg_pool)]
            idx = self._bg_idx % len(self._bg_pool)
            self._bg_idx += 1
            if downscale > 1:
                h, w = rgb.shape[:2]
                small = cv2.resize(rgb, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
                rgb = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            return rgb, idx
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
        return rgb, idx
