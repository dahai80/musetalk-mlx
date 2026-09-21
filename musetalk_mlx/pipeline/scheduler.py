import logging
import time
from collections import deque

import mlx.core as mx
import numpy as np

from .. import config
from ..face.crop import FaceCropper
from ..pipeline.blending import crop_bbox_to_xyxy, paste_back

log = logging.getLogger(__name__)


class RenderScheduler:
    # Render scheduling split out of MuseTalkSession (audit 0921 A-1 step 4b):
    # batched round submission (lazy graphs, render-ahead depth 2), round
    # finishing, thermal single-path fallback and the plain single render.
    # Collaborators are injected; pipe/tracker go through GETTERS because the
    # session swaps pipe on reload and tests substitute the tracker — a
    # captured ref would keep old weights alive (4GB budget) or miss swaps.

    def __init__(
        self,
        *,
        pending,
        out_q,
        pipe_getter,
        gens_getter,
        clear_gens,
        ladder_provider,
        bg_frame,
        bg_cache_snapshot,
        emit,
        paste_item,
        mask,
        profiler,
        cropper,
        croppers,
        tracker_getter,
        last_frame_getter,
        last_frame_setter,
    ):
        self._pending = pending  # shared deque ref (session-owned)
        self._out_q = out_q  # shared deque ref (session-owned)
        self._pipe_getter = pipe_getter
        self._gens_getter = gens_getter
        self._clear_gens = clear_gens
        self._ladder_provider = ladder_provider
        self._bg_frame = bg_frame
        self._bg_cache_snapshot = bg_cache_snapshot
        self._emit = emit
        self._paste_item = paste_item
        self._mask = mask
        self.profiler = profiler
        self._cropper = cropper
        self._croppers = croppers  # shared dict ref: patch size -> FaceCropper
        self._tracker_getter = tracker_getter
        self._last_frame_getter = last_frame_getter
        self._last_frame_setter = last_frame_setter
        self._inflight = deque()  # lazily dispatched rounds (img, metas, frames, items)
        self._reuse = 0
        self._ddim_unavailable_logged = False

    def next_step(self, ladder, pop: bool = True) -> object | None:
        # Hot-loop body of get_output_frame after the idle checks (see session).
        # pop=False: the dedicated render thread drives this — round(s) are
        # rendered+emitted but the frame stays in out_q for the consumer
        # (get_output_frame); popping here would make the producer eat its own
        # output (A3 finding 2026-09-21: 150 rounds rendered, 0 frames in q).
        # Single thermal read per frame: the stateless thermal_tier() reads OS
        # state each call, and two reads in one frame (get_output_frame + _render)
        # can straddle a state flip — batched dispatch commits to normal while
        # _render then applies critical, jumping tiers (audit A4). The controller
        # also applies hysteresis so fair<->serious chatter does not flap steps.
        normal = ladder["frame_reuse"] == 1 and ladder["patch"] == 256 and ladder["bg_downscale"] == 1
        gen = self._gens_getter()
        if config.BATCH > 1 and normal and gen is not None:
            # Reset steps: the batched path skips _render, so a prior thermal
            # downgrade (single-path, low steps) would otherwise leave reduced
            # steps stuck after recovery. Normal tier = default DDIM steps (audit P1-29).
            self._apply_ddim_steps(ladder["ddim_steps"])
            # Render-ahead depth 2: keep TWO lazy rounds queued so the GPU
            # stays busy across the paste/emit CPU window (single stream runs
            # rounds serially, but the queue must never drain while the CPU
            # does numpy/cv2 work). Depth 1 left ~20ms/round of GPU idle
            # (measured 89.5ms/round wall vs 72 ideal).
            while len(self._inflight) < 2:
                if not self._submit_round():
                    break
            if not self._inflight:
                # cache miss / compiled failure: whole round falls back singly.
                # Pass the already-read ladder so _render does NOT re-sample
                # (audit A4 — prior code re-read thermal_tier() here, racing
                # the dispatch decision).
                self._render_all_singly(ladder)
                if not pop:
                    return None
                if self._out_q:
                    return self._out_q.popleft()
                return None
            state = self._inflight.popleft()
            self._finish_round(state)
            if not pop:
                # Wait until the paste/emit for this round has completed, then
                # leave the frame(s) queued for the consumer (pop=False: do
                # NOT take one — see _wait_out_q).
                self._wait_out_q(pop=False)
                return None
            return self._wait_out_q()
        chunk, pts = self._pending.popleft()
        frame = self._render(chunk, ladder)
        self._emit(frame, pts)
        if not pop:
            return None
        return self._out_q.popleft()

    def _wait_out_q(self, timeout_s: float = 0.2, pop: bool = True):
        # Paste/emit runs on the worker thread (mp paste round trip ~15ms);
        # briefly wait so None keeps its pre-async meaning: nothing pending
        # anywhere, not "paste still in flight". Poll, never block forever.
        # pop=False (render-thread producer path): wait for emit WITHOUT
        # taking a frame — popping here discards it (the caller returns None)
        # and half the rendered output never reaches the consumer (A3 finding
        # 2026-09-21: 1500 emitted, 750 published, out_q empty — 2:1 loss).
        deadline = time.monotonic() + timeout_s
        while not self._out_q:
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)
        return self._out_q.popleft() if pop else None

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
            # Snapshot the cache list under the lock so a concurrent reload
            # swap cannot detach the reference mid-iteration (audit R8).
            bg_cache = self._bg_cache_snapshot()
            if bg_cache and bg_idx < len(bg_cache):
                cached = bg_cache[bg_idx]
            if cached is None:
                # cache miss (idle/thermal frame): render this round singly
                pf.end()
                self._pending.extendleft(reversed(items))
                return False
            frames.append(frame)
            latents.append(cached.latent)
            chunks.append(chunk)
            metas.append(cached)
        pf.end()
        if not latents:
            return False
        pipe = self._pipe_getter()
        dtype = pipe.dtype  # #928 public accessor
        pf.begin("unet_build")
        # Both args must be dtype-exact: VAE-encode latents come back fp32
        # (SafeGroupNorm fp32 protect), and a mixed fp32/fp16 call makes
        # mx.compile specialize a slow fp32 graph (~50x slower measured).
        lat = mx.concatenate(latents, 0).astype(dtype)
        ch = mx.concatenate([mx.array(c[None]) for c in chunks], 0).astype(dtype)
        try:
            # joint UNet+VAE-decode graph; output is RGB [0,1] (B,3,H,W)
            img = self._gens_getter()(lat, ch)
        except Exception as e:
            log.warning("compiled batched render failed (%s); plain path", e)
            self._clear_gens()
            self._pending.extendleft(reversed(items))
            return False
        pf.end()
        self._inflight.append((img, metas, frames, items))
        return True

    def _finish_round(self, state) -> None:
        # Sync + materialize a previously submitted round. By now the NEXT
        # round is already dispatched, so this sync rides on a deep GPU queue.
        # The "render_eval" stage times the numpy readback, which is where the
        # LAZY unet+decode graph from _submit_round actually executes on GPU.
        # It is NOT a decode-only cost — it includes the deferred unet eval too
        # (audit B4 profiler honesty). The "unet_build" stage in _submit_round
        # is the CPU-side graph construction only.
        img, metas, frames, items = state
        pf = self.profiler
        pf.begin("render_eval")
        faces = self._decode_faces(img)
        pf.end()
        for i, meta in enumerate(metas):
            pf.begin("warp")
            self._paste_item(frames[i], faces[i], crop_bbox_to_xyxy(meta[1]), items[i][1])
            pf.end()

    def _render_all_singly(self, ladder) -> None:
        # Inherit the caller's thermal ladder — _render must NOT re-sample the
        # OS thermal state (two reads in one frame can straddle a flip, audit A4).
        while self._pending:
            chunk, pts = self._pending.popleft()
            self._emit(self._render(chunk, ladder), pts)

    def _render(self, chunk, ladder=None) -> np.ndarray:
        pf = self.profiler
        pf.begin("frame_out")
        # Caller passes the already-read ladder (single thermal read per frame,
        # audit A4). Only read here when called outside next_step (none
        # today, but keep a safe fallback rather than asserting).
        if ladder is None:
            ladder = self._ladder_provider()
        if ladder["frame_reuse"] > 1:
            self._reuse = (self._reuse + 1) % ladder["frame_reuse"]
            if self._reuse and self._last_frame_getter() is not None:
                # Copy: a consumer that mutates the returned buffer in place
                # would otherwise pollute _last_frame and corrupt the next
                # reused frame (audit P1-34). Non-reuse frames share the alias
                # by contract — consumers must treat emitted frames read-only.
                return self._last_frame_getter().copy()
        else:
            self._reuse = 0
        # step-count / ICB need fusion-mlx #911/#912; apply if the pipe exposes it.
        self._apply_ddim_steps(ladder["ddim_steps"])
        frame, bg_idx = self._bg_frame(downscale=ladder["bg_downscale"])
        # FR-END-003 strategy D (last resort): face patch 256 -> 128; croppers
        # cached per size so no 256->128 jump path skips the earlier rungs.
        patch = ladder["patch"]
        cached = None
        bg_cache = self._bg_cache_snapshot()
        if bg_cache and patch == 256 and bg_idx < len(bg_cache):
            cached = bg_cache[bg_idx]
        pipe = self._pipe_getter()
        if cached is not None:
            landmarks, bbox, latent = cached
            pf.end()
        else:
            tracker = self._tracker_getter()
            landmarks = tracker.update(frame)
            if landmarks is None or tracker.idle:
                pf.end()
                return frame
            if patch not in self._croppers:
                self._croppers[patch] = FaceCropper(size=patch, upperbondrange=self._cropper.upperbondrange)
            crop, bbox = self._croppers[patch].crop(frame, landmarks)
            pf.end()
            pf.begin("vae")
            latent = pipe.get_latents_for_unet(crop)
            # #928: pipe.dtype is the public accessor (was pipe._dtype reach-in).
            if pipe.dtype is not None:
                latent = latent.astype(pipe.dtype)
            pf.end()
        pf.begin("unet")
        face = None
        gen = self._gens_getter()
        if gen is not None:
            # Compiled joint path: gen() returns a LAZY mx.array (graph built,
            # not executed). The real unet+decode eval fires at the numpy
            # readback in _decode_faces. Splitting unet/vae_dec stages here
            # misattributes ~all cost to vae_dec and reports ~0ms for unet
            # (audit B4 profiler honesty). Use ONE "render" stage so the
            # reported number is the true joint unet+decode cost.
            pf.end()
            pf.begin("render")
            try:
                dtype = pipe.dtype
                img = gen(latent, mx.array(chunk[None]).astype(dtype))
                face = self._decode_faces(img)[0]
            except Exception as e:
                log.warning("compiled render failed (%s); plain path", e)
                self._clear_gens()
                face = None
            pf.end()
        if face is None:
            pred = self._unet_forward(pipe, latent, chunk)
            if config.DECODE_128 and latent.shape[-1] == config.LATENT:
                pred = _pool2x(pred)
            pf.end()
            pf.begin("vae_dec")
            face = pipe.decode_latents(pred)[0]
            pf.end()
        pf.begin("warp")
        out = paste_back(frame, face, crop_bbox_to_xyxy(bbox), mask_provider=self._mask)
        pf.end()
        self._last_frame_setter(out)
        return out

    def _unet_forward(self, pipe, latent, chunk, steps=1):
        # Plain (uncompiled) UNet forward via the #928 public render core
        # (pipe._run_unet encapsulates timestep / apply_pe / dtype). steps=1 is
        # the realtime single-step t=0 fast path; the thermal ladder threads a
        # higher steps count only on the offline path (realtime stays
        # single-step — multi-step DDIM is Nx slower and breaks the 33ms budget).
        dtype = pipe.dtype  # #928 public accessor (was getattr(pipe, "_dtype"))
        audio = mx.array(chunk[None]).astype(dtype)
        if latent.dtype != dtype:
            latent = latent.astype(dtype)
        return pipe._run_unet(latent, audio, steps)

    def _apply_ddim_steps(self, steps: int) -> None:
        # #927 landed: fusion-mlx MuseTalkPipeline.set_ddim_steps exists and
        # sets the pipe's _ddim_steps state. NOTE: musetalk-mlx's realtime render
        # path (_unet_forward / compiled closures) calls pipe._run_unet with
        # steps=1 (single-step t=0) to hold the 33ms budget — multi-step DDIM
        # (15/8) is Nx slower and breaks 30FPS. set_ddim_steps is therefore
        # invoked here for observability + future offline multi-step paths, but
        # the realtime hot loop stays single-step. The serious-tier compute
        # lever for realtime is patch/bg-downscale/frame-reuse at critical,
        # not step-cut.
        setter = getattr(self._pipe_getter(), "set_ddim_steps", None)
        if setter is not None:
            setter(steps)
            log.debug("ddim steps set to %d", steps)
            return
        if not self._ddim_unavailable_logged:
            self._ddim_unavailable_logged = True
            log.warning(
                "set_ddim_steps unavailable on fusion-mlx pipe; serious-tier "
                "step-cut is a no-op. Upgrade fusion-mlx >=0.10.3."
            )

    def _decode_faces(self, img) -> np.ndarray:
        # Compiled-joint path output: RGB float [0,1] (B,3,256,256) -> BGR
        # uint8, same contract as pipe.decode_latents. Transpose + scale run
        # in numpy: the MLX-side transpose/fp32 cast forced two extra GPU
        # copies before a 2x-bytes readback for zero visual gain.
        arr = np.array(img)
        return ((arr.transpose(0, 2, 3, 1) * 255).round().astype(np.uint8))[..., ::-1]


def _pool2x(z):
    # DECODE_128 (speed-over-quality switch): 2x2 average-pool the UNet
    # output latent before the VAE decoder. Decoder FLOPs scale with
    # spatial^2, so 32^2 -> 16^2 gives a 128^2 face patch at ~1/4 decode
    # cost. UNet + audio conditioning untouched. Pure ops, compile-safe.
    b, c, h, w = z.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"_pool2x requires even spatial dims, got {(h, w)}")
    return z.reshape(b, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))
