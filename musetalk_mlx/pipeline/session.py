import logging
import queue
import threading
import time
from collections import deque

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
from ..pipeline.paste_proc import PasteProcess
from ..utils.audio import AudioWindower
from ..utils.profiling import StageProfiler
from ..utils.thermal import ThermalController
from .background import BackgroundStore, BgCacheEntry  # noqa: F401 -- re-export (tests import from session)

log = logging.getLogger(__name__)


def _unet_forward(pipe, latent, chunk, steps=1):
    # Plain (uncompiled) UNet forward via the #928 public render core
    # (pipe._run_unet encapsulates timestep / apply_pe / dtype). steps=1 is the
    # realtime single-step t=0 fast path; the thermal ladder threads a higher
    # steps count only on the offline path (realtime stays single-step —
    # multi-step DDIM is  Nx slower and breaks the 33ms budget).
    dtype = pipe.dtype  # #928 public accessor (was getattr(pipe, "_dtype"))
    audio = mx.array(chunk[None]).astype(dtype)
    if latent.dtype != dtype:
        latent = latent.astype(dtype)
    return pipe._run_unet(latent, audio, steps)


def _pool2x(z):
    # DECODE_128 (speed-over-quality switch): 2x2 average-pool the UNet
    # output latent before the VAE decoder. Decoder FLOPs scale with
    # spatial^2, so 32^2 -> 16^2 gives a 128^2 face patch at ~1/4 decode
    # cost. UNet + audio conditioning untouched. Pure ops, compile-safe.
    b, c, h, w = z.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"_pool2x requires even spatial dims, got {(h, w)}")
    return z.reshape(b, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5))


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
        # LCM Phase-4 stub: the fusion-mlx pipe is single-step t=0 with no
        # distilled-weights API; LCMFastSession generates nothing distinct.
        # Instantiate only when explicitly enabled so the disabled default does
        # not allocate a dead object that misleads readers into thinking a
        # fast path exists (audit E2). When enabled, LCMFastSession validates
        # the pipe and raises on misuse.
        self.lcm = LCMFastSession(self.pipe) if config.LCM_ENABLED else None
        self._bg_store = BackgroundStore(bg_video_path)
        self._windower = AudioWindower(sr=self.sr, fps=self.fps)
        # Stateful thermal controller (hysteresis, one-tier-at-a-time downgrade)
        # — the stateless thermal_tier() wrapper chatters 1<->2 on M5 Max and
        # flaps ddim_steps 15<->8 mid-stream. Held per-session; reset on reload
        # (audit A1/A4/E1 — ThermalController was defined but never wired).
        self._thermal = ThermalController()
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
        self._croppers = {}  # patch 256/128 (thermal ladder); croppers dict stays on session
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
        # Bounded paste queue: an unbounded queue + a blocked IPC read would
        # OOM the process if the paste subprocess stalls (audit P0-4). Each
        # item holds a full bg frame + face; cap at a few rounds and let the
        # main thread fall back to inline paste when full.
        self._paste_q = queue.Queue(maxsize=max(config.BATCH * 4, 8))
        self._emit_lock = threading.RLock()
        self._out_q_cap = config.BATCH * 4  # drop oldest past this (no backpressure sink, audit P2-3)
        self._closed = False
        # atexit safety net: if the caller forgets close()/`with`, still tear
        # down the paste subprocess + worker thread at interpreter exit. __del__
        # is unreliable for cross-process cleanup (audit A6). Registered via a
        # WEAK reference: a strong `atexit.register(self.close)` pins the whole
        # session (bg_pool ~1.5GB + pipe ~2GB) until interpreter exit (audit
        # 0921 P1-3). The callback dies with the session object.
        import atexit
        import weakref

        session_ref = weakref.ref(self)

        def _atexit_close():
            s = session_ref()
            if s is not None:
                s.close()

        self._atexit_cb = _atexit_close
        atexit.register(_atexit_close)
        self._paste_worker_alive = True
        self._paste_worker_restarted = False  # one main-thread restart budget (0921 P0-2)
        self._paste_generation = 0  # bumped by drain; stale workers exit within 0.5s
        self._paste_worker_t = threading.Thread(target=self._paste_worker, name="musetalk-paste", daemon=True)
        self._paste_worker_t.start()
        # GIL bypass: the paste blend runs in a child process (own GIL); the
        # worker thread only does IPC. Fallback on start failure or any IPC
        # error is in-process paste_back on the worker thread.
        self._paste_mp = None
        if config.PASTE_MULTIPROC:
            try:
                self._paste_mp = PasteProcess()
                self._paste_mp.start()
            except Exception as e:
                log.warning("paste proc unavailable (%s); thread-only paste", e)
                self._paste_mp = None
        log.info(
            "MuseTalkSession ready: bg=%s frames=%d step=%d (%.1fms)",
            bg_video_path,
            len(self._bg_pool or []),
            self.step,
            self.step / self.sr * 1000,
        )

    # Compatibility shims (audit 0921 A-1 split): tests and internal render
    # paths reach the bg state through these names. Reads/writes delegate to
    # the store; the setter lazily binds a store shell so reload() unit-test
    # shells built via __new__ keep working unchanged.
    def _require_store(self):
        st = self.__dict__.get("_bg_store")
        if st is None:
            st = BackgroundStore.__new__(BackgroundStore)
            self._bg_store = st
        return st

    @property
    def _bg_pool(self):
        return self._bg_store._bg_pool

    @_bg_pool.setter
    def _bg_pool(self, v):
        self._require_store()._bg_pool = v

    @property
    def _bg_cache(self):
        return self._bg_store._bg_cache

    @_bg_cache.setter
    def _bg_cache(self, v):
        self._require_store()._bg_cache = v

    @property
    def _bg_cache_lock(self):
        return self._bg_store._bg_cache_lock

    @_bg_cache_lock.setter
    def _bg_cache_lock(self, v):
        self._require_store()._bg_cache_lock = v

    @staticmethod
    def _build_pipe(weights_dir, mlx_dir):
        # Pipeline-level SmartConv2d wrapping (fusion-mlx default ON) measured
        # 1.5x SLOWER inside the joint compiled UNet+decode graph — keep the
        # native conv2d path unless config.SMART_CONV opts in. Must be set
        # before from_pretrained* so the module tree stays unwrapped.
        import os

        # os.environ is process-global (audit 0921 A-3). Only writing "0" left
        # a stale value from a previous SMART_CONV-off session in the same
        # process, so a later SMART_CONV-on session silently stayed on the slow
        # path. Mirror config.SMART_CONV explicitly on every call.
        os.environ["FUSION_MUSETALK_SMART_CONV"] = "1" if config.SMART_CONV else "0"
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
        # Drain in-flight paste + emitted frames so old-model frames do not
        # leak into the post-reload stream (audit P1-27). Restart the worker
        # against a clean queue.
        self._drain_paste_queue()
        with self._emit_lock:
            self._out_q.clear()
            self._expected_pts = None
        self._audio_prefix = None  # reset prefix cache on model swap
        # Reset the thermal controller: a model swap changes the thermal
        # envelope (different weight memory), so held hysteresis state from the
        # old pipe is no longer valid (audit A1).
        self._thermal = ThermalController()
        try:
            new_pipe = self._build_pipe(wd, md)
        except Exception as e:
            log.error("ReloadModel failed (%s); keeping old pipe, no black screen", e)
            return False
        # Peak-memory guard (audit 0921 P1-1): swap FIRST, overwrite every
        # other strong ref to the old pipe (lcm), then drop it + flush the MLX
        # allocator cache BEFORE _precompute_cache — otherwise old-pipe weights
        # (~1-2GB) stay alive alongside the new pipe while precompute encodes
        # every bg frame, breaking the 4GB budget.
        old, self.pipe = self.pipe, new_pipe
        self.lcm = LCMFastSession(new_pipe) if config.LCM_ENABLED else None
        self._setup_graph_pass()
        old = None  # noqa: F841 - drop the last strong ref before clear_cache
        mx.clear_cache()
        if config.FP16 and hasattr(self.pipe, "astype"):
            new_pipe.astype(mx.float16)
        # cached latents belong to the old pipe dtype/weights; must rebuild.
        # The prior `if getattr(self, "_bg_pool", None)` was FALSY for an empty
        # pool ([]) — it cleared _bg_cache and never rebuilt, dropping the
        # session to the slow live DWPose+encode path (~150ms/frame, ~6fps)
        # permanently after reload (audit A8). Rebuild whenever a pool exists
        # (even empty); only skip when there is genuinely no pool attr.
        pool = getattr(self, "_bg_pool", None)
        if pool is not None:
            # Rebuild synchronously — _precompute_cache runs DWPose+VAE encode
            # per frame, which is slow, but an async rebuild races the render
            # thread on _bg_cache (R8) and on _tracker state (E4). Reload is an
            # explicit operator action; a bounded stall is acceptable, a
            # silent race is not. The live path in _render handles cache misses
            # (None entries) until the rebuild completes.
            self._precompute_cache(pool)
        else:
            with self._bg_cache_lock:
                self._bg_cache = []
        self._set_render_cache()
        self._weights_dir = wd
        self._mlx_dir = md
        log.info("ReloadModel done")
        return True

    def push_audio(self, pcm: np.ndarray) -> None:
        # Validate dtype/range so garbage PCM fails loudly, not silently into
        # Whisper (audit P2-5). NaN/Inf sanitized inside AudioWindower.push.
        arr = np.asarray(pcm)
        if arr.dtype != np.float32:
            log.warning("push_audio got %s, casting to float32", arr.dtype)
        self._windower.push(pcm)

    def _drain_paste_queue(self) -> None:
        # Flush + restart the paste worker against an empty queue so stale
        # old-model items do not emit after a reload (audit P1-27). The worker
        # is daemon; joining+respawning is bounded.
        #
        # Generation token (audit 0921 P0-2): bump FIRST so any stale worker
        # (incl. one that self-restarted under the old mechanism) exits within
        # its 0.5s get-timeout, then join, THEN install a fresh queue and
        # respawn on this (main) thread — no window with two workers consuming
        # one queue, no thread accumulation across reloads.
        self._paste_generation += 1
        try:
            self._paste_q.put_nowait(None)
        except queue.Full:
            pass
        if self._paste_worker_t is not None and self._paste_worker_t.is_alive():
            self._paste_worker_t.join(timeout=2.0)
        self._paste_q = queue.Queue(maxsize=max(config.BATCH * 4, 8))
        self._paste_worker_alive = True
        self._paste_worker_restarted = False  # fresh restart budget after reload
        # Only respawn if this session actually runs a paste worker (real
        # sessions always have one from __init__; shells built for unit tests
        # of reload() do not, and must not spawn a live worker here).
        if not self._closed and self._paste_worker_t is not None:
            self._paste_worker_t = threading.Thread(
                target=self._paste_worker, name="musetalk-paste", daemon=True
            )
            self._paste_worker_t.start()

    def close(self) -> None:
        # Lifecycle: stop the paste worker, kill the paste subprocess, close the
        # imageio reader. Idempotent; safe to call from atexit + explicit close.
        if self._closed:
            return
        self._closed = True
        try:
            import atexit

            if getattr(self, "_atexit_cb", None) is not None:
                atexit.unregister(self._atexit_cb)
                self._atexit_cb = None
        except Exception:
            pass
        # Generation bump makes the worker exit within 0.5s even if it is
        # blocked on an empty queue (audit 0921 P0-2).
        self._paste_generation += 1
        try:
            self._paste_q.put_nowait(None)  # sentinel: worker exits its loop
        except queue.Full:
            pass
        if self._paste_worker_t is not None and self._paste_worker_t.is_alive():
            self._paste_worker_t.join(timeout=2.0)
        if self._paste_mp is not None:
            try:
                self._paste_mp.stop()
            except Exception as e:
                log.warning("paste proc stop failed (%s)", e)
            self._paste_mp = None
        try:
            self._bg_store.close()
        except Exception as e:
            log.debug("bg reader close (%s)", e)
        log.info("MuseTalkSession closed")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # __del__ is NOT a reliable lifecycle hook: at interpreter shutdown the
        # module globals (queue/threading/subprocess) may be torn down, raising
        # inside close() and leaving the paste subprocess orphaned (audit A6).
        # atexit was the safety net while it held a STRONG ref — but that pinned
        # the session in memory (audit 0921 P1-3). With the weakref atexit
        # callback, a session garbage-collected without close() would lose its
        # safety net entirely, so the finalizer is now a best-effort close()
        # (idempotent, and modules are still alive mid-session). If __del__
        # fires during interpreter shutdown and fails, the warning below keeps
        # it observable.
        if getattr(self, "_closed", True):
            return
        try:
            self.close()
        except Exception:
            log.warning("MuseTalkSession finalized without clean close() — paste proc may leak")

    def end_of_stream(self) -> None:
        # Offline boundary: zero-pad the sub-window audio tail so the final
        # <5s renders instead of being silently dropped. No-op mid-stream.
        if self._windower.flush():
            self._encode_windows()

    def interrupt(self) -> int:
        # Barge-in (audit 0921 P3 — K12 conversation core requirement): drop
        # every not-yet-emitted artifact of the current utterance and return to
        # standby. Explicit API only — VAD-based auto-trigger is a future hook.
        # Returns the number of queued audio chunks (not frames) discarded, for
        # observability.
        if self._closed:
            return 0
        dropped = len(self._pending)
        self._pending.clear()
        # In-flight rounds hold LAZY mx graphs (submitted but not eval'd).
        # Dropping the ref cancels the never-started GPU work silently — no
        # sync needed, nothing was materialized yet.
        inflight = len(self._inflight)
        self._inflight.clear()
        with self._emit_lock:
            self._out_q.clear()
            self._expected_pts = None  # next emitted frame re-seeds the observer
        # _last_frame is intentionally KEPT: standby base frame, no black screen.
        # Prefix tail of the interrupted utterance would smear its embedding
        # into the next turn — same rationale as reload() (session.py:258).
        self._audio_prefix = None
        self._windower.reset()
        # Light paste-queue drain (get_nowait loop), NOT _drain_paste_queue():
        # that joins/respawns the worker (2s bound) which would dominate barge-in
        # latency. Tolerance: the worker may still emit <=2 frames (~66ms) it
        # already dequeued; generation is unchanged so the worker SURVIVES.
        drained = 0
        while True:
            try:
                self._paste_q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        log.info(
            "barge-in interrupt: %d pending chunks, %d inflight rounds, %d queued paste items discarded",
            dropped,
            inflight,
            drained,
        )
        return dropped

    def get_output_frame(self):
        # Render the next 33ms step. Returns (frame_bgr, pts) or None.
        self._encode_windows()
        if self._out_q:
            return self._out_q.popleft()
        if not self._pending and not self._inflight:
            # Nothing left to render; the paste worker may still be finishing
            # an in-flight round, so briefly wait before declaring idle.
            out = self._wait_out_q()
            if out is not None:
                return out
            return None
        # Single thermal read per frame: the stateless thermal_tier() reads OS
        # state each call, and two reads in one frame (get_output_frame + _render)
        # can straddle a state flip — batched dispatch commits to normal while
        # _render then applies critical, jumping tiers (audit A4). The controller
        # also applies hysteresis so fair<->serious chatter does not flap steps.
        ladder = self._thermal.ladder()
        normal = ladder["frame_reuse"] == 1 and ladder["patch"] == 256 and ladder["bg_downscale"] == 1
        gen = self._compiled_generate_128 if config.DECODE_128 else self._compiled_generate
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
                if self._out_q:
                    return self._out_q.popleft()
                return None
            state = self._inflight.popleft()
            self._finish_round(state)
            out = self._wait_out_q()
            if out is not None:
                return out
            return None
        chunk, pts = self._pending.popleft()
        frame = self._render(chunk, ladder)
        self._emit(frame, pts)
        return self._out_q.popleft()

    def _wait_out_q(self, timeout_s: float = 0.2):
        # Paste/emit runs on the worker thread (mp paste round trip ~15ms);
        # briefly wait so None keeps its pre-async meaning: nothing pending
        # anywhere, not "paste still in flight". Poll, never block forever.
        deadline = time.monotonic() + timeout_s
        while not self._out_q:
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)
        return self._out_q.popleft()

    def _emit(self, frame, pts) -> None:
        # Single egress point: PTS sync-deviation observability (FR-LK-001).
        # Self-locking (RLock) so every caller — worker thread, main-thread
        # single path, render_all_singly, inline fallback — serializes the
        # _expected_pts read-modify-write and _out_q append order. Without
        # this, batched + single paths emitted out of PTS order (audit P0-5).
        with self._emit_lock:
            self.profiler.tick()
            if self._expected_pts is not None:
                dev = abs(pts - self._expected_pts)
                if dev > 2 * self.step / self.sr:
                    log.warning("PTS sync deviation %.1fms at pts=%.3fs", dev * 1000, pts)
            self._expected_pts = pts + self.step / self.sr
            # Cap: a stalled consumer (LiveKit encoder) must not let _out_q
            # grow unbounded; drop the oldest past the cap (realtime tolerates
            # frame drop, not backlog) (audit P2-3).
            if len(self._out_q) >= self._out_q_cap:
                self._out_q.popleft()
                log.warning("out_q full (%d), dropped oldest frame", self._out_q_cap)
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
            # Snapshot the cache list under the lock so a concurrent reload
            # swap cannot detach the reference mid-iteration (audit R8).
            with self._bg_cache_lock:
                bg_cache = self._bg_cache
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
        dtype = self.pipe.dtype  # #928 public accessor
        pf.begin("unet_build")
        # Both args must be dtype-exact: VAE-encode latents come back fp32
        # (SafeGroupNorm fp32 protect), and a mixed fp32/fp16 call makes
        # mx.compile specialize a slow fp32 graph (~50x slower measured).
        lat = mx.concatenate(latents, 0).astype(dtype)
        ch = mx.concatenate([mx.array(c[None]) for c in chunks], 0).astype(dtype)
        try:
            # joint UNet+VAE-decode graph; output is RGB [0,1] (B,3,H,W)
            gen = self._compiled_generate_128 if config.DECODE_128 else self._compiled_generate
            img = gen(lat, ch)
        except Exception as e:
            log.warning("compiled batched render failed (%s); plain path", e)
            self._compiled_generate = None
            self._compiled_generate_128 = None
            self._pending.extendleft(reversed(items))
            return False
        pf.end()
        self._inflight.append((img, metas, frames, items))
        return True

    def _paste_worker(self) -> None:
        # Render-thread offload: paste + emit for rounds whose parse-mask was
        # a cache hit (the common case — mask is static per identity). Items
        # are (frame, face, bbox_xyxy, alpha, pts). Each frame is isolated in
        # try/except: a single paste failure must NOT kill the worker and
        # freeze the session — skip the frame and continue.
        #
        # Generation token (audit 0921 P0-2): the worker captures its own
        # generation + queue ref at start and exits within 0.5s when either
        # goes stale (reload drain bumps the generation and installs a new
        # queue). Restarts happen ONLY on main threads (drain / _paste_item) —
        # the prior self-restart-inside-the-dying-worker plus drain's
        # unconditional respawn left a window with two workers consuming one
        # queue and thread count growing every reload.
        gen = self._paste_generation
        q = self._paste_q
        while not self._closed and gen == self._paste_generation:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                log.error("paste worker queue get failed (%s); exiting", e)
                break
            if item is None:
                break
            try:
                frame, face, bbox, alpha, pts = item
                if alpha is None or alpha.size == 0:
                    # Worker thread: cache-hit-only mask lookup (audit 0921
                    # P2 — the prior full mouth_mask here could run the parse
                    # backend OFF the main thread, contradicting the "worker
                    # never calls MLX" contract). Miss -> feather paste; the
                    # main thread (_paste_item) backfills the cache.
                    a, _ = self._mask.mouth_mask_cached_only(frame, bbox)
                    if a is not None and a.size:
                        out = paste_back(frame, face, bbox, alpha=a)
                    else:
                        out = paste_back(frame, face, bbox)
                else:
                    out = None
                    if self._paste_mp is not None:
                        out = self._paste_mp.paste(frame, face, bbox, alpha)
                    if out is None:
                        out = paste_back(frame, face, bbox, alpha=alpha)
                with self._emit_lock:
                    self._last_frame = out
                    self._emit(out, pts)
            except Exception as e:
                # One bad frame (degenerate bbox, cv2 shape mismatch) must not
                # take down realtime output. Log and keep the worker alive.
                log.error("paste worker frame failed at pts=%.3fs (%s); skipped", pts, e)
        # Worker exiting: mark dead ONLY. Restart responsibility lives on the
        # main thread (_paste_item watchdog, audit 0921 P0-2) — the prior
        # self-restart here raced _drain_paste_queue's unconditional respawn
        # into a double-consumer window and accumulated threads across reloads.
        self._paste_worker_alive = False
        if self._closed:
            # Normal shutdown path — not a fault; error level here produced
            # false "paste worker died" pages on every clean close (0921 audit
            # follow-up).
            log.info("paste worker thread exited (closed)")
        else:
            log.error("paste worker thread exited (stale generation); inline paste until main-thread restart")

    def _submit_paste(self, frame, face, bbox, alpha, pts) -> None:
        # Try the worker queue; if full (subprocess stalled / backlog), paste
        # inline on the main thread so the bounded paste_q never deadlocks
        # the render loop (audit P0-4 fallback).
        try:
            self._paste_q.put_nowait((frame, face, bbox, alpha, pts))
        except queue.Full:
            log.warning("paste queue full; inline paste on render thread")
            out = paste_back(
                frame, face, bbox, alpha=alpha if alpha is not None else None, mask_provider=self._mask
            )
            with self._emit_lock:
                self._last_frame = out
                self._emit(out, pts)

    def _maybe_restart_paste_worker(self) -> None:
        # Main-thread-only restart (audit 0921 P0-2): the worker never respawns
        # itself, so there is no double-worker window. One restart budget per
        # session (reset by reload drain) bounds a death-loop; beyond the
        # budget, inline paste keeps output alive at reduced fps.
        if self._paste_worker_restarted:
            return
        self._paste_worker_restarted = True
        log.warning("paste worker dead; main-thread restart (one budget, audit 0921 P0-2)")
        self._paste_worker_t = threading.Thread(target=self._paste_worker, name="musetalk-paste", daemon=True)
        self._paste_worker_t.start()
        self._paste_worker_alive = True

    def _paste_item(self, frame, face, bbox, pts) -> None:
        # Main-thread entry: fetch alpha (fills the parse cache on miss) and
        # hand the item to the worker. If the worker has died (audit P0-3
        # liveness), paste inline so output never stalls. A dead worker also
        # gets ONE main-thread restart attempt (audit 0921 P0-2) so a transient
        # MemoryError does not permanently halve fps (prior R1 intent), while
        # the budget keeps a death-loop bounded.
        alpha, _ = self._mask.mouth_mask(frame, bbox)
        if not self._paste_worker_alive and not self._closed:
            self._maybe_restart_paste_worker()
        if not self._paste_worker_alive or self._closed:
            out = paste_back(frame, face, bbox, alpha=alpha if alpha.size else None, mask_provider=self._mask)
            with self._emit_lock:
                self._last_frame = out
                self._emit(out, pts)
            return
        self._submit_paste(frame, face, bbox, alpha if alpha.size else None, pts)

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
                # Chunk i is frame (i - skip) of this window. PTS advances by
                # the true step size (sr//fps samples / sr), not 1/fps — the
                # 1/fps form drifts 0.67% (533/16000 != 1/30) and accumulates a
                # frame every ~150 steps (audit P1-30, FR-LK-001).
                self._pending.append((chunks[i], pts0 + (i - skip) * self.step / self.sr))
            log.debug("encoded window pts=%.3fs -> %d chunks (skipped %d prefix)", pts0, n - skip, skip)

    def _render(self, chunk, ladder=None) -> np.ndarray:
        pf = self.profiler
        pf.begin("frame_out")
        # Caller passes the already-read ladder (single thermal read per frame,
        # audit A4). Only read here when called outside get_output_frame (none
        # today, but keep a safe fallback rather than asserting).
        if ladder is None:
            ladder = self._thermal.ladder()
        if ladder["frame_reuse"] > 1:
            self._reuse = (self._reuse + 1) % ladder["frame_reuse"]
            if self._reuse and self._last_frame is not None:
                # Copy: a consumer that mutates the returned buffer in place
                # would otherwise pollute _last_frame and corrupt the next
                # reused frame (audit P1-34). Non-reuse frames share the alias
                # by contract — consumers must treat emitted frames read-only.
                return self._last_frame.copy()
        else:
            self._reuse = 0
        # step-count / ICB need fusion-mlx #911/#912; apply if the pipe exposes it.
        self._apply_ddim_steps(ladder["ddim_steps"])
        frame, bg_idx = self._bg_frame(downscale=ladder["bg_downscale"])
        # FR-END-003 strategy D (last resort): face patch 256 -> 128; croppers
        # cached per size so no 256->128 jump path skips the earlier rungs.
        patch = ladder["patch"]
        cached = None
        with self._bg_cache_lock:
            bg_cache = self._bg_cache
        if bg_cache and patch == 256 and bg_idx < len(bg_cache):
            cached = bg_cache[bg_idx]
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
            # #928: pipe.dtype is the public accessor (was pipe._dtype reach-in).
            if self.pipe.dtype is not None:
                latent = latent.astype(self.pipe.dtype)
            pf.end()
        pf.begin("unet")
        face = None
        gen = self._compiled_generate_128 if config.DECODE_128 else self._compiled_generate
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
                dtype = self.pipe.dtype
                img = gen(latent, mx.array(chunk[None]).astype(dtype))
                face = self._decode_faces(img)[0]
            except Exception as e:
                log.warning("compiled render failed (%s); plain path", e)
                self._compiled_generate = None
                self._compiled_generate_128 = None
                face = None
            pf.end()
        if face is None:
            pred = _unet_forward(self.pipe, latent, chunk)
            if config.DECODE_128 and latent.shape[-1] == config.LATENT:
                pred = _pool2x(pred)
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
        # #927 landed: fusion-mlx MuseTalkPipeline.set_ddim_steps exists and
        # sets the pipe's _ddim_steps state. NOTE: musetalk-mlx's realtime render
        # path (_unet_forward / compiled closures) calls pipe._run_unet with
        # steps=1 (single-step t=0) to hold the 33ms budget — multi-step DDIM
        # (15/8) is Nx slower and breaks 30FPS. set_ddim_steps is therefore
        # invoked here for observability + future offline multi-step paths, but
        # the realtime hot loop stays single-step. The serious-tier compute
        # lever for realtime is patch/bg-downscale/frame-reuse at critical,
        # not step-cut.
        setter = getattr(self.pipe, "set_ddim_steps", None)
        if setter is not None:
            setter(steps)
            log.debug("ddim steps set to %d", steps)
            return
        if not getattr(self, "_ddim_unavailable_logged", False):
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

    def _setup_graph_pass(self) -> None:
        # FR-MLX-003: consume the fusion-mlx #911/#918 graph passes when
        # available: structural rewrite (Conv+GN+SiLU fusion) + SmartConv2d
        # shape-dispatched conv (#919) on the module tree, then compile the
        # joint UNet+VAE-decode forward (PE applied inside, mirroring
        # generate_faces; generate_faces itself calls mx.eval, illegal under
        # mx.compile). One graph for unet->decode is ~2x faster than two
        # compiled calls (no compiled-input boundary).
        self._compiled_generate = None
        self._compiled_generate_128 = None
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
            if config.DECODE_128:

                def render_fn_128(latent, audio):
                    pred = pipe.unet(latent, mx.array([UNET_TIMESTEP]), apply_pe(audio))
                    return mx.clip(pipe.vae.decode(_pool2x(pred) / pipe.scaling_factor) / 2 + 0.5, 0, 1)

                self._compiled_generate_128 = compile_with_custom_pass(render_fn_128)
                log.info("DECODE_128 joint graph compiled (latent 2x2 avg-pool before decode)")
        except Exception as e:
            # Clear BOTH compiled entries — a partial assignment (128 set before
            # the main graph raised) would leave inconsistent state. Warning, not
            # info: operators must see that graph optimization is off (audit P2-4).
            self._compiled_generate = None
            self._compiled_generate_128 = None
            log.warning("graph pass unavailable (%s); plain generate_faces", e)

    def _preload_bg(self) -> None:
        self._bg_store.preload(self._tracker, self._cropper, self.pipe)

    def _precompute_cache(self, frames) -> None:
        self._bg_store.precompute_cache(frames, self._tracker, self._cropper, self.pipe)

    def _bg_frame(self, downscale: int = 1):
        return self._bg_store.get_frame(downscale)

    def _setup_graph_pass(self) -> None:
        # FR-MLX-003: consume the fusion-mlx #911/#918 graph passes when
        # available: structural rewrite (Conv+GN+SiLU fusion) + SmartConv2d
        # shape-dispatched conv (#919) on the module tree, then compile the
        # joint UNet+VAE-decode forward (PE applied inside, mirroring
        # generate_faces; generate_faces itself calls mx.eval, illegal under
        # mx.compile). One graph for unet->decode is ~2x faster than two
        # compiled calls (no compiled-input boundary).
        self._compiled_generate = None
        self._compiled_generate_128 = None
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
            if config.DECODE_128:

                def render_fn_128(latent, audio):
                    pred = pipe.unet(latent, mx.array([UNET_TIMESTEP]), apply_pe(audio))
                    return mx.clip(pipe.vae.decode(_pool2x(pred) / pipe.scaling_factor) / 2 + 0.5, 0, 1)

                self._compiled_generate_128 = compile_with_custom_pass(render_fn_128)
                log.info("DECODE_128 joint graph compiled (latent 2x2 avg-pool before decode)")
        except Exception as e:
            # Clear BOTH compiled entries — a partial assignment (128 set before
            # the main graph raised) would leave inconsistent state. Warning, not
            # info: operators must see that graph optimization is off (audit P2-4).
            self._compiled_generate = None
            self._compiled_generate_128 = None
            log.warning("graph pass unavailable (%s); plain generate_faces", e)
