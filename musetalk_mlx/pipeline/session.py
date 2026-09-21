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
from ..pipeline.blending import paste_back
from ..pipeline.lcm import LCMFastSession
from ..pipeline.paste_proc import PasteProcess
from ..utils.audio import AudioWindower
from ..utils.profiling import StageProfiler
from ..utils.thermal import ThermalController
from .background import BackgroundStore, BgCacheEntry  # noqa: F401 -- re-export (tests import from session)
from .scheduler import RenderScheduler

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
        self._last_frame = None
        self._audio_prefix = None  # fusion-mlx #914: prior window's tail embedding
        self.profiler = StageProfiler()
        self._croppers = {}  # patch 256/128 (thermal ladder); croppers dict stays on session
        self._expected_pts = None  # observability: PTS sync-deviation logging
        self._tune_mlx_memory()
        self._setup_graph_pass()
        self._scheduler = self._make_scheduler()
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
        # Producer backpressure (RENDER_HIGH_WATER) is primary; drop-oldest
        # here is the last resort (audit P2-3).
        self._out_q_cap = config.RENDER_HIGH_WATER
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
        # Dedicated render thread (see config.RENDER_HIGH_WATER): decouples the
        # ~57ms b2 round from the 33ms publish pacing. Without it the pacer
        # consumer drives rendering one round per tick and caps publish FPS at
        # ~2/(round+period) — measured 17.4 fps vs 35 fps render capacity.
        self._render_lock = threading.Lock()  # held across next_step; interrupt()/reload() take it too
        self._render_ev = threading.Event()  # wake: new audio windows pending
        self._render_t = threading.Thread(target=self._render_loop, name="musetalk-render", daemon=True)
        self._render_t.start()
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
        # Fast path: same weights + same mlx-dir = state-only reset (no pipe
        # rebuild). A full from_pretrained reloads 3GB weights and recompiles
        # graphs (~10s stall + mlx_active doubles to 6GB, breaking the 4GB
        # budget). For same-version hot-reload (the A4 stress scenario and the
        # common "reset drift" operator action), only the runtime queues /
        # prefix / thermal state need resetting — the pipe and bg cache stay.
        # A weights change falls through to the full rebuild below.
        same = wd == self._weights_dir and md == self._mlx_dir
        if same and getattr(self, "pipe", None) is not None:
            log.info("ReloadModel: same weights; state-only reset (no pipe rebuild)")
            rl = getattr(self, "_render_lock", None)
            if rl is not None:
                rl.acquire()
            # Keep _pending: same-weight encoded chunks are still valid, so the
            # render thread resumes immediately instead of waiting 5s for the
            # windower to re-buffer a full hop (A4: resume 10s -> <1s). Only
            # inflight (lazy graphs tied to a compile generation) and emitted
            # frames are stale.
            self._inflight = deque()
            self._drain_paste_queue()
            with self._emit_lock:
                self._out_q.clear()
                self._expected_pts = None
            self._audio_prefix = None
            self._thermal = ThermalController()
            if rl is not None:
                rl.release()
            self._render_ev_set()
            return True
        log.info("ReloadModel: draining %d pending chunks, rebuilding pipe", len(self._pending))
        # Serialize against the render thread (producer) so no round is
        # mid-flight against the old pipe while the swap happens.
        rl = getattr(self, "_render_lock", None)
        if rl is not None:
            rl.acquire()
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
            if rl is not None:
                rl.release()
            return False
        # Peak-memory guard (audit 0921 P1-1): swap FIRST, overwrite every
        # other strong ref to the old pipe (lcm), then drop it + flush the MLX
        # allocator cache BEFORE _precompute_cache — otherwise old-pipe weights
        # (~1-2GB) stay alive alongside the new pipe while precompute encodes
        # every bg frame, breaking the 4GB budget.
        old, self.pipe = self.pipe, new_pipe
        self.lcm = LCMFastSession(new_pipe) if config.LCM_ENABLED else None
        # Drop compiled closures BEFORE clearing the old pipe ref: a compiled
        # graph captures the old pipe in its closure, so without this the old
        # 1-2GB weights stayed alive alongside the new pipe (A4 finding:
        # mlx_active 2006->5416MB across 3 reloads, breaking the 4GB budget).
        # _set_render_cache rebuilds them lazily on the next render.
        self._clear_compiled_gens()
        self._setup_graph_pass()
        old = None  # noqa: F841 - drop the last strong ref before clear_cache
        mx.clear_cache()
        if config.FP16 and hasattr(self.pipe, "astype"):
            new_pipe.astype(mx.float16)
        # BG cache: the old cache's landmarks/bbox (DWPose, weight-independent)
        # and latents (sd-vae-ft-mse, unchanged across same-version reloads)
        # stay valid — astype(pipe.dtype) in _render handles a dtype shift. Keep
        # the old cache so the render thread resumes immediately (<2s, A4
        # requirement) instead of waiting 24s for synchronous precompute. A
        # background thread rebuilds against the new pipe and atomically swaps
        # when done; until then the live path handles any None entries.
        pool = getattr(self, "_bg_pool", None)
        if pool is not None:
            self._bg_rebuild_thread = threading.Thread(
                target=self._bg_store.precompute_cache,
                args=(pool, self._tracker, self._cropper, self.pipe),
                name="bg-reload",
                daemon=True,
            )
            self._bg_rebuild_thread.start()
        else:
            with self._bg_cache_lock:
                self._bg_cache = []
        self._set_render_cache()
        self._weights_dir = wd
        self._mlx_dir = md
        log.info("ReloadModel done")
        if rl is not None:
            rl.release()
        return True

    def push_audio(self, pcm: np.ndarray) -> None:
        # Validate dtype/range so garbage PCM fails loudly, not silently into
        # Whisper (audit P2-5). NaN/Inf sanitized inside AudioWindower.push.
        arr = np.asarray(pcm)
        if arr.dtype != np.float32:
            log.warning("push_audio got %s, casting to float32", arr.dtype)
        self._windower.push(pcm)
        self._render_ev_set()

    def _render_loop(self) -> None:
        # Dedicated producer thread: pumps RenderScheduler.next_step while
        # audio windows are pending, up to RENDER_HIGH_WATER out_q frames.
        # get_output_frame becomes a pure consumer (pop out_q, no rendering).
        # The render lock is held across next_step so interrupt()/reload()
        # serialize against mid-round teardown; it is uncontended in steady
        # state (one producer, uncontended Lock ~ns).
        log.info("render thread started")
        while not self._closed:
            self._encode_windows()
            if not self._pending and not self._inflight:
                if self._out_q:
                    # Consumer still draining; producer idles WITHOUT clearing
                    # the event first (a push between check and clear must not
                    # be lost — standard clear-after-check wake pattern).
                    time.sleep(0.01)
                    continue
                self._render_ev.clear()
                if self._pending or self._inflight or self._out_q:
                    continue
                self._render_ev.wait(timeout=0.5)
                continue
            if len(self._out_q) >= config.RENDER_HIGH_WATER:
                # Backpressure: consumer behind, producer pauses. The queue
                # drains at 30fps while the producer pauses; catching up
                # unbounded would grow RTT (buffered frames = latency).
                time.sleep(0.005)
                self._render_ev.clear()
                self._render_ev.wait(0.05)
                continue
            try:
                with self._render_lock:
                    ladder = self._thermal.ladder()
                    # pop=False: frames stay in out_q for the consumer.
                    self._scheduler.next_step(ladder, pop=False)
                self._render_ev.set()  # re-check pending: more windows may have arrived
            except Exception:
                # A render failure must not kill the producer thread (Rule 12):
                # log loudly and idle briefly; next push re-seeds the pipeline.
                log.exception("render thread render-loop iteration failed")
                self._render_ev.clear()
                self._render_ev.wait(0.2)

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
        # Stop the render producer BEFORE the paste worker: it must not submit
        # new rounds (or paste items) while teardown drains the queues.
        rt = getattr(self, "_render_t", None)
        if rt is not None and rt.is_alive():
            self._render_ev.set()  # wake out of the idle wait to see _closed
            rt.join(timeout=2.0)
            if rt.is_alive():
                log.warning("render thread did not stop within 2s; leaving daemon")
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

    @property
    def idle(self) -> bool:
        # True when the pipeline holds no recoverable work. Consumer-side
        # (benchmark drain / host apps): out_q empty, nothing pending, no
        # in-flight round, and the paste worker has nothing queued. The paste
        # worker may still be pasting the last items — the _wait_out_q grace in
        # get_output_frame covers that residual window.
        return not self._pending and not self._inflight and not self._out_q and self._paste_q.empty()

    def end_of_stream(self) -> None:
        # Offline boundary: zero-pad the sub-window audio tail so the final
        # <5s renders instead of being silently dropped. No-op mid-stream.
        if self._windower.flush():
            self._encode_windows()
        self._render_ev_set()

    def _render_ev_set(self):
        # Wake helper: shells may lack the event (render-thread sessions only).
        ev = getattr(self, "_render_ev", None)
        if ev is not None:
            ev.set()

    def interrupt(self) -> int:
        # Barge-in (audit 0921 P3 — K12 conversation core requirement): drop
        # every not-yet-emitted artifact of the current utterance and return to
        # standby. Explicit API only — VAD-based auto-trigger is a future hook.
        # Returns the number of queued audio chunks (not frames) discarded, for
        # observability.
        if self._closed:
            return 0
        dropped = len(self._pending)
        # Serialize against the render thread mid-round: it may hold items it
        # popped from _pending inside next_step; without the lock a round could
        # re-emit frames AFTER the interrupt (stale utterance frames).
        # Serialize against the render thread mid-round: it may hold items it
        # popped from _pending inside next_step; without the lock a round could
        # re-emit frames AFTER the interrupt (stale utterance frames). Shells
        # built via __new__ have no render thread/lock — clear without locking.
        rl = getattr(self, "_render_lock", None)
        if rl is None:
            self._pending.clear()
            # In-flight rounds hold LAZY mx graphs (submitted but not eval'd).
            # Dropping the ref cancels the never-started GPU work silently — no
            # sync needed, nothing was materialized yet.
            inflight = len(self._inflight)
            self._inflight.clear()
        else:
            with rl:
                self._pending.clear()
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
        # Consumer-only once the render thread runs (see _render_loop): pops
        # out_q; never renders inline. Shells built via __new__ (unit tests)
        # and threadless sessions keep the inline path so their private
        # attributes and call patterns keep working unchanged.
        if self._out_q:
            return self._out_q.popleft()
        if getattr(self, "_render_t", None) is None:
            # Threadless session/shell: render inline (pre-thread behavior).
            self._encode_windows()
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
            return self._scheduler.next_step(ladder)
        # Wait while the producer is actively working (bounded): a single call
        # still returns a frame when work is pending (host apps + tests), but
        # with a backlog out_q pops are instant — the 33ms pacer is no longer
        # the one driving the 57ms render rounds (A3 finding 2026-09-21).
        deadline = time.monotonic() + 2.0
        while not self._out_q:
            if not self._pending and not self._inflight and not self._windower.has_window():
                # Producer idle (audio gap / ended): nothing will arrive.
                break
            if time.monotonic() >= deadline:
                rt = getattr(self, "_render_t", None)
                log.warning(
                    "get_output_frame: producer busy but no frame in 2s "
                    "(pending=%d inflight=%d out_q=%d window=%s thread_alive=%s)",
                    len(self._pending),
                    len(self._inflight),
                    len(self._out_q),
                    self._windower.has_window(),
                    rt is not None and rt.is_alive(),
                )
                break
            time.sleep(0.002)
        return self._out_q.popleft() if self._out_q else None

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
        # Inline paste on the render thread (A3 finding 2026-09-21): paste_back
        # with a cached alpha is ~0.2-0.5ms, so routing through the paste
        # worker (thread paste under GIL contention, or the mp 12MB pickle IPC
        # round trip) was pure overhead — next_step p95 hit 305ms waiting for
        # the worker to emit while the render thread sat idle. The full
        # mouth_mask (parse on miss) runs here on the render thread, which is
        # also the MLX thread — single-threaded parse, no contract break.
        # Worker/mp machinery stays for interrupt/drain paths (tests) but the
        # hot path never queues.
        alpha, _ = self._mask.mouth_mask(frame, bbox)
        out = paste_back(frame, face, bbox, alpha=alpha if alpha.size else None)
        with self._emit_lock:
            self._last_frame = out
            self._emit(out, pts)

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

    def _make_scheduler(self):
        # Render scheduling split (audit 0921 A-1 step 4b): injected context,
        # pipe/tracker via getters so reload/test substitution stays live.
        return RenderScheduler(
            pending=self._pending,
            out_q=self._out_q,
            pipe_getter=lambda: self.pipe,
            gens_getter=self._gens_getter,
            clear_gens=self._clear_compiled_gens,
            ladder_provider=lambda: self._thermal.ladder(),
            bg_frame=self._bg_frame,
            bg_cache_snapshot=self._bg_cache_snapshot,
            emit=self._emit,
            paste_item=self._paste_item,
            mask=self._mask,
            profiler=self.profiler,
            cropper=self._cropper,
            croppers=self._croppers,
            tracker_getter=lambda: self._tracker,
            last_frame_getter=lambda: self._last_frame,
            last_frame_setter=lambda f: setattr(self, "_last_frame", f),
        )

    def _gens_getter(self):
        return self._compiled_generate_128 if config.DECODE_128 else self._compiled_generate

    def _clear_compiled_gens(self) -> None:
        # Clear BOTH compiled entries — a partial state (main graph raised
        # before the 128 variant) would leave inconsistent state (audit P2-4).
        self._compiled_generate = None
        self._compiled_generate_128 = None

    def _wait_out_q(self, timeout_s: float = 0.2):
        # Thin delegate: the wait loop lives on the scheduler (owns the round
        # machinery) but idle callers on the session still hit this name.
        return self._scheduler._wait_out_q(timeout_s)

    def _bg_cache_snapshot(self):
        # Snapshot the cache list under the lock so a concurrent reload
        # swap cannot detach the reference mid-iteration (audit R8).
        with self._bg_cache_lock:
            return self._bg_cache

    def _require_scheduler(self):
        # Lazily bind a scheduler shell so reload()/barge-in unit-test shells
        # built via __new__ keep setting _inflight directly (compat shim).
        st = self.__dict__.get("_scheduler")
        if st is None:
            st = RenderScheduler.__new__(RenderScheduler)
            self._scheduler = st
        return st

    @property
    def _inflight(self):
        return self._require_scheduler()._inflight

    @_inflight.setter
    def _inflight(self, v):
        self._require_scheduler()._inflight = v

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
