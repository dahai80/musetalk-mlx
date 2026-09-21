import logging
import threading
import time
from typing import NamedTuple

import cv2
import imageio
import mlx.core as mx
import numpy as np

from .. import config

log = logging.getLogger(__name__)


class BgCacheEntry(NamedTuple):
    # One precomputed bg frame's pipeline state (audit 0921 P2: was a bare
    # tuple indexed positionally `cached[2]` — NamedTuple keeps tuple
    # compatibility but makes field order self-documenting).
    landmarks: object
    bbox: object
    latent: object


class BackgroundStore:
    # FR-END-002: base-video state split out of MuseTalkSession (audit 0921
    # A-1): the imageio reader, the preloaded frame pool, the per-frame
    # precompute cache and the looped frame serving. Collaborators
    # (tracker/cropper/pipe) are passed per call, NOT captured — the session
    # swaps pipe on reload and the tracker in tests; a captured ref would
    # keep the old pipe alive across reload (breaks the 4GB budget) or miss
    # the substitution.

    def __init__(self, bg_video_path):
        self._bg = imageio.get_reader(str(bg_video_path))
        self._bg_n = int(self._bg.count_frames())
        if self._bg_n <= 0:
            # count_frames() can report 0 for some readers; a truly empty bg
            # video would ZeroDivision/IndexError on the modulo path. Validate
            # eagerly so the failure is attributable (audit P1-33).
            try:
                _ = next(iter(self._bg))
                self._bg_n = 1
                self._bg.seek(0)
            except Exception as e:
                raise ValueError(f"bg video {bg_video_path} has no readable frames ({e})")
        self._bg_idx = 0
        self._bg_pool = None  # FR-END-002: preloaded base-video frame pool
        self._bg_cache = []  # per-frame BgCacheEntry|None precompute
        # _bg_cache is written by precompute_cache (reload path, currently
        # synchronous) and read by the render paths. A lock keeps the
        # list-reference swap atomic so a future async rebuild cannot hand the
        # render thread a half-populated list (audit R8).
        self._bg_cache_lock = threading.Lock()

    def preload(self, tracker, cropper, pipe) -> None:
        # FR-END-002: preload the base-video frames into a memory pool (looped
        # serving from RAM, no per-frame decode cost). CAP at BG_POOL_MAX_FRAMES
        # — a long base video (10min = ~3.7GB at 1080p) would otherwise OOM the
        # process at startup. Beyond the cap, serve on-demand from the imageio
        # reader with a small LRU (audit A7).
        frames = []
        cap = config.BG_POOL_MAX_FRAMES
        budget_bytes = config.BG_POOL_BUDGET_MB * 1024 * 1024
        acc = 0
        try:
            for i, rgb in enumerate(self._bg):
                if i >= cap:
                    break
                f = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                acc += f.nbytes
                if acc > budget_bytes:
                    log.warning(
                        "bg pool hit byte budget %dMB at frame %d (%.0fMB); rest served on-demand (audit H4)",
                        config.BG_POOL_BUDGET_MB,
                        i,
                        acc / (1024 * 1024),
                    )
                    break
                frames.append(f)
        except Exception as e:
            # imageio v2 reader raises at stream end; keep collected frames.
            log.debug("bg preload stopped at %d frames (%s)", len(frames), e)
        self._bg_pool = frames
        self._bg_n = len(frames)
        self._bg_idx = 0
        mb = sum(f.nbytes for f in frames) / (1024 * 1024)
        if self._bg_n < cap:
            log.info("bg pool preloaded %d frames (pool %.0fMB)", len(frames), mb)
        else:
            # Pool capped — total bg length unknown without a second pass; the
            # on-demand path in get_frame handles frames beyond the pool.
            log.warning(
                "bg pool capped at %d frames (%.0fMB); longer bg served on-demand from reader",
                len(frames),
                mb,
            )
        if config.PRECOMPUTE and frames:
            self.precompute_cache(frames, tracker, cropper, pipe)

    def precompute_cache(self, frames, tracker, cropper, pipe) -> None:
        # MuseTalk-realtime-style offline pass: per base frame precompute
        # landmarks, crop bbox and VAE latent once, so the hot loop skips
        # DWPose (~100ms/frame) and VAE encode (~50ms/frame). Latents are
        # tiny ((1,8,32,32) fp16 ≈ 16KB/frame). Live fallback kept for
        # cache misses (idle frames, thermal patch 128).
        cache = []
        t0 = time.monotonic()
        for i, fr in enumerate(frames):
            lm = tracker.update(fr)
            if lm is None:
                cache.append(None)
                continue
            crop, bbox = cropper.crop(fr, lm)
            lat = pipe.get_latents_for_unet(crop)
            mx.eval(lat)
            cache.append(BgCacheEntry(lm, bbox, lat))
            if (i + 1) % 100 == 0:
                log.info("bg cache precompute %d/%d (%.1fs)", i + 1, len(frames), time.monotonic() - t0)
        # Atomic swap: the render thread reads _bg_cache; assigning the
        # fully-built list in one step under the lock means it never sees a
        # partial list (audit R8).
        with self._bg_cache_lock:
            self._bg_cache = cache
        tracker.fails = 0
        tracker.idle = False
        log.info(
            "bg cache precomputed %d/%d frames in %.1fs",
            sum(1 for c in cache if c is not None),
            len(cache),
            time.monotonic() - t0,
        )

    def get_frame(self, downscale: int = 1):
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

    def close(self) -> None:
        try:
            self._bg.close()
        except Exception as e:
            log.debug("bg reader close (%s)", e)
