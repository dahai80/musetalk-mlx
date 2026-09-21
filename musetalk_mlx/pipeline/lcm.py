import logging

import mlx.core as mx

from .. import config

log = logging.getLogger(__name__)


class LCMFastSession:
    # Phase 4 stub (non-blocking). The fusion-mlx MuseTalkPipeline.generate_faces
    # is inherently single-step (t=0 UNet inpaint) — there is no multi-step DDIM
    # loop in the MLX port, and no distilled 1-step weights API to switch to.
    # Therefore LCM_STEPS has no pipe-side effect: the "fast path" is already the
    # only path. This stub exists for config/flag wiring parity with the PRD; it
    # does NOT add a second generation mechanism.
    # Disabled by default. If LCM_ENABLED is set True, we validate the pipe
    # rather than silently pretending to run a distinct path (audit P0-8/P1-22).

    def __init__(self, pipe, steps: int = config.LCM_STEPS):
        self.pipe = pipe
        self.steps = steps
        if not config.LCM_ENABLED:
            log.info("LCM fast path disabled (pipe is single-step t=0; main release uses it directly)")

    @property
    def enabled(self) -> bool:
        return config.LCM_ENABLED

    def generate(self, latent, audio_chunk):
        if not self.enabled:
            return None
        if self.pipe is None:
            raise RuntimeError("LCM enabled but pipe is None")
        if audio_chunk is None:
            raise ValueError("LCM generate received None audio_chunk")
        dtype = self.pipe.dtype  # #928 public accessor
        audio_mx = mx.array(audio_chunk[None]).astype(dtype)
        # NOTE: generate_faces is single-step t=0 in fusion-mlx; steps is a no-op
        # here until a distilled-weights API lands upstream. Logged, not asserted,
        # so a future pipe that ignores steps still works.
        log.debug("LCM generate (single-step pipe, configured steps=%d unused)", self.steps)
        face = self.pipe.generate_faces(latent, audio_mx)[0]
        return face
