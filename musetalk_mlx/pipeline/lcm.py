import logging

import mlx.core as mx

from .. import config

log = logging.getLogger(__name__)


class LCMFastSession:
    # Phase 4 stub (non-blocking): when LCM_ENABLED and 1-step weights are
    # distilled+converted (separate task; main release uses multi-step DDIM),
    # the session delegates generation here with LCM_STEPS. No distillation
    # training in this repo. Disabled by default.

    def __init__(self, pipe, steps: int = config.LCM_STEPS):
        self.pipe = pipe
        self.steps = steps
        if not config.LCM_ENABLED:
            log.info("LCM fast path disabled (main release = DDIM %d steps)", config.DDIM_STEPS)

    @property
    def enabled(self) -> bool:
        return config.LCM_ENABLED

    def generate(self, latent, audio_chunk):
        if not self.enabled:
            return None
        dtype = getattr(self.pipe, "_dtype", mx.float32)
        face = self.pipe.generate_faces(latent, mx.array(audio_chunk[None]).astype(dtype))[0]
        log.debug("LCM generate steps=%d", self.steps)
        return face
