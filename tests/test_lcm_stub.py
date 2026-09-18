import numpy as np

from musetalk_mlx import config
from musetalk_mlx.pipeline.lcm import LCMFastSession


class _Pipe:
    _dtype = None

    def generate_faces(self, latent, chunk):
        return np.zeros((1, 256, 256, 3), dtype=np.uint8)


def test_lcm_disabled_by_default():
    assert config.LCM_ENABLED is False
    lcm = LCMFastSession(_Pipe())
    assert lcm.enabled is False
    assert lcm.generate(None, np.zeros((50, 384), dtype=np.float32)) is None


def test_lcm_steps_constant():
    assert config.LCM_STEPS == 1
    lcm = LCMFastSession(_Pipe())
    assert lcm.steps == 1


def test_lcm_main_release_uses_ddim():
    # main release = multi-step DDIM, not LCM
    assert config.DDIM_STEPS > config.LCM_STEPS
