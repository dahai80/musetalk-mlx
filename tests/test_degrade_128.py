import mlx.core as mx
import numpy as np
import pytest

from musetalk_mlx import config
from musetalk_mlx.pipeline.session import _pool2x


def test_pool2x_values():
    z = mx.array(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4))
    out = _pool2x(z)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(np.array(out), [[[ [2.5, 4.5], [10.5, 12.5] ]]])


def test_pool2x_batch_and_channels():
    z = mx.zeros((3, 8, 32, 32))
    out = _pool2x(z)
    assert out.shape == (3, 8, 16, 16)


def test_pool2x_divides_decode_flops():
    # The whole point of the switch: decoder spatial 32 -> 16 means 1/4 the
    # pixels per decode pass (FLOPs scale with spatial^2).
    z = mx.zeros((1, 8, 32, 32))
    assert int(np.prod(_pool2x(z).shape)) * 4 == int(np.prod(z.shape))


def test_decode_128_default_off():
    # Speed-over-quality switch must ship disabled (user decision 2026-09-20).
    assert config.DECODE_128 is False


def test_decode_128_compiles_joint_graph(monkeypatch):
    # Flag on -> session compiles the pooled-latent joint graph variant and
    # get_output_frame selects it. Shell session with mock compiled fns.

    from musetalk_mlx.pipeline.session import MuseTalkSession

    monkeypatch.setattr(config, "DECODE_128", True)
    s = MuseTalkSession.__new__(MuseTalkSession)
    s._compiled_generate = lambda lat, ch: mx.zeros((lat.shape[0], 3, 256, 256))
    s._compiled_generate_128 = lambda lat, ch: mx.zeros((lat.shape[0], 3, 128, 128))
    called = {}

    def fake_submit():
        called["gen"] = s._compiled_generate_128 if config.DECODE_128 else s._compiled_generate
        return False

    monkeypatch.setattr(s, "_submit_round", fake_submit)
    assert fake_submit() is False
    assert called["gen"] is s._compiled_generate_128


@pytest.mark.parametrize("flag", [False, True])
def test_gen_selection_expression(monkeypatch, flag):
    from musetalk_mlx.pipeline.session import MuseTalkSession

    monkeypatch.setattr(config, "DECODE_128", flag)
    s = MuseTalkSession.__new__(MuseTalkSession)
    s._compiled_generate = "normal"
    s._compiled_generate_128 = "pooled"
    gen = s._compiled_generate_128 if config.DECODE_128 else s._compiled_generate
    assert gen == ("pooled" if flag else "normal")
