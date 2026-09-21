import logging
from collections import deque

import mlx.core as mx
import numpy as np
import pytest

from musetalk_mlx import config
from musetalk_mlx.pipeline.background import BgCacheEntry
from musetalk_mlx.pipeline.scheduler import RenderScheduler, _pool2x

log = logging.getLogger(__name__)


class _FakePipe:
    dtype = mx.float32

    def _run_unet(self, latent, audio, steps=1):
        return latent

    def decode_latents(self, pred):
        return np.zeros((1, 256, 256, 3), dtype=np.float32)


class _NoopProfiler:
    def begin(self, stage):
        pass

    def end(self):
        pass

    def tick(self):
        pass


def _mk(monkeypatch, pending, *, bg_cache=None, gen=None, batch=2):
    pending = deque(pending)
    monkeypatch.setattr(config, "BATCH", batch)
    box = {"pastes": [], "clears": 0, "last_frame": None, "ladder_reads": 0}
    emitted = deque()

    def bg_frame(downscale=1):
        return (np.zeros((4, 4, 3), dtype=np.uint8), 0)

    def provider():
        box["ladder_reads"] += 1
        return {"frame_reuse": 1, "patch": 256, "bg_downscale": 1, "ddim_steps": 15}

    sched = RenderScheduler(
        pending=pending,
        out_q=emitted,
        pipe_getter=lambda: _FakePipe(),
        gens_getter=lambda: gen,
        clear_gens=lambda: box.__setitem__("clears", box["clears"] + 1),
        ladder_provider=provider,
        bg_frame=bg_frame,
        bg_cache_snapshot=lambda: bg_cache,
        emit=lambda f, p: emitted.append((f, p)),
        paste_item=lambda f, face, bbox, pts: (box["pastes"].append(pts), emitted.append((f, pts))),
        mask=None,
        profiler=_NoopProfiler(),
        cropper=type("C", (), {"upperbondrange": 0})(),
        croppers={},
        tracker_getter=lambda: (_ for _ in ()).throw(AssertionError("tracker must not run on cache hit")),
        last_frame_getter=lambda: box["last_frame"],
        last_frame_setter=lambda f: box.__setitem__("last_frame", f),
    )
    return sched, box


def test_pool2x_shapes():
    z = mx.zeros((1, 8, 32, 32))
    assert _pool2x(z).shape == (1, 8, 16, 16)
    with pytest.raises(ValueError):
        _pool2x(mx.zeros((1, 8, 31, 32)))


def test_render_ahead_depth2_batched(monkeypatch):
    gen_calls = []

    def gen(lat, ch):
        gen_calls.append((lat.shape, ch.shape))
        return mx.zeros((lat.shape[0], 3, 256, 256))

    cache = [BgCacheEntry(object(), (0, 0, 8, 8), mx.zeros((1, 8, 32, 32)))] * 4
    pending = [(np.zeros((50, 384), dtype=np.float32), i / 30.0) for i in range(4)]
    sched, box = _mk(monkeypatch, pending, bg_cache=cache, gen=gen)
    out = sched.next_step({"frame_reuse": 1, "patch": 256, "bg_downscale": 1, "ddim_steps": 15})
    assert out is not None
    assert len(box["pastes"]) == 2
    assert len(sched._inflight) == 1
    assert len(sched._pending) == 0
    assert len(gen_calls) == 2  # two rounds, each ONE joint unet+decode call
    assert gen_calls[0][0][0] == 2  # batch dim: 2 latents concatenated per round


def test_cache_miss_requeues(monkeypatch):
    pending = [(np.zeros((50, 384), dtype=np.float32), i / 30.0) for i in range(2)]
    sched, box = _mk(monkeypatch, pending, bg_cache=[], gen=None)
    ok = sched._submit_round()
    assert ok is False
    assert len(sched._pending) == 2
    assert sched._pending[0][1] == 0.0
    assert box["clears"] == 0


def test_compiled_failure_clears_gens(monkeypatch):
    def bad_gen(lat, ch):
        raise RuntimeError("boom")

    cache = [BgCacheEntry(object(), (0, 0, 8, 8), mx.zeros((1, 8, 32, 32)))] * 2
    pending = [(np.zeros((50, 384), dtype=np.float32), i / 30.0) for i in range(2)]
    sched, box = _mk(monkeypatch, pending, bg_cache=cache, gen=bad_gen)
    monkeypatch.setattr(sched, "_render_all_singly", lambda ladder: None)
    ok = sched._submit_round()
    assert ok is False
    assert box["clears"] == 1
    assert len(sched._pending) == 2


def test_ladder_passed_not_reread(monkeypatch):
    def gen(lat, ch):
        return mx.zeros((1, 3, 256, 256))

    cache = [BgCacheEntry(object(), (0, 0, 64, 64), mx.zeros((1, 8, 32, 32)))]
    pending = [(np.zeros((50, 384), dtype=np.float32), 0.0)]
    sched, box = _mk(monkeypatch, pending, bg_cache=cache, gen=gen)
    chunk = pending.pop(0)[0]
    sched._render(chunk, {"frame_reuse": 1, "patch": 256, "bg_downscale": 1, "ddim_steps": 8})
    assert box["ladder_reads"] == 0
    assert box["last_frame"] is not None
