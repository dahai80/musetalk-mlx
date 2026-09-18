from musetalk_mlx import config
from musetalk_mlx.utils.thermal import ladder_for_tier


def test_normal_ladder():
    g = ladder_for_tier(config.THERMAL_NORMAL)
    assert g["ddim_steps"] == config.DDIM_STEPS
    assert g["frame_reuse"] == 1
    assert g["bg_downscale"] == 1
    assert g["patch"] == config.PATCH


def test_serious_step_cut_only():
    g = ladder_for_tier(config.THERMAL_SERIOUS)
    assert g["ddim_steps"] == config.DDIM_STEPS_SERIOUS
    assert g["frame_reuse"] == 1  # no frame reuse at serious
    assert g["patch"] == config.PATCH  # patch stays 256


def test_critical_full_ladder():
    g = ladder_for_tier(config.THERMAL_CRITICAL)
    assert g["ddim_steps"] == config.DDIM_STEPS_SERIOUS
    assert g["frame_reuse"] == config.FRAME_REUSE
    assert g["bg_downscale"] == config.BG_DOWNSCALE_CRITICAL
    assert g["patch"] == config.PATCH_CRITICAL


def test_no_256_to_128_jump():
    # serious must never drop patch to 128; only critical does, and only after
    # frame-reuse + bg-downscale are already active.
    s = ladder_for_tier(config.THERMAL_SERIOUS)
    c = ladder_for_tier(config.THERMAL_CRITICAL)
    assert s["patch"] == config.PATCH
    assert c["patch"] == config.PATCH_CRITICAL
    assert c["frame_reuse"] > 1 and c["bg_downscale"] > 1  # precede patch drop


def test_order_monotone():
    tiers = [config.THERMAL_NORMAL, config.THERMAL_SERIOUS, config.THERMAL_CRITICAL]
    cfgs = [ladder_for_tier(t) for t in tiers]
    # patch size non-increasing, reuse non-decreasing
    patches = [c["patch"] for c in cfgs]
    assert patches == sorted(patches, reverse=True)
    reuse = [c["frame_reuse"] for c in cfgs]
    assert reuse == sorted(reuse)
