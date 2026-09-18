import logging

from .. import config

log = logging.getLogger(__name__)

try:
    from Foundation import NSProcessInfo
except ImportError:
    NSProcessInfo = None


def thermal_state() -> int:
    """macOS thermal state: 0 nominal, 1 fair, 2 serious, 3 critical (else 0)."""
    if NSProcessInfo is None:
        return 0
    return int(NSProcessInfo.processInfo().thermalState())


def thermal_tier() -> int:
    """Map the OS thermal state to a PRD degradation tier (FR-END-003)."""
    s = thermal_state()
    if s >= 3:
        return config.THERMAL_CRITICAL
    if s >= 2:
        return config.THERMAL_SERIOUS
    return config.THERMAL_NORMAL


def ladder_for_tier(tier: int) -> dict:
    # FR-END-003 mandated order, never jump 256 -> 128 directly:
    # normal -> serious(step-cut) -> critical(frame-reuse -> bg-downscale -> patch-128).
    if tier >= config.THERMAL_CRITICAL:
        return {
            "ddim_steps": config.DDIM_STEPS_SERIOUS,
            "frame_reuse": config.FRAME_REUSE,
            "bg_downscale": config.BG_DOWNSCALE_CRITICAL,
            "patch": config.PATCH_CRITICAL,
        }
    if tier >= config.THERMAL_SERIOUS:
        return {
            "ddim_steps": config.DDIM_STEPS_SERIOUS,
            "frame_reuse": 1,
            "bg_downscale": 1,
            "patch": config.PATCH,
        }
    return {
        "ddim_steps": config.DDIM_STEPS,
        "frame_reuse": 1,
        "bg_downscale": 1,
        "patch": config.PATCH,
    }
