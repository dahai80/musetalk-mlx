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
