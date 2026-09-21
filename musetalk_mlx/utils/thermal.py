import logging
from typing import TypedDict

from .. import config

log = logging.getLogger(__name__)

try:
    from Foundation import NSProcessInfo
except ImportError:
    NSProcessInfo = None

# One-shot startup warning: without NSProcessInfo (PyObjC missing) thermal_state
# always returns 0 (nominal) — the session runs normal-tier only and will NOT
# degrade under thermal throttle, risking sustained frame drops on a hot machine.
# Operators must see this once rather than discover it via a thermal event (audit E6).
if NSProcessInfo is None:
    log.warning(
        "NSProcessInfo unavailable (PyObjC not installed); thermal monitoring is "
        "OFF — session runs normal-tier only, no degradation under throttle. "
        "Install PyObjC for FR-END-003 thermal ladder."
    )


def thermal_state() -> int:
    """macOS thermal state: 0 nominal, 1 fair, 2 serious, 3 critical (else 0)."""
    if NSProcessInfo is None:
        return 0
    return int(NSProcessInfo.processInfo().thermalState())


def thermal_tier() -> int:
    """Map the OS thermal state to a PRD degradation tier (FR-END-003).

    Stateless convenience wrapper. Callers that need hysteresis (no tier
    chatter at the fair/serious boundary) should hold a ThermalController."""
    s = thermal_state()
    if s >= 3:
        return config.THERMAL_CRITICAL
    if s >= 2:
        return config.THERMAL_SERIOUS
    return config.THERMAL_NORMAL


class Ladder(TypedDict):
    # Per-tier degradation knobs (audit 0921 P2: was a bare dict, key typos
    # would surface as KeyError deep in the render path).
    ddim_steps: int
    frame_reuse: int
    bg_downscale: int
    patch: int


def ladder_for_tier(tier: int) -> "Ladder":
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


class ThermalController:
    # Stateful thermal ladder with hysteresis (audit P1-8/P1-9). The OS
    # thermalState chatters 1<->2 every few seconds on M5 Max under load; a
    # stateless mapping made ddim_steps flap 15<->8 and broke render cadence.
    # Promotion needs PROMOTE_CONFIRM consecutive readings; demotion needs
    # DEMOTE_CONFIRM (fewer) so recovery is prompt but stable. Downgrades step
    # one tier at a time — never jump normal -> critical directly.
    PROMOTE_CONFIRM = 4
    DEMOTE_CONFIRM = 2
    MIN_HOLD_STEPS = 6

    def __init__(self):
        self._tier = config.THERMAL_NORMAL
        self._pending = config.THERMAL_NORMAL
        self._confirm = 0
        self._hold = 0
        self._last_state = -1

    @property
    def tier(self) -> int:
        return self._tier

    def step(self) -> int:
        raw = thermal_tier()
        state = thermal_state()
        if state != self._last_state:
            # Observable fair entry (audit P4-5): log every state transition.
            log.info("thermal state %d -> tier %d (current held tier %d)", state, raw, self._tier)
            self._last_state = state
        if raw > self._tier:
            target = min(raw, self._tier + 1)  # step one tier at a time
            if self._pending == target:
                self._confirm += 1
            else:
                self._pending = target
                self._confirm = 1
            if self._confirm >= self.PROMOTE_CONFIRM and self._hold >= self.MIN_HOLD_STEPS:
                log.warning("thermal tier upgrade %d -> %d", self._tier, target)
                self._tier = target
                self._pending = target
                self._confirm = 0
                self._hold = 0
        elif raw < self._tier:
            if self._pending == raw:
                self._confirm += 1
            else:
                self._pending = raw
                self._confirm = 1
            if self._confirm >= self.DEMOTE_CONFIRM:
                log.info("thermal tier downgrade %d -> %d", self._tier, raw)
                self._tier = raw
                self._pending = raw
                self._confirm = 0
                self._hold = 0
        else:
            self._confirm = 0
            self._pending = self._tier
        self._hold += 1
        return self._tier

    def ladder(self) -> "Ladder":
        return ladder_for_tier(self.step())
