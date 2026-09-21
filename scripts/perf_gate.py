#!/usr/bin/env python3
"""Performance gate: pacing (FPS + interval p95) + RTT (stable-state p95).

Local pre-release gate — needs real weights + clean GPU + a running LiveKit
server, so it is NOT wired into CI. CI exercises the gate LOGIC on synthetic
reports (see .github/workflows/ci.yml). Real gate runs are a RELEASE_CHECKLIST
step.

Usage:
    python scripts/perf_gate.py results/realtime_pacing.json results/realtime_rtt.json

Thresholds (env-overridable):
    MT_GATE_FPS         default 30.0
    MT_GATE_INTERVAL_P95 default 33.4   # ms (one 30fps frame + jitter)
    MT_GATE_RTT_P95     default 80.0    # ms (PRD <=80ms, steady-state)
"""

import json
import os
import sys
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _gate_pacing(path: Path) -> tuple[bool, str]:
    d = json.loads(path.read_text())
    fps = d.get("fps", 0)
    p95 = d.get("interval_p95_ms", 1e9)
    thr_fps = _f("MT_GATE_FPS", 30.0)
    thr_p95 = _f("MT_GATE_INTERVAL_P95", 33.4)
    ok = fps >= thr_fps and p95 <= thr_p95
    return ok, f"pacing: fps={fps:.2f} (>= {thr_fps}), interval_p95={p95:.1f}ms (<= {thr_p95})"


def _gate_rtt(path: Path) -> tuple[bool, str]:
    d = json.loads(path.read_text())
    thr = _f("MT_GATE_RTT_P95", 80.0)
    # New dual-scope format (release-audit P0-2).
    if "stable_rtt" in d:
        stable = d["stable_rtt"]
        p95 = stable.get("p95_ms", 1e9)
        ok = p95 <= thr
        return ok, f"rtt: stable_p95={p95:.1f}ms (<= {thr}) [{stable.get('scope', '')}]"
    # Legacy single-metric: fails the gate (conflated cold-start + steady).
    return False, (
        f"rtt: legacy single-metric report (rtt_p50={d.get('rtt_p50_ms')}); "
        "rerun with the dual-scope probe (stable_rtt/first_packet_rtt)"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: perf_gate.py <pacing.json> [rtt.json]", file=sys.stderr)
        return 2
    ok_all = True
    msgs = []
    pacing = Path(sys.argv[1])
    if not pacing.exists():
        print(f"FAIL: pacing report {pacing} not found", file=sys.stderr)
        return 2
    ok, msg = _gate_pacing(pacing)
    msgs.append(msg)
    ok_all = ok_all and ok
    if len(sys.argv) >= 3:
        rtt = Path(sys.argv[2])
        if rtt.exists():
            ok, msg = _gate_rtt(rtt)
            msgs.append(msg)
            ok_all = ok_all and ok
        else:
            print(f"WARN: rtt report {rtt} not found; skipping RTT gate", file=sys.stderr)
    status = "PASS" if ok_all else "FAIL"
    print(f"{status}:")
    for m in msgs:
        print(f"  {m}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
