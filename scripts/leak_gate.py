#!/usr/bin/env python3
"""Leak-budget gate: exit 0 iff the stress report passes (leak <= 50MB).

Local pre-release gate — stress runs need real weights + clean GPU, so this is
NOT wired into CI. Usage:

    musetalk-mlx-stress ... --report results/stress_report.json
    python scripts/leak_gate.py results/stress_report.json
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: leak_gate.py <stress_report.json>", file=sys.stderr)
        return 2
    report = Path(sys.argv[1])
    if not report.exists():
        print(f"FAIL: report {report} not found", file=sys.stderr)
        return 2
    data = json.loads(report.read_text())
    leak_mb = data.get("leak_mb")
    passed = data.get("pass")
    if passed is None or leak_mb is None:
        print("FAIL: report missing 'pass' or 'leak_mb' fields", file=sys.stderr)
        return 2
    budget = data.get("leak_budget_mb", 50.0)
    ok = bool(passed) and float(leak_mb) <= float(budget)
    status = "PASS" if ok else "FAIL"
    print(f"{status}: leak={leak_mb}MB budget={budget}MB pass={passed}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
