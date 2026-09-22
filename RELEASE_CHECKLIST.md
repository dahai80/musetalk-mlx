# Release Checklist (local pre-release gates)

These gates need real weights + a clean GPU + (for RTT) a running LiveKit
server, so they are NOT wired into CI. CI only exercises the gate *logic* on
synthetic reports. Run these locally before any commercial release and attach
the JSON artifacts.

## Clean-GPU protocol (required for all perf numbers)

Background GPU load (other sessions, the linguakids watchdog) inflates numbers
several-fold. Before measuring:

```bash
/path/to/fusion-mlx/start.sh stop
launchctl bootout gui/501/com.linguakids.watchdog 2>/dev/null || true
ioreg -r -k "Device Utilization" -d 1 | grep "Device Utilization"  # expect ≈0
```

Start a fresh process for each run. Restore after: `/path/to/fusion-mlx/start.sh start`.

## 1. Leak gate (2h stress)

```bash
musetalk-mlx-stress --weights weights --mlx-dir weights-mlx --video <base.mp4> \
    --audio <audio.wav> --duration 120 --with-reload \
    --report results/stress_2h.json
python scripts/leak_gate.py results/stress_2h.json
```

Pass: `leak_mb <= 50`, `pass=true`, fragmentation probe OK, phys footprint
reported per-minute. Attach the JSON.

## 2. Perf gate (FPS + interval p95)

```bash
musetalk-mlx-benchmark --weights weights --mlx-dir weights-mlx --video <base.mp4> \
    --report results/benchmark.json
# or the realtime pacing report from a LiveKit E2E run:
#   results/realtime_pacing.json
python scripts/perf_gate.py results/realtime_pacing.json
```

Pass: `fps >= 30`, `interval_p95_ms <= 33.4`. Override thresholds via
`MT_GATE_FPS` / `MT_GATE_INTERVAL_P95` if shipping a degraded-quality tier
(e.g. `MT_DECODE_128=1`).

## 3. RTT gate (steady-state, dual-scope)

```bash
musetalk-mlx-realtime --weights weights --mlx-dir weights-mlx --video <base.mp4> \
    --audio <audio.wav> --livekit-url wss://... --token <jwt> \
    --rtt-report results/realtime_rtt.json
python scripts/perf_gate.py results/realtime_pacing.json results/realtime_rtt.json
```

Pass: `stable_rtt.p95_ms <= 80`. The `first_packet_rtt` scope (includes the 5s
AudioWindower fill) is reported honestly but is NOT the PRD <=80ms target —
disclose it to the product side separately.

## 4. Weight integrity + authenticity

```bash
# After convert or download, generate + sign the manifest:
musetalk-mlx-sign --weights weights-mlx --priv-key <key.pem> --pub-key <key.pub.pem>
# Load-time verify runs automatically (MT_WEIGHTS_VERIFY=1 default). Confirm:
MT_WEIGHTS_PUBKEY=<key.pub.pem> musetalk-mlx-offline --weights weights --mlx-dir weights-mlx \
    --audio <audio.wav> --video <base.mp4> --out /tmp/out.mp4
```

Pass: log shows `weight verification OK: integrity + authenticity OK`. A
tampered bundle raises and aborts load.

## 5. Cross-chip smoke (M1-M4)

At least one benchmark + short stress per chip family to establish the minimum
supported machine (PRD claims M1-M5). Record results in `results/`.
