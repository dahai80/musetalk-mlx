# Integration tests

fusion-mlx v0.10.1 fixed #915 (strict-load key mismatch, `mel_filters_80.npy`
packaged). Neural-core integration **verified** (7 passing in
`tests/integration/`); DWPose/face-parse weights now load strict-verified —
but **DWPose output is corrupted** (new upstream bug #917), so real-landmark
lip-sync parity remains blocked.

## v0.10.2 verification (#916/#917 fixed)

- fusion-mlx v0.10.2: DWPose forward matches ONNX ground truth (SPP-stage
  residual bug); `face_landmarks` returns frame-space coords directly.
- `_FrameSpaceDWPose` consumer workaround REMOVED (was double-scaling).
- Real-video verification: 10/10 frames detected, bbox 3.9% of frame area,
  0 border-pinned coords; tracker fails=0 through a full session.
- Graph-pass wiring fixed: `generate_faces` calls `mx.eval` (illegal inside
  mx.compile); session now compiles the pure UNet forward (+PE) and decodes
  outside. Compiled == plain output (cosine 1.0), no runtime fallback.
- Offline E2E with real weights + real landmarks: 150-frame video written,
  no guard fallback.

## v0.10.3 verification (#918/#919/#920)

- **#920 (allocator cache)**: verified. `tune_mlx_memory()` runs in both
  `from_pretrained*` paths (env-overridable); consumer-side caps remain in
  `session._tune_mlx_memory` (new `mx.set_cache_limit` API, deprecated
  `mx.metal` spelling as fallback). RSS stays ~4.2GB in the hot loop.
- **#918 (graph pass)**: verified honest. `compile_with_custom_pass` now logs
  that a bare wrapper is plain mx.compile; the real rewrite is
  `apply_patterns(root)`. Wired into `session._setup_graph_pass` (after fp16
  cast, before compile). On musetalk topology it matches 0 sites — resnets
  are `norm -> silu -> conv` and the pattern needs `conv -> norm -> silu`
  sibling adjacency — harmless by design.
- **#919 (SmartConv2d)**: verified correct, gated OFF. Parity clean after
  wrapping (UNet cosine 0.9996, VAE decode PSNR 67.4dB). But the microbench
  win does not survive in-graph: same-process A/B, joint compiled
  UNet+decode b2 p50 103.4ms plain vs 157.0ms with patterns+SmartConv (1.5x
  SLOWER — in-graph allocator traffic kills the im2col GEMM advantage,
  matching the earlier consumer-side shim finding). `config.SMART_CONV`
  gates it (default False); revisit per fusion-mlx release.
- 60/60 pytest green; parity thresholds all pass after rewrites.

## Performance (M5 Max, clean GPU, fusion-mlx server + linguakids watchdog stopped)

Methodology: background GPU load inflates every number several-fold. Before
measuring: `~/fusion/fusion-mlx/start.sh stop`,
`launchctl bootout gui/$(id -u)/com.linguakids.watchdog`, and check
`ioreg -l | grep '"Device Utilization %"'` ≈ 0. Other Claude sessions spawn
`fusion_mlx.media.image_worker` / `fusion-mlx-server` on demand — verify with
`ps aux` between runs.

Component micro-benchmarks (fp16, clean GPU, fresh process):

| Path | Time |
|---|---|
| UNet compiled b1 / b2 | 29.4ms / 46.7ms (23.4ms/frame) |
| VAE decode plain b2 | 131ms (65.6ms/frame) |
| VAE decode compiled b2 | 115ms (57.6ms/frame) |
| **Joint UNet+decode compiled b2 (one graph)** | **82.1ms (41.0ms/frame)** |
| Split compiled UNet then compiled decode b2 | 162ms — compiled-input boundary penalty, do not use |

Same-process A/B (session hot loop): split 178ms/round → joint 82ms/round
(2.2x). Sustained-load same-process standalone: 145-210ms/round (clock
degradation + background contention), session p50 145-165ms/round.

Memory: MLX default allocator cache grew to 13.55GB during the hot loop with
render spikes p90 >1.2s; `set_cache_limit(1GB)` + `set_memory_limit(3GB)`
(`session._tune_mlx_memory`) → cache 0.91GB, spikes p90 304ms, RSS 17GB →
4.2GB (PRD ≤4GB essentially met).

Current E2E sustained: ~8 FPS under chronic background GPU load (5
consecutive retries 7.5-8.0); a clean-window number was not obtainable during
this session. To hit 30FPS the remaining gap is upstream: conv2d kernel
cliffs (#919 — VAE decode 58ms/frame where ~20ms is reachable) and the
no-op graph pass (#918). Consumer-side levers (fp16, precompute, batch-2,
joint compile, memory caps, im2col-GEMM shim — the last one measured 7x
WORSE in-graph and was rejected) are exhausted.

NOTE: `com.linguakids.watchdog` (launchd) auto-restarts the fusion-mlx
server every interval — benchmarks are meaningless while it runs.

## Verified (tests/integration/, green)

- `test_neural_core.py`:
  - VAE encode → 8ch latent (1,8,32,32) → decode → 256×256×3 uint8 (FR-MLX-004).
  - UNet(t=0, audio cross-attn) → recon face, no NaN, no black-screen (FR-MLX-003).
  - `encode_audio` returns `(chunks, tail)`; prefix carries across windows (#914).
- `test_session_e2e.py`:
  - Full `MuseTalkSession` renders generated frames (crop→VAE→UNet→blend) via a
    stub landmark backend (DWPose blocked — see below).
  - Audio-inherited PTS, monotonic, never system clock (FR-LK-001).
  - `NumpyFrameSink` receives frames + PTS.
- Offline CLI with real v0.10.1 weights: runs clean, 60 frames @30fps; DWPose
  detections route through the bbox sanity guard → idle-blink fallback (no
  broken crops from the corrupted backend — see #917).
- Face-parse (#910) verified with real 79999_iter weights: sane 19-class labels
  (skin/lips/hair counts plausible) on a real face crop.

## #915 fix verified (v0.10.1)

- `convert_dwpose.py` / `convert_face_parsing.py` remap + transpose to the MLX
  param tree; both converters print `(strict OK)`.
- `mel_filters_80.npy` packaged in the wheel.
- Note: source `.pth` checkpoints in legacy tar format need `weights_only=False`
  (mmpose/resnet18 files); converters use `weights_only=True` — worked around by
  patching `torch.load` at conversion time (upstream follow-up candidate).

## Blocked — fusion-mlx #917 (DWPose output corruption)

| Subsystem | Status |
|---|---|
| DWPose (#909) | weights load strict-OK; **decoded landmarks are garbage** — mouth cluster std (60,92) px, border-pinned coords (x=0/254/384), bbox covers 75% of frame. `LandmarkTracker` bbox sanity guard routes to idle-blink. |
| Face-parsing (#910) | **working** — real 79999_iter weights, sane labels. |

## Scenarios to run once #917 is fixed

1. **Offline end-to-end parity** — `musetalk-mlx-offline` against MuseTalk torch
   reference: image PSNR ≥ 38 dB, SSIM ≥ 0.95 (PRD V2); manual lip-quality review.
2. **LiveKit realtime** — local LiveKit server, join room, Web client receives:
   30 FPS, RTT ≤ 80ms, no tearing/NaN; PTS sync deviation measured.
3. **2h stability stress** — leak ≤ 50 MB + max-contiguous-allocatable fragmentation;
   CVPixelBufferPool reuse.
4. **Thermal ladder** — stress validates the 4-step order (step-cut → frame-reuse →
   bg-downscale → patch 256→128); no 256→128 jump.
5. **Edge inputs** — silence / clipping / <10ms: no crash, auto-standby.
6. **Per-stage benchmark** — STFT/Whisper/UNet/VAE/warp/frame-out timing → FPS, RTT,
   unified-memory, fragmentation, PTS sync, keypoint alerts.

## Layered parity (tests/parity/, PRD V1.1-RC2 thresholds)

Torch-reference fixtures generated once via `musetalk_mlx/tools/gen_parity_fixtures.py`
(torch CPU env, seed 0); comparison at test time is torch-free. Gated on fixture
presence (`tests/parity/fixtures/meta.json`).

| Check | Threshold | Status |
|---|---|---|
| Whisper encoder stacked hidden states | cosine ≥ 0.98 | fixture + test ready |
| VAE encode latent (posterior mean) | cosine ≥ 0.98 | fixture + test ready |
| VAE decode image | PSNR ≥ 38 dB | fixture + test ready |
| 8-ch UNet predicted latent (t=0, post-PE audio) | cosine ≥ 0.98 | fixture + test ready |
| DWPose landmarks | n/a | **gated on fusion-mlx #917** |
| Full-pipeline image (offline end-to-end) | PSNR ≥ 38 / SSIM ≥ 0.95 | gated on #917 |
