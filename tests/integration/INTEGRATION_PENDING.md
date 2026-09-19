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

## Performance (M5 Max, clean GPU, fusion-mlx server + linguakids watchdog stopped)

Component micro-benchmarks (fp16): UNet 32ms, VAE decode 23ms, VAE encode
(2x) 51ms — components fit the 33ms budget only marginally in total.
Sustained full-pipeline: 5.8 FPS (was 3.7 before precompute). Gaps:

1. Sustained GPU load degrades MLX small-kernel throughput ~3x vs short
   benchmarks (boost vs sustained clocks).
2. Hot-loop DWPose + VAE encode per frame — fixed via offline precompute
   (config.PRECOMPUTE, ~60s startup for 550 frames, ~9MB latents).
3. Remaining lever: batched UNet inference (fusion-mlx `run_batched`;
   batch-2 ≈ 27ms/frame at 54ms latency, fits RTT ≤ 80ms).

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
