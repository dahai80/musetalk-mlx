# Integration tests

fusion-mlx v0.10.0 landed #909–#914. Neural-core integration is **verified**
(7 passing in `tests/integration/`); DWPose/face-parse **model weight loading
is blocked** by a fusion-mlx convert-script + packaging bug (#915).

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
- Offline CLI: `musetalk-mlx-offline --mlx-dir /tmp/mtlk_mlx_dist ...` writes a
  valid 30fps mp4 (30 frames, 768×576, std 60.96 — real content).

## Blocked — fusion-mlx #915

| Subsystem | Status |
|---|---|
| DWPose (#909) | code landed; **weights fail to load** — `convert_dwpose.py` dumps raw mmpose keys (`backbone.stem.2.bn.*`, `head.gau.*`) but `DWPoseMLX` expects `backbone.stem.layers.*` → silently untrained → `LandmarkTracker` idle-blink fallback. Parity vs torch reference not yet measurable. |
| Face-parsing (#910) | code landed; **weights fail to load** — `convert_face_parsing.py` dumps raw resnet18 keys (`layer4.1.conv2.*`) but `BiSeNetMLX` expects `spatial.b1.conv.*` → feather-mask fallback. |
| `mel_filters_80.npy` | **not packaged in wheel** — `log_mel_spectrogram` FileNotFoundError from `site-packages`; worked around by manual asset copy for this run. |

## Scenarios to run once #915 is fixed

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
