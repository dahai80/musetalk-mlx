# Integration tests — pending fusion-mlx

End-to-end integration is **gated** on fusion-mlx landing all tracked issues.
Until then, only no-model unit tests run in `tests/`.

## Gated fusion-mlx issues

| Issue | Capability | Blocks |
|---|---|---|
| dahai80/fusion-mlx#909 | DWPose/RTMPose MLX face landmarks | real landmark detection (idle-blink fallback active) |
| dahai80/fusion-mlx#910 | Face-parsing BiSeNet MLX | production blend mask (feather fallback active) |
| dahai80/fusion-mlx#911 | Conv+GN+SiLU graph pass + SafeGroupNorm | Phase 3 perf (30 FPS) |
| dahai80/fusion-mlx#912 | Metal ICB batched encode | Phase 3 perf (RTT ≤ 80ms) |
| dahai80/fusion-mlx#913 | IOSurface↔CVPixelBuffer zero-copy | true zero-copy LiveKit output (copy fallback active) |
| dahai80/fusion-mlx#914 | encode_audio prefix-context cache | embedding-level window-boundary smoothing (overlap hop active) |

## Scenarios to run once unblocked

1. **Offline end-to-end parity** — `musetalk-mlx-offline --weights ... --audio ... --video ... --out ...`
   against MuseTalk torch reference: image PSNR ≥ 38 dB, SSIM ≥ 0.95 (PRD V2);
   manual lip-quality review on `MuseTalk/data/audio/` + `data/video/`.
2. **LiveKit realtime** — local LiveKit server, C++/Python joins room, Web client
   receives: 30 FPS, RTT ≤ 80ms, no tearing, no NaN/black-screen; PTS sync deviation measured.
3. **2h stability stress** — leak ≤ 50 MB + max-contiguous-allocatable fragmentation
   monitored; CVPixelBufferPool reuse validated.
4. **Thermal ladder** — stress validates the 4-step order (step-cut → frame-reuse →
   bg-downscale → patch 256→128); no 256→128 jump.
5. **Edge inputs** — silence (all-zero), clipping, < 10ms chunks: no crash, auto-standby.
6. **Per-stage benchmark** — STFT / Whisper / UNet / VAE / warp / frame-out timing →
   report FPS, RTT, unified-memory, fragmentation, PTS sync, keypoint alerts.
