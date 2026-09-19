# Pipeline constants — musetalk-mig-mlx PRD V1.1-RC2 (../arch/musetalk-mig-mlx-v2-0918.md).
# Parity/perf gates: do not tune casually.

FPS = 30
SR = 16000
CHANNELS = 1

# FR-END-001: 33ms inference step aligned to 30FPS, 5s sliding audio window.
# Step is derived as SR // FPS (533 samples) so a 5s window is an integer
# number of steps (150). Windows overlap by OVERLAP_S so the boundary audio is
# re-encoded (musetalk-mlx half of FR-END-001); the embedding-level prefix cache
# is fusion-mlx issue #914.
STEP_MS = 33
WINDOW_S = 5.0
OVERLAP_S = 0.2  # tail/head overlap between consecutive 5s windows

# Face patch (256x256 crop -> 32x32 latent) and 1080p output.
PATCH = 256
LATENT = 32
OUT_W = 1920
OUT_H = 1080

# Whisper-tiny frontend (mirrors fusion_mlx.video.musetalk_mlx.config).
AUDIO_FEATURE_DIM = 384
WHISPER_FPS = 50.0
CHUNK_AROUND = 2  # 2 left + center + 2 right per video frame

# FR-LK-001: output video PTS inherits the input audio PTS.
# Never stamp video frames from the system clock.

# FR-END-003 thermal tiers, in degradation order. Never jump 256 -> 128
# directly: step cut -> frame reuse -> bg downscale -> patch 128.
THERMAL_NORMAL = 0
THERMAL_SERIOUS = 1
THERMAL_CRITICAL = 2
DDIM_STEPS = 15
DDIM_STEPS_SERIOUS = 8
FRAME_REUSE = 3  # critical: infer 1 of N frames, repeat the rest
BG_DOWNSCALE_CRITICAL = 2  # critical: bg rendered at 1/N res then upsampled
PATCH_CRITICAL = 128  # critical last-resort: face patch 256 -> 128
BLEND_EXPAND = 1.5  # production paste expand factor (MuseTalk blending.get_crop_box)
BLEND_UPPER_BOUNDARY_RATIO = 0.5  # keep lower mouth region of the parse mask

# FR-MLX-001 / FR-END-006 landmark robustness.
KEYPOINT_FAIL_IDLE = 5  # consecutive failures -> idle-blink state
KALMAN_HISTORY = 5

# Non-functional budgets.
MEM_BUDGET_GB = 4.0
LEAK_BUDGET_MB = 50.0  # over a 2h session

# Phase 4 (non-blocking): LCM single-step stub. No distillation here — the
# 1-step weights / distillation are a separate task; main release uses
# multi-step DDIM (DDIM_STEPS). Flag flips the session to the LCM fast path
# once fusion-mlx ships compatible 1-step weights.
LCM_ENABLED = False
LCM_STEPS = 1

# FR-MLX-003: consume the fusion-mlx #911 graph pass (Conv+GN+SiLU fusion,
# mx.compile). Auto-degrades to plain ops when fusion-mlx lacks it.
GRAPH_OPT = True

# fp16 pipeline cast (PRD 30FPS + <=4GB unified memory). fp32 batch-1 UNet
# is ~40ms on M5 Max (over the 33.3ms budget); fp16 is ~33ms.
FP16 = True

# MuseTalk-realtime-style offline precompute: landmarks + crop bbox + VAE
# latent per base frame, so the hot loop skips DWPose + encode per frame.
PRECOMPUTE = True

# Batched hot path: one UNet + one VAE decode per BATCH steps. RTT-safe
# (BATCH=2 adds one 33ms step; audio-to-video RTT stays <=80ms).
BATCH = 2
