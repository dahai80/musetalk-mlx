# Pipeline constants — musetalk-mig-mlx PRD V1.1-RC2 (../arch/musetalk-mig-mlx-v2-0918.md).
# Parity/perf gates: do not tune casually.

FPS = 30
SR = 16000
CHANNELS = 1

# FR-END-001: 33ms inference step aligned to 30FPS, 5s sliding audio window.
# Step is derived as SR // FPS (533 samples) so a 5s window is an integer
# number of steps (150).
STEP_MS = 33
WINDOW_S = 5.0

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

# FR-MLX-001 / FR-END-006 landmark robustness.
KEYPOINT_FAIL_IDLE = 5  # consecutive failures -> idle-blink state
KALMAN_HISTORY = 5

# Non-functional budgets.
MEM_BUDGET_GB = 4.0
LEAK_BUDGET_MB = 50.0  # over a 2h session
