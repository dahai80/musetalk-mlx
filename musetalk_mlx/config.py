# Pipeline constants — musetalk-mig-mlx PRD V1.1-RC2 (../arch/musetalk-mig-mlx-v2-0918.md).
# Parity/perf gates: do not tune casually. Deploy-time overrides: key switches
# read MT_* env vars (audit P3-8) so a deployment can retune without repackaging.

import os


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


FPS = 30
SR = 16000
CHANNELS = 1

# FR-END-001: 33ms inference step aligned to 30FPS, 5s sliding audio window.
# Step is derived as SR // FPS (533 samples) so a 5s window is an integer
# number of steps (150). Windows overlap by OVERLAP_S so the boundary audio is
# re-encoded (musetalk-mlx half of FR-END-001); the embedding-level prefix cache
# is fusion-mlx issue #914. STEP_MS is the true per-step duration (1000/FPS);
# do NOT use a rounded 33 — it drifts 0.67% vs the sample-derived step and
# accumulates a frame every ~150 steps (audit P1-24).
STEP_MS = 1000.0 / FPS
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
# CHUNK_AROUND=2 -> 2 left + center + 2 right whisper chunks per video frame.
# At WHISPER_FPS=50 each chunk is 20ms, so the window spans 5*20=100ms centered
# on the frame: ~40ms look-ahead contributes to audio-to-video RTT. With
# BATCH=2 (+33ms) this must stay within the 80ms RTT budget (FR-LK-001) — do
# not raise CHUNK_AROUND without re-checking RTT (audit P4-7).
CHUNK_AROUND = 2

# FR-LK-001: output video PTS inherits the input audio PTS.
# Never stamp video frames from the system clock.

# FR-END-003 thermal tiers, in degradation order. Never jump 256 -> 128
# directly: step cut -> frame reuse -> bg downscale -> patch 128.
# NOTE: PATCH_CRITICAL=128 implies a 16x16 latent (not 32x32). The UNet/VAE
# must accept that shape — session validates compatibility at startup; if the
# core does not support 128, this tier is unreachable and should be removed
# (audit P1-26).
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

# Non-functional budgets.
MEM_BUDGET_GB = 4.0
LEAK_BUDGET_MB = 50.0  # over a 2h session

# FR-END-002 bg frame pool cap. _preload_bg() materializes every base-video
# frame as 1080p BGR uint8 (~6.2MB/frame) — a 10min clip is ~3.7GB, blowing the
# 4GB budget at startup. Two bounds: a frame count (loop length) AND a byte
# budget (memory). Whichever binds first stops the preload; beyond it the
# session serves frames on-demand from the imageio reader. Prior default 900
# frames = 5.6GB, already over the 4GB budget — the byte guard is the real
# fence (audit H4). 240 frames = 8s@30fps loop, ~1.5GB.
BG_POOL_MAX_FRAMES = _env_int("MT_BG_POOL_MAX_FRAMES", 240)
BG_POOL_BUDGET_MB = _env_int("MT_BG_POOL_BUDGET_MB", 1024)

# Phase 4 (non-blocking): LCM single-step stub. The fusion-mlx pipe is already
# single-step t=0; LCM_STEPS has no pipe-side effect until a distilled-weights
# API lands upstream. Disabled by default.
LCM_ENABLED = _env_bool("MT_LCM_ENABLED", False)
LCM_STEPS = 1

# FR-MLX-003: consume the fusion-mlx #911 graph pass (Conv+GN+SiLU fusion,
# mx.compile). Auto-degrades to plain ops when fusion-mlx lacks it.
GRAPH_OPT = _env_bool("MT_GRAPH_OPT", True)

# fusion-mlx #919 SmartConv2d (shape-dispatched im2col GEMM). Correct
# (parity clean) but measured 1.5x SLOWER inside the joint compiled
# UNet+decode graph on M5 Max (103.4ms -> 157.0ms p50): the microbench win
# at cliff shapes does not survive in-graph allocator traffic. Off by
# default; revisit per fusion-mlx release.
SMART_CONV = _env_bool("MT_SMART_CONV", False)

# fp16 pipeline cast (PRD 30FPS + <=4GB unified memory). fp32 batch-1 UNet
# is ~40ms on M5 Max (over the 33.3ms budget); fp16 is ~33ms.
FP16 = _env_bool("MT_FP16", True)

# MuseTalk-realtime-style offline precompute: landmarks + crop bbox + VAE
# latent per base frame, so the hot loop skips DWPose + encode per frame.
PRECOMPUTE = _env_bool("MT_PRECOMPUTE", True)

# Batched hot path: one UNet + one VAE decode per BATCH steps. RTT-safe
# (BATCH=2 adds one 33ms step; audio-to-video RTT stays <=80ms).
BATCH = _env_int("MT_BATCH", 2)

# Speed-over-quality decode switch (2026-09-20 decision): 2x2 average-pool the
# UNet output latent before the VAE decoder, so the decoder runs at 128^2
# output instead of 256^2 (decoder FLOPs scale with spatial^2: decode
# ~38-61ms -> ~10-18ms per round on M5 Max). UNet + audio conditioning are
# untouched. Default OFF — quality-gated (lip detail th/v/f, PSNR >= 38dB
# gate applies to the 256 path only). Enable when GPU throughput is the
# binding constraint and a softer face is acceptable.
DECODE_128 = _env_bool("MT_DECODE_128", False)

# GIL bypass for the paste/blend CPU work: run the cv2/numpy blend in a child
# PROCESS (own GIL) so MLX's busy-wait sync on the render thread cannot starve
# it. IPC moves only the expanded crop (~1.7MB per frame round trip), not the
# 1080p frame. DEFAULT OFF (audit A3): the double-track (worker thread + child
# process) adds 3 handoff layers + 2 pickle round-trips per frame with no
# measured win — _paste_alpha/paste_back are cv2/numpy and release the GIL
# natively, and MLX's sync-point GIL hold is unaffected by a paste subprocess.
# Enable only if a benchmark proves inline paste >15ms/round AND MLX holds the
# GIL through the sync; ship that benchmark with the change.
PASTE_MULTIPROC = _env_bool("MT_PASTE_MULTIPROC", False)
