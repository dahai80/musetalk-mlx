# musetalk-mlx

MuseTalk 1.5 lip-sync digital human on Apple Silicon. The business layer of the
K12 English-teacher real-time digital human, built on the **fusion-mlx** MLX
foundation. Spec: [`../arch/musetalk-mig-mlx-v2-0918.md`](../arch/musetalk-mig-mlx-v2-0918.md) (PRD V1.1-RC2).

## Split with fusion-mlx

| Layer | Location | Contents |
|---|---|---|
| Neural core | `fusion-mlx` → `fusion_mlx.video.musetalk_mlx` | UNet, SD-VAE, Whisper-tiny, `MuseTalkPipeline`, weight loaders |
| Business | this repo | face processing, realtime session, thermal degradation, LiveKit adapter, tools |

The neural core is consumed as a dependency — no model code is duplicated here.

## Pipeline

```
16k mono PCM -> 5s windows (33ms steps) -> Whisper-tiny embedding (50Hz)
base video frame -> 68-pt DWPose face landmarks -> 256x256 crop -> VAE encode
-> UNet(t=0, audio cross-attn) -> VAE decode -> face-parse mask paste-back
-> output frame with audio-inherited PTS -> LiveKit
```

## Hard targets (PRD)

30 FPS · audio-to-video RTT ≤ 80ms · ≤ 4GB unified memory · macOS 14+ · no PyTorch at runtime.

## Setup

```bash
cd musetalk-mlx
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The `fusion-mlx` dependency is a local `file://` pin on `~/fusion/fusion-mlx` (see `pyproject.toml`).

### Weights

Expected layout (matches `MuseTalkPipeline.from_pretrained`):

```
weights/
├── sd-vae-ft-mse/                  # stabilityai/sd-vae-ft-mse
├── MuseTalk/musetalkV15/unet.pth   # TMElyralab/MuseTalk
└── whisper-tiny/                   # openai/whisper-tiny
```

Download via https://hf-mirror.com, or convert the MuseTalk clone's `models/` layout with `musetalk-mlx-convert`.

## Commands

| Command | Purpose |
|---|---|
| `musetalk-mlx-convert --weights ... --out ...` | PyTorch weights → MLX safetensors dist dir |
| `musetalk-mlx-offline --weights ... --audio ... --video ... --out ...` | offline: audio + base video → output video |
| `musetalk-mlx-realtime --weights ... --video ... --livekit-url ... --token ...` | realtime: streaming → LiveKit (FR-LK-001/002) |
| `pytest tests/ -v` | run tests |
| `ruff check .` | lint |

## DWPose / face landmarks

`LandmarkTracker` consumes 68-point DWPose face landmarks (COCO-WholeBody
keypoints `[23:91]`, same convention as MuseTalk `preprocessing.get_landmark_and_bbox`)
and derives the axis-aligned crop bbox via `derive_face_bbox`. Landmarks are
smoothed with a vectorized constant-velocity Kalman filter; a single-frame
detection miss holds the last pose, `KEYPOINT_FAIL_IDLE` consecutive misses
trigger idle-blink (emit the base frame unchanged).

The MLX DWPose/RTMPose backend ships in fusion-mlx, but its decoded landmarks
are currently corrupted (upstream #917) and returned in network-input space
(#916 — rescaled locally). `load_dwpose_backend()` wraps the backend and a bbox
sanity guard in `LandmarkTracker` degrades to idle-blink instead of producing
broken crops. The bbox math, Kalman smoother, guard, and idle logic are covered
by `tests/test_landmarks.py` independent of the model.

## Phase status

- [x] Phase 1 (partial): neural core in fusion-mlx v0.2.0; DWPose bbox math + Kalman + idle-blink wired; MLX DWPose backend pending fusion-mlx #909
- [x] Phase 2 (business layer): overlapping audio windower (prefix-smoothing), production blend paste-back + pluggable face-parse mask, zero-copy frame sink (copy fallback), offline demo polish. Embedding-level prefix cache pending fusion-mlx #914; face-parse model pending #910; true zero-copy pending #913.
- [x] Phase 3 (business layer): full thermal degradation ladder (FR-END-003), ReloadModel hot-reload (FR-MLX-006), LiveKit realtime adapter (FR-LK-001/002, audio-inherited PTS, bidirectional audio), realtime runner CLI. Graph pass + ICB perf pending fusion-mlx #911/#912.
- [x] Phase 4 (stub): LCMFastSession config stub (disabled; main release uses multi-step DDIM). Distillation training out of scope.
- [ ] Integration testing: neural core verified (7 integration tests green); #915 fix verified (strict weight load + mel packaging); real-landmark lip-sync parity blocked by fusion-mlx #917 (DWPose output corruption) — see `tests/integration/INTEGRATION_PENDING.md`.

## Performance (M5 Max)

Measured with the fusion-mlx server and any other GPU consumers stopped —
background GPU load inflates numbers several-fold (see
`tests/integration/INTEGRATION_PENDING.md`).

| Optimization | Effect |
|---|---|
| fp16 cast + offline precompute (landmarks/bbox/VAE latent) | DWPose + encode out of the hot loop |
| Joint `mx.compile` of UNet+VAE-decode (one graph) | 178ms → 82ms per batch-2 round (2.2x); split compiled calls pay a compiled-input boundary penalty |
| **MLX 0.32.0 pin** | 0.32.2 ships a decode kernel regression on the compiled joint graph: 73.6 vs 61.6ms/round measured. `mlx==0.32.0` + `mlx-metal==0.32.0` required for the numbers below |
| **Allocator limits POST-warmup** (cache 4GB + budget 8GB after load/precompute) | limits set before load poison the allocator watermark: 84–92ms vs 62–67ms per round, all process long. Cache sweep at 2.5/3/3.5/4/5/6GB: 61.8/58.9/59.1/57.0/56.9/58.4ms |
| Depth-2 render-ahead | two lazy rounds queued so the GPU stays busy across paste/emit CPU work (single stream; sync is stream-wide) |
| Paste worker thread + `cv2.blendLinear` | warp/blend/emit (pure cv2/numpy) off the render thread; blend 7x faster than numpy fp32 math; parse-mask cache misses deferred to the main thread |
| Batch-2 hot path (`config.BATCH`) | one UNet+decode per 2 steps, RTT-safe (adds one 33ms step) |

Isolated clean-GPU microbenchmarks (fp16, MLX 0.32.0, v0.10.4): compiled joint
UNet+VAE-decode batch-2 = **60.8–67ms/round** (30.4–33.5ms/frame, **30–33 FPS**
GPU-only ceiling). Live pipeline: **24.4 FPS average** over a 400-frame run
(was 15.4), with steady windows touching 30 FPS — GPU clock state dominates
run-to-run variance (idle-boosted processes measure 60ms/round; hot ones 95ms
for identical code). Remaining known costs: ~20ms/round of paste/emit CPU work
partly starved by the GIL during MLX's busy-wait sync, and window-boundary
whisper encodes. The earlier "58ms decode / 8 FPS sustained" figures were
contention-contaminated (linguakids watchdog auto-restarting the fusion-mlx
server) and are retracted.

## PRD V1.1-RC2 gap-fill (this release)

- fusion-mlx v0.10.2 verification (#916/#917): real-landmark path green — 10/10 detections, sane bbox; consumer `_FrameSpaceDWPose` workaround removed.
- Graph-pass consumption fixed: compiles the pure UNet forward (`generate_faces` calls `mx.eval`, illegal under `mx.compile`); output identical to plain (cosine 1.0).
- fp16 pipeline cast (`config.FP16`) + MuseTalk-realtime-style offline precompute of landmarks/bbox/VAE-latents per base frame (`config.PRECOMPUTE`), removing DWPose + encode from the hot loop.

- Layered parity tests vs PyTorch (`tests/parity/`): Whisper encoder, VAE encode (cosine >= 0.98), VAE decode (PSNR >= 38dB), 8-ch UNet (cosine >= 0.98). One-time fixture generation in a torch env (`musetalk_mlx/tools/gen_parity_fixtures.py`); comparison at test time is torch-free.
- Kalman PREDICTION on single-frame landmark miss (FR-END-006): pos+velocity propagation, idle-blink only after 5 consecutive misses; bbox-plausibility + outlier-rejection guards.
- Full thermal ladder wiring in-session (FR-END-003): step-cut -> frame-reuse(3) -> bg-downscale(2) -> patch 256->128 (never a direct 256->128 jump).
- Base-video frame preload memory pool (FR-END-002) with streaming fallback.
- Graph-pass consumption via fusion-mlx `compile_with_custom_pass` (#911), probe-based graceful fallback.
- Zero-copy frame sink on `MetalZeroCopyBridge.array_to_cvbuffer` (#913) with copy fallback.
- Per-stage profiler (STFT/Whisper/UNet/VAE/warp/frame-out), phys-footprint memory sampling, max-contiguous-alloc probe (`musetalk_mlx/utils/profiling.py`).
- Ops tooling: `musetalk-mlx-benchmark` (FPS + per-stage budgets, PRD 9.5) and `musetalk-mlx-stress` (long-session leak <= 50MB + fragmentation probe).
- Edge-input tests: sub-10ms audio, all-silence, full-scale clipping.

## fusion-mlx dependency issues

| Issue | Capability | Phase |
|---|---|---|
| [#909](https://github.com/dahai80/fusion-mlx/issues/909) | DWPose/RTMPose MLX face landmarks | 1 |
| [#910](https://github.com/dahai80/fusion-mlx/issues/910) | Face-parsing BiSeNet MLX | 2 |
| [#911](https://github.com/dahai80/fusion-mlx/issues/911) | Conv+GN+SiLU graph pass + SafeGroupNorm | 3 |
| [#912](https://github.com/dahai80/fusion-mlx/issues/912) | Metal ICB batched encode | 3 |
| [#913](https://github.com/dahai80/fusion-mlx/issues/913) | IOSurface↔CVPixelBuffer zero-copy | 3 |
| [#914](https://github.com/dahai80/fusion-mlx/issues/914) | encode_audio prefix-context cache | 2 |
| [#915](https://github.com/dahai80/fusion-mlx/issues/915) | convert_dwpose/convert_face_parsing key mismatch + mel asset packaging | 1/2 |
| [#916](https://github.com/dahai80/fusion-mlx/issues/916) | DWPose coords in network-input space, not frame space | 1 |
| [#917](https://github.com/dahai80/fusion-mlx/issues/917) | DWPose output corrupted despite strict load | 1 |
| [#918](https://github.com/dahai80/fusion-mlx/issues/918) | `compile_with_custom_pass` is a no-op stub (patterns never applied) — **fixed v0.10.3, 0 matches on musetalk topology** | 3 |
| [#919](https://github.com/dahai80/fusion-mlx/issues/919) | Metal conv2d fp16 throughput cliffs up to 8x between shapes — **fixed v0.10.3; SmartConv2d correct but 1.5x slower in-graph, gated off** | 3 |
| [#920](https://github.com/dahai80/fusion-mlx/issues/920) | Default allocator cache grows unbounded, multi-second render spikes — **fixed v0.10.3** | 3 |
| [#921](https://github.com/dahai80/fusion-mlx/issues/921) | Metal conv2d fp16 kernel cliffs — sole remaining 30FPS blocker (decode 58ms -> ~20ms needed) | 3 |

## Layout

```
musetalk_mlx/
├── config.py          # pipeline constants (PRD gates)
├── face/              # landmarks (Kalman, idle fallback) + 256 crop
├── pipeline/          # MuseTalkSession + feather blending
├── livekit/           # phase3 adapter (optional extra)
├── utils/             # audio windower + thermal tiers
└── tools/             # weight conversion + offline CLI
```
