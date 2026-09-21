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
# fusion-mlx neural core (editable, from its checkout):
pip install -e ~/fusion/fusion-mlx
pip install -e ".[dev]"
```

`fusion-mlx[video]>=0.10.2,<0.11` is a versioned dependency (no hardcoded local path);
install the fusion-mlx checkout editable first so the version constraint is satisfied.

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

## Environment variables

Deploy-time overrides (no repackaging needed). All are `MT_*` env vars read at import time.

| Variable | Default | Purpose |
|---|---|---|
| `MT_LCM_ENABLED` | `false` | Phase-4 LCM stub (no effect until fusion-mlx ships distilled 1-step weights) |
| `MT_GRAPH_OPT` | `true` | fusion-mlx #911 Conv+GN+SiLU graph-pass + joint mx.compile |
| `MT_SMART_CONV` | `false` | fusion-mlx #919 SmartConv2d (measured 1.5x slower in-graph; off) |
| `MT_FP16` | `true` | fp16 pipeline cast (30FPS + <=4GB budget) |
| `MT_PRECOMPUTE` | `true` | offline landmarks+bbox+latent precompute per base frame |
| `MT_BATCH` | `2` | batched UNet+decode depth (RTT-safe at 2) |
| `MT_DECODE_128` | `false` | 2x2 avg-pool latent before decode (speed over quality) |
| `MT_PASTE_MULTIPROC` | `false` | paste blend in a child process (off — no measured win, adds IPC) |
| `MT_BG_POOL_MAX_FRAMES` | `240` | bg frame pool cap (8s@30fps, ~304MB); longer bg served on-demand from the reader |

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
- [x] Phase 2 (business layer): overlapping audio windower (prefix-smoothing), production blend paste-back + pluggable face-parse mask, zero-copy frame egress (memoryview handoff to livekit VideoFrame, FR-LK-001), offline demo polish. Embedding-level prefix cache pending fusion-mlx #914; face-parse model pending #910; native IOSurface→CVPixelBuffer direct-encoding pending #913 (livekit Python SDK does not accept CVPixelBuffer today; bridge wired for non-LiveKit sinks).
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
| Depth-2 render-ahead | two lazy rounds queued so the GPU stays busy across paste/emit CPU work (single stream; sync is stream-wide). Caveat: MLX lazy eval currently serializes — a submitted round's graph only materializes on sync, so the depth-2 queue does not yet overlap GPU compute with paste. Overlap needs eager dispatch (fusion-mlx issue pending) |
| Paste worker thread + `cv2.blendLinear` | the hot path pastes inline on the dedicated render thread (cached alpha ~0.2ms); `cv2.blendLinear` blend is 7x faster than numpy fp32 math. The worker thread path is retained for fallback/drain |
| **Subprocess paste worker** (`config.PASTE_MULTIPROC`, default OFF) | paste/blend runs in a child process (own GIL), immune to the GIL starvation from MLX's busy-wait sync on the render thread. 4-byte length-framed pickle protocol over stdin/stdout; IPC ships the expanded crop only, not the full frame. Default OFF: the hot path now pastes inline on the render thread (cached alpha ~0.2ms, no IPC); the worker is retained as a fallback for drain/IPC-failure paths |
| **`DECODE_128` speed switch** (`config.DECODE_128`, default OFF) | 2x2 average-pool the UNet output latent (32²→16²) before VAE decode → 128² face patch at ~1/4 decode FLOPs. Speed-over-quality for scenarios that tolerate it; ships disabled, opt-in by flag |
| Batch-2 hot path (`config.BATCH`) | one UNet+decode per 2 steps, RTT-safe (adds one 33ms step) |

Isolated clean-GPU microbenchmarks (fp16, MLX 0.32.0, allocator caps on,
v0.10.4), 60-round burst:

| Config | GPU/round (b2) | GPU ceiling | Live e2e FPS |
|---|---|---|---|
| Full decode (default) | 61.0ms | 32.8 FPS | 19.4 |
| `DECODE_128` on | 34.1ms | 58.7 FPS | 27.7 |

The live-to-ceiling gap (~19ms/frame) is paste + readback serialized on the
main thread — depth-2 render-ahead does not yet overlap because MLX lazy
evaluation materializes a round's graph only on sync (see caveat above).
**The 30 FPS lever is the VAE decoder**: decode alone is 59ms of the 61ms
round, dominated by the 3-upsample conv stack. Conv+GroupNorm+SiLU MSL fusion
(fusion-mlx issue, design below) is the path to cut decode to ~30ms and clear
30 FPS at full 256² quality. `DECODE_128` already clears 30 FPS at 128² when
quality can be traded.

The earlier "58ms decode / 8 FPS sustained" and "24.4 FPS live" figures were
contention-contaminated (linguakids watchdog auto-restarting the fusion-mlx
server) and are retracted; the table above is clean-GPU.

## PRD V1.1-RC2 gap-fill (this release)

- fusion-mlx v0.10.2 verification (#916/#917): real-landmark path green — 10/10 detections, sane bbox; consumer `_FrameSpaceDWPose` workaround removed.
- Graph-pass consumption fixed: compiles the pure UNet forward (`generate_faces` calls `mx.eval`, illegal under `mx.compile`); output identical to plain (cosine 1.0).
- fp16 pipeline cast (`config.FP16`) + MuseTalk-realtime-style offline precompute of landmarks/bbox/VAE-latents per base frame (`config.PRECOMPUTE`), removing DWPose + encode from the hot loop.

- Layered parity tests vs PyTorch (`tests/parity/`): Whisper encoder, VAE encode (cosine >= 0.98), VAE decode (PSNR >= 38dB), 8-ch UNet (cosine >= 0.98). One-time fixture generation in a torch env (`musetalk_mlx/tools/gen_parity_fixtures.py`); comparison at test time is torch-free.
- Kalman PREDICTION on single-frame landmark miss (FR-END-006): pos+velocity propagation, idle-blink only after 5 consecutive misses; bbox-plausibility + outlier-rejection guards.
- Full thermal ladder wiring in-session (FR-END-003): step-cut -> frame-reuse(3) -> bg-downscale(2) -> patch 256->128 (never a direct 256->128 jump). **Note (audit 0921 P1-7):** the serious-tier step-cut (`set_ddim_steps` 15→8) is a no-op on the realtime path — the realtime render loop is fixed single-step t=0 to hold the 33ms budget (multi-step DDIM is Nx slower). Serious-tier only takes effect on the offline multi-step path. Realtime degradation relies on the critical tier (frame-reuse / bg-downscale / patch-128).
- Base-video frame preload memory pool (FR-END-002) with streaming fallback.
- Graph-pass consumption via fusion-mlx `compile_with_custom_pass` (#911), probe-based graceful fallback.
- Zero-copy frame egress (FR-LK-001): `LiveKitAdapter.publish_frame` hands the BGRA pixel buffer to the livekit rtc `VideoFrame` as a `memoryview` of a pre-allocated rotating numpy pool — the SDK's `get_address` passes the buffer pointer to the FFI (`ctypes.addressof(c_char.from_buffer)`), no `tobytes`/`bytearray` copy (prior path did 3 copies/frame). `capture_frame` is a synchronous FFI request; the native encoder reads the pointer during the call. `ZeroCopySink` exposes the fusion-mlx #913 `MetalZeroCopyBridge` (IOSurface-backed CVPixelBuffer, native path; CVPixelBufferCreate+memcpy fallback) for non-LiveKit sink consumers (offline, future native VideoToolbox direct-encoding); the livekit Python SDK does not accept a CVPixelBuffer directly, so the bridge's IOSurface path is reserved for a future native encoder integration.
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
| [#924](https://github.com/dahai80/fusion-mlx/issues/924) | MSL fused Conv+GN+SiLU kernel — open, blocks 30FPS (decode ~45ms -> ~20ms) | 3 |
| [#927](https://github.com/dahai80/fusion-mlx/issues/927) | `set_ddim_steps` API missing — serious-tier step-cut is a no-op, thermal ladder jumps normal->critical | 3 |
| [#928](https://github.com/dahai80/fusion-mlx/issues/928) | Public API contract — musetalk-mlx reaches into private attrs (`_dtype`/`UNET_TIMESTEP`/`apply_pe`/`unet`); version lock is a fallback, not a contract — **closed v0.10.5, migrated to `pipe.dtype`/`pipe._run_unet`** | 3 |
| [#932](https://github.com/dahai80/fusion-mlx/issues/932) | `apply_patterns` matches 0 modules on MuseTalk UNet/VAE — pattern matcher blind to code-level GN→SiLU→Conv call sequences; sole remaining 30FPS blocker (render_eval 108ms/round, need 66ms) | 3 |

## Operations runbook

### Single-machine deployment (Apple Silicon, macOS 14+)

1. **Prereqs**: Xcode 18.0 command-line tools, Python 3.11 venv, `brew install ffmpeg`. PyObjC required for thermal monitoring: `pip install pyobjc`. Without it the session logs a warning and runs normal-tier only (no thermal degradation).
2. **Weights**: place under `weights/` (MuseTalk `unet.pth` ~3.2GB, `whisper-tiny`, `sd-vae-ft-mse`, `weights/eval`). Torch-free users must obtain pre-converted MLX safetensors (see Weight distribution below) — `musetalk-mlx-convert` requires a torch env and cannot run torch-free.
3. **fusion-mlx**: install editable from its checkout (`pip install -e ~/fusion/fusion-mlx[video]`), pinned `>=0.10.2,<0.11`. Start/stop the service with `~/fusion/fusion-mlx/start.sh start|stop`.
4. **Offline**: `musetalk-mlx-offline --weights weights --audio in.wav --video base.mp4 --out out.mp4`. Output is muxed with source audio via ffmpeg (audit B1).
5. **Realtime (LiveKit)**: `musetalk-mlx-realtime --weights weights --video base.mp4 --livekit-url wss://... --token <jwt>`. Token is held only for connect/reconnect and dropped on close.
6. **Auto-start (launchd)**: wrap the realtime CLI in a `~/Library/LaunchAgents/io.musetalk.mlx.plist` with `KeepAlive=true` so a crash restarts the daemon.

### Monitoring / alerting

- Logs are plain `logging` (stderr). For production, forward stderr to a structured sink (Loki/Cloudwatch) and alert on `ERROR`/`WARNING` rate.
- Key signals to alert on: `bg pool hit byte budget`, `LiveKit room disconnected`, `paste worker thread exited`, `set_ddim_steps unavailable`, `thermal` tier transitions, `NSProcessInfo unavailable`.
- `musetalk-mlx-stress` reports RSS + MLX active/cache/peak memory per sample; wire its JSON output into a 2h leak budget alert (threshold 50MB drift).

### Rollback

- fusion-mlx upper bound `<0.11`: a 0.11 release must be manually verified before bumping. To roll back, `pip install 'fusion-mlx[video]<0.11'` and restart.
- `MuseTalkSession.reload(mlx_dir)` hot-swaps weights without restart; drain happens at a window boundary, standby base frames emitted mid-swap (no black screen). Not a substitute for a version rollback — use for weight refresh only.
- **Reload fps dip (expected, not a fault)**: right after a successful reload the bg latent cache is rebuilt synchronously (DWPose + VAE encode per base frame, ~150ms/frame). Until it completes, cache misses fall back to the live encode path at roughly 6 FPS; output stays on standby base frames — no black screen, no audio drop. The dip self-heals as the cache fills (audit 0921 P1-6). Do not page on a post-reload fps warning that clears within seconds.

### Weight distribution (P0 gap — not yet shipped)

`weights/` is ~3.2GB and currently hand-placed. Commercial release requires one of:
- Pre-converted MLX safetensors published to HuggingFace (mirror via https://hf-mirror.com) with a `musetalk-mlx-download` fetch script, or
- A signed bundle distributed alongside the installer.

This is an open release blocker (audit B5); `musetalk-mlx-convert` exists but needs a torch env, so torch-free end users cannot self-build today.

## Threat model

- **LiveKit token**: short-lived JWT held in process memory for connect + reconnect watchdog only; never logged, dropped on `close()`. Rotate per session; do not embed long-lived tokens in launchd plists — read from Keychain or a secrets manager at startup.
- **Weights**: integrity is trust-on-first-load. `torch.load` uses `weights_only=True` (`eval/syncnet.py`); `MT_SYNCNET_UNSAFE_LOAD=1` is an explicit opt-out for dev only. Commercial builds should sign-verify the weight bundle before load (not yet implemented).
- **User input (audio/video)**: audio is decoded via `librosa` (no code execution); video via `imageio`/`cv2`. Untrusted media should be scanned before ingest — the pipeline does not sandbox decode. For multi-tenant deployments, run each session in a seatbelt/container.
- **IPC**: `paste_proc.py` uses pickle over a pipe, but the subprocess is spawned from `sys.executable -m` (trusted parent). No external input crosses the pickle boundary.
- **subprocess**: all shell calls (`ffmpeg`/`ffprobe`) are list-form, no `shell=True`.

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
