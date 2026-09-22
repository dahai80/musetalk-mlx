# musetalk-mlx

MLX port of [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) lip-sync digital
human, running on Apple Silicon with a torch-free runtime. The business layer of the
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

## Demo

Lip-synced outputs from [`musetalk-mlx-cases`](musetalk-mlx-cases/), reproducing the
[MuseTalk](https://github.com/TMElyralab/MuseTalk) 1.0 TestCases on MLX (no PyTorch at
runtime). Videos are hosted on the [demo-v1 release](https://github.com/dahai80/musetalk-mlx/releases/tag/demo-v1)
and rendered inline below.

<table>
<tr><th>Input</th><th>Output (musetalk-mlx)</th></tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/yongen.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_yongen.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/musk.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_musk.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/monalisa.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_monalisa.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/sit.jpg" width="200"><br><sub>sun1 (blink amplified)</sub></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_sun1.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/man.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_man.mp4" controls preload></video></td>
</tr>
</table>

Full case suite (sit, sun2, video1, …) + how to regenerate: see
[`musetalk-mlx-cases/README.md`](musetalk-mlx-cases/README.md).

## Setup

```bash
cd musetalk-mlx
python3.12 -m venv .venv
source .venv/bin/activate
# fusion-mlx neural core (editable, from its checkout):
pip install -e /path/to/fusion-mlx
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
| `MT_GRAPH_OPT` | `true` | fusion-mlx #911 Conv+GN+SiLU graph-pass + joint mx.compile |
| `MT_SMART_CONV` | `false` | fusion-mlx #919 SmartConv2d (measured 1.5x slower in-graph; off) |
| `MT_FP16` | `true` | fp16 pipeline cast (30FPS + <=4GB budget) |
| `MT_PRECOMPUTE` | `true` | offline landmarks+bbox+latent precompute per base frame |
| `MT_BATCH` | `2` | batched UNet+decode depth (RTT-safe at 2) |
| `MT_DECODE_128` | `false` | 2x2 avg-pool latent before decode (speed over quality) |
| `MT_PASTE_MULTIPROC` | `false` | paste blend in a child process (off — no measured win, adds IPC) |
| `MT_BG_POOL_MAX_FRAMES` | `240` | bg frame pool cap (8s@30fps, ~304MB); longer bg served on-demand from the reader |
| `MT_MLX_CACHE_GB` | `4` | MLX allocator cache cap (GB). Perf-optimal; set `1` to enforce the 4GB budget (RSS ~4.2GB, ~30% FPS cost) |
| `MT_MLX_LIMIT_GB` | `8` | MLX memory limit (GB). Set `3` to enforce the 4GB budget |
| `MT_CLEAR_CACHE_EVERY` | `300` | render loop calls `mx.clear_cache()`+`gc.collect()` every N frames to bound RSS in long sessions (0 disables) |

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
- [x] Phase 3 (business layer): full thermal degradation ladder (FR-END-003), ReloadModel hot-reload (FR-MLX-006), LiveKit realtime adapter (FR-LK-001/002, audio-inherited PTS, bidirectional audio), realtime runner CLI. Graph pass fusion-mlx #921/#924/#932 closed (module-level ResnetBlock2D fusion + MSL Conv+GN+SiLU kernel landed, default ON); ICB (#912) remains an upstream open item (not consumed by the Python/MLX path — route divergence).
- [x] Phase 4 (LCM): N/A in the MLX architecture. The fusion-mlx pipe is inherently single-step t=0, so there is no multi-step DDIM loop to distill into a 1-step fast path — LCM distillation is a no-op here. The prior `LCMFastSession` stub + `MT_LCM_ENABLED` flag instantiated a dead object never called from the render hot path, implying a fast path that does not exist; both removed for honesty. If fusion-mlx ships a distilled-weights API, reintroduce a real fast path (not a stub).
- [x] Integration testing: neural core verified (8 integration tests green in `tests/integration/`); #915 fix verified (strict weight load + mel packaging); #916/#917 fixed in fusion-mlx v0.10.2 — real-landmark path green (10/10 detections, sane bbox, `_FrameSpaceDWPose` workaround removed); offline E2E runs clean (150-frame video, real landmarks, no guard fallback); A3 LiveKit realtime verified (29.82 FPS, RTT 33.3ms, PTS monotonic); A4 2h stress running. Remaining gate: offline PSNR ≥ 38dB / SSIM ≥ 0.95 vs MuseTalk torch reference — `gen_gt.py` tool exists (subprocess into MuseTalk checkout, MPS), not yet run against the torch GT. See `tests/integration/INTEGRATION_PENDING.md`.

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
| **Dedicated render thread + inline paste** (A3) | producer thread pumps `scheduler.next_step`; `get_output_frame` = pure consumer (pop out_q, wait-while-busy). Paste inline on the render thread (cached alpha ~0.2ms, no worker queue). Fixed a 2:1 frame loss (`_wait_out_q(pop=False)` waits for emit without discarding) |
| **Async LiveKit publish thread + H264** (A3) | H264 codec (VideoToolbox hardware) — VP8 software capped ~5fps at 1080p. Dedicated `_publish_loop` thread with bounded deque (cap 4, drop oldest) decouples the GIL from the render thread's MLX readback (render_eval 140ms→55ms) |
| **Zero-copy egress** (FR-LK-001, A3) | BGRA handed to rtc `VideoFrame` as `memoryview` of a rotating numpy pool — SDK passes the pointer to FFI, no `tobytes`/`bytearray` (prior path: 3 copies/frame) |

Isolated clean-GPU microbenchmarks (fp16, MLX 0.32.0, allocator caps on,
v0.10.4), 60-round burst:

| Config | GPU/round (b2) | GPU ceiling |
|---|---|---|
| Full decode (default) | 61.0ms | 32.8 FPS |
| `DECODE_128` on | 34.1ms | 58.7 FPS |

LiveKit E2E (A3, clean GPU, 1080p, `MT_DECODE_128=1`, local LiveKit server
v1.9.1, `lk_receive.py` probe):

| Metric | Value |
|---|---|
| Receiver FPS | **29.82** |
| Publish-interval p50 | 33.3ms |
| render_eval | 55ms |
| Steady drops | 0 |
| RTT (render-pipeline segment) | 33.3ms ≤ 80ms |
| PTS | strictly monotonic |

The live-to-ceiling gap was closed by the render-thread + async publish +
zero-copy architecture (A3): the prior ~19ms/frame gap was paste + readback +
publish GIL contention serialized on the pacer thread. The VAE decoder
full-quality gap (#921/#924/#932 — conv2d fp16 cliffs + MSL Conv+GN+SiLU
fusion + module-level graph patterns) is now closed in fusion-mlx; the A3
LiveKit E2E measured 29.82 FPS at 128² with `DECODE_128=1`. Full 256² quality
30 FPS requires the fused MSL kernel to be re-benchmarked on a clean GPU with
the latest fusion-mlx (the 19.4 FPS full-decode figure predates the #921/#924
fixes).

The earlier "58ms decode / 8 FPS sustained" and "24.4 FPS live" figures were
contention-contaminated (linguakids watchdog auto-restarting the fusion-mlx
server) and are retracted; the A3 LiveKit E2E table above is clean-GPU.

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
- Barge-in interrupt (audit 0921 P3, K12 core gap): `MuseTalkSession.interrupt()` + `LiveKitAdapter.interrupt()` clear pending/inflight/out_q and the audio prefix (truncated-utterance embedding tail would poison the next turn); `consumed` stays monotonic (output PTS never regresses); `_last_frame` kept as standby (no black screen); paste queue lightly cleared (no 2s join). Explicit API only — VAD-based auto-trigger is a future item.
- Deadline-driven push model (audit A-1): `realtime_infer.py` pacer uses `next_deadline += period` with overrun resync instead of `sleep(period*0.5)` spin; jitter instrumentation (publish-interval p50/p95/max, overruns, pts_drift) logged every 10s + written to `results/realtime_pacing.json`. Transition state toward the PRD production "no Python main loop" target (final state = LiveKit queueing, Phase 3 follow-up).
- Session split (audit A-1): `BackgroundStore` (bg pool / cache / precompute) and `RenderScheduler` (round submit/finish/inflight/thermal application) extracted from `session.py`; public API unchanged, audit comments migrated verbatim, profiler stage strings untouched.

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
| [#921](https://github.com/dahai80/fusion-mlx/issues/921) | Metal conv2d fp16 kernel cliffs — **closed**: MuseTalk realtime loop reaches 30 FPS GPU-side | 3 |
| [#924](https://github.com/dahai80/fusion-mlx/issues/924) | MSL fused Conv+GN+SiLU kernel — **closed**: `fusion_mlx/custom_kernels/fused_conv_gn_silu.py` landed (conv_stats + gn_affine_silu kernels) | 3 |
| [#927](https://github.com/dahai80/fusion-mlx/issues/927) | `set_ddim_steps` runtime DDIM step-count setter — **closed v0.10.5**: `pipe.set_ddim_steps(n)` sets the pipe's `_ddim_steps` state; `_run_unet(steps=None)` consumes it. musetalk-mlx's realtime hot loop stays single-step (steps=1, t=0) to hold the 33ms budget; the serious-tier step-cut (15→8) takes effect on the offline multi-step path via `_apply_ddim_steps` | 3 |
| [#928](https://github.com/dahai80/fusion-mlx/issues/928) | Public API contract — musetalk-mlx reaches into private attrs (`_dtype`/`UNET_TIMESTEP`/`apply_pe`/`unet`); version lock is a fallback, not a contract — **closed v0.10.5, migrated to `pipe.dtype`/`pipe._run_unet`** | 3 |
| [#932](https://github.com/dahai80/fusion-mlx/issues/932) | `apply_patterns` module-level graph fusion — **closed**: whole-block ResnetBlock2D fusion (GN→SiLU→Conv pairs), default ON (`FUSION_MUSETALK_GRAPH_PATTERNS=1`), runs before SmartConv2d wrapping | 3 |

## Operations runbook

### Single-machine deployment (Apple Silicon, macOS 14+)

1. **Prereqs**: Xcode 18.0 command-line tools, Python 3.11 venv, `brew install ffmpeg`. PyObjC required for thermal monitoring: `pip install pyobjc`. Without it the session logs a warning and runs normal-tier only (no thermal degradation).
2. **Weights**: place under `weights/` (MuseTalk `unet.pth` ~3.2GB, `whisper-tiny`, `sd-vae-ft-mse`, `weights/eval`). Torch-free users must obtain pre-converted MLX safetensors (see Weight distribution below) — `musetalk-mlx-convert` requires a torch env and cannot run torch-free.
3. **fusion-mlx**: install editable from its checkout (`pip install -e /path/to/fusion-mlx[video]`), pinned `>=0.10.2,<0.11`. Start/stop the service with `/path/to/fusion-mlx/start.sh start|stop`.
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

## Acknowledgements

This project is an **MLX port** of [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)
(1.5 lip-sync digital human) for Apple Silicon. The neural architecture (12-channel
audio-conditioned UNet, SD-VAE, Whisper-tiny, DWPose landmarks, face-parsing paste-back)
follows the original MuseTalk design; the models, weights, and inference logic are
reimplemented on [MLX](https://github.com/ml-explore/mlx) with a torch-free runtime.

- Original MuseTalk repo: <https://github.com/TMElyralab/MuseTalk>
- Neural foundation: [fusion-mlx](https://github.com/dahai80/fusion-mlx)
- Demo assets (base video/audio) originate from the MuseTalk `data/` set.
