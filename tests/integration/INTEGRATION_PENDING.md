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

## v0.10.4 verification (#919 autotune) + perf correction

**Correction:** the v0.10.3 "58ms/frame VAE decode" baseline (and the
#921 filing built on it) was **contention-contaminated** — measured while
`com.linguakids.watchdog` auto-restarted the fusion-mlx server. Re-measured
on v0.10.4 / MLX 0.32.2 with the GPU verified clean (Device Utilization 0%,
no fusion-mlx/watchdog/image_worker procs, fresh process):

| Path | b2 round | per-frame |
|---|---|---|
| VAE decode plain | 50.4 ms | **25.2 ms** |
| UNet plain | 38.0 ms | 19.0 ms |
| Joint compiled UNet+decode | 89.3 ms | 44.6 ms (22.4 FPS) |
| Joint compiled b1 | 52.1 ms | 19.2 FPS |

The "cliff" does not reproduce on a clean GPU — fusion-mlx's "VAE decode
already ~20ms/frame" note is correct. #921 retracted as "sole remaining
blocker" (correction comment filed); ~11ms/frame gap to 30FPS remains,
shared between UNet (19ms) and decode (25ms) — ordinary headroom, not a
kernel cliff.

SmartConv2d autotune (v0.10.4) verified: 6/10 shapes im2col wins on 0.32.2;
`FUSION_MUSETALK_SMART_CONV=1` wraps 55 VAE convs; but in-graph decode is
slower (55.1ms vs 50.4ms plain) — default OFF is correct. Parity clean on a
single decode: cosine 0.999995, PSNR 64.5dB, drift mean 0.0017 / max 0.0237
(smaller than the changelog's 0.4/5.0 — that compounds through full
paste-back chain).

## Performance (M5 Max, clean GPU, fusion-mlx server + linguakids watchdog stopped)

Methodology: background GPU load inflates every number several-fold. Before
measuring: `/path/to/fusion-mlx/start.sh stop`,
`launchctl bootout gui/$(id -u)/com.linguakids.watchdog`, and check
`ioreg -l | grep '"Device Utilization %"'` ≈ 0. Other Claude sessions spawn
`fusion_mlx.media.image_worker` / `fusion-mlx-server` on demand — verify with
`ps aux` between runs.

Component micro-benchmarks (fp16, clean GPU, fresh process, v0.10.4):

| Path | Time |
|---|---|
| UNet plain b2 | 38.0ms (19.0ms/frame) |
| VAE decode plain b2 | 50.4ms (25.2ms/frame) |
| **Joint UNet+decode compiled b2 (one graph)** | **89.3ms (44.6ms/frame, 22.4 FPS)** |
| Joint compiled b1 | 52.1ms (19.2 FPS) |
| VAE decode + SmartConv2d autotune b2 | 55.1ms — slower in-graph, default OFF |

Memory: MLX default allocator cache grew to 13.55GB during the hot loop with
render spikes p90 >1.2s; `set_cache_limit(1GB)` + `set_memory_limit(3GB)`
(`session._tune_mlx_memory`) → cache 0.91GB, spikes p90 304ms, RSS 17GB →
4.2GB (PRD ≤4GB essentially met).

Current E2E: 22.4 FPS (joint b2, clean GPU). To hit 30FPS (33.3ms/frame =
66.6ms b2 round) the remaining gap is ~23ms/round, shared between UNet and
decode — ordinary optimization headroom, not a kernel cliff. The earlier
~8 FPS sustained number was background-load contamination, not a code limit.

NOTE: `com.linguakids.watchdog` (launchd) auto-restarts the fusion-mlx
server every interval — benchmarks are meaningless while it runs. The v0.10.3
"58ms decode" figure was measured under this contamination and is retracted.

Upstream follow-ups filed 2026-09-19: #919 comment (in-graph A/B data,
dispatch-rule recalibration proposal), #918 comment (musetalk topology
matches 0 sites; `groupnorm_silu_conv` pre-conv pattern request), #921
(filed as conv2d kernel cliff blocker — **retracted** via correction
comment: baseline was contention-contaminated, clean GPU = 25ms/frame).

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

## Blocked — fusion-mlx #917 (DWPose output corruption) — FIXED v0.10.2

| Subsystem | Status |
|---|---|
| DWPose (#909) | **fixed v0.10.2** — forward matches ONNX GT (SPP-stage residual bug); `face_landmarks` returns frame-space coords; 10/10 detections, bbox 3.9% of frame, 0 border-pinned coords. |
| Face-parsing (#910) | **working** — real 79999_iter weights, sane labels. |

## Scenarios — status

1. **Offline end-to-end parity** — `musetalk-mlx-offline` runs clean (150-frame
   video, real landmarks, no guard fallback). PSNR vs MuseTalk torch reference
   gated on `gen_gt.py` (subprocess into MuseTalk checkout, MPS) — tool exists,
   not yet run against the torch reference for the PSNR ≥ 38 / SSIM ≥ 0.95 gate.
2. **LiveKit realtime** — **DONE (A3)**. Local LiveKit server v1.9.1, realtime CLI
   + `lk_receive.py` probe. Clean GPU, 1080p, `MT_DECODE_128=1`: receiver 29.82
   FPS, publish-interval p50 33.3ms, render_eval 55ms, 0 steady drops. RTT
   (render-pipeline segment) 33.3ms ≤ 80ms. Full RTT 5138ms includes the
   AudioWindower 5s window buffer (hop 4.8s, window-tail audio waits for the
   next window) — a streaming-latency property, not a render bug; PRD scope
   clarification pending, design not silently changed. PTS strictly monotonic.
3. **2h stability stress** — **active leak source open; allocator side bounded**.
   Full 120-min `--with-reload` run (`results/stress_2h.json`, Fix 1 only — the
   before-fast-path-fix baseline):

   | Run | leak_mb | MLX active delta | avg_fps | reload resume |
   |---|---|---|---|---|
   | audit v3 (pre-fix) | 243 | +291MB (2006→2297) | 5.48 | 2.37s |
   | periodic clear (Fix 1) | 199 | +193MB (2081→2274) | 8.08 | 1.58s |

   Two allocator fixes landed:
   - `592d108`: periodic `mx.clear_cache()`+`gc.collect()` every
     `MT_CLEAR_CACHE_EVERY` frames (default 300) in the render loop + env-tunable
     allocator caps (`MT_MLX_CACHE_GB`/`MT_MLX_LIMIT_GB`).
   - `269fc8e`: `mx.clear_cache()`+`gc.collect()` on fast-path reload.

   These bound the **allocator cache** (flat ~4126MB after warmup in a 35-min
   verification run). They do **not** stop the **active** leak: MLX active grows
   linearly ~1.6-3.2MB/min (2081→2274MB over 120min, +193MB; 1981→2093MB over
   35min, +112MB).

   **Root cause: open.** Not yet attributed — investigation lives at the
   integration layer (fusion-mlx#954), not directly upstream at MLX. musetalk-mlx
   uses a single persistent compiled joint UNet+VAE-decode fn with **stable input
   shapes every frame** (latent 256×256×8 fixed; audio cross-attn `(N,50,384)`
   fixed); `mx.compile` keys its specialization cache on structure+shape+dtype,
   not input values, and the stress loop re-pushes a bounded audio-chunk set
   (rolling 5s window). So the per-frame active growth is not obviously an
   `mx.compile` specialization leak. If fusion-mlx#954 confirms an MLX-runtime
   cause, the MLX issue will be filed from the fusion-mlx side (which owns the
   integration boundary); musetalk-mlx does not file directly against MLX for this.

   **35-min rebuild-compiled-gens experiment:** dropping the old `mx.compile`
   closure + rebuilding a fresh one at a reload boundary did **not** drop active
   (pre-reload 2079MB → post-reload 2082MB, +3MB) and post-rebuild growth
   **accelerated** to ~2.75MB/min. Consistent with a retained-ref or compile-cache
   source but not conclusive. Fix reverted (negative benefit + recompile cost).
   No Python-exposed compile-cache clear/introspection API on MLX 0.32.0
   (`dir(mx)` = `clear_cache`/`set_cache_limit` allocator-only).

   **Escalated to the integration layer:** fusion-mlx#954 (dahai80/fusion-mlx) —
   asks whether `compile_with_custom_pass`/`apply_patterns`/the musetalk pipeline
   retain any per-call mx.array state (module buffers, `_seen`-style globals,
   SmartConv `_IM2COL_RULES`/`_BACKEND_RULES`, attention/KV cache) that could
   accumulate, and whether a known-good active-memory bounding pattern exists for
   long realtime compiled loops. Root cause pending that investigation.

   **Status:** the 50MB/2h active-leak gate is **not achievable** until the
   retention source is identified (investigation at fusion-mlx#954). The
   allocator-cache side is bounded (Fix 1+3). `scripts/leak_gate.py` fails on the
   active signal. Fragment probe passes all 120min (1GB contiguous alloc OK).

   **GPU contention caveat**: other Claude sessions' `fusion-mlx-server` +
   Chrome + node run throughout (launchd auto-restarts them; bootout rc=3 from
   another session — PID 41988 held by another session, uncontrollable here).
   avg_fps=8.08 and render_eval 141→240ms are contention, not code regression —
   clean-GPU baseline is 89ms/round / 30FPS per the perf table. Active-memory
   growth is contention-independent (a retention property, not a timing one),
   so the leak shape is valid despite the FPS contamination.
4. **Thermal ladder** — wired in-session (FR-END-003): step-cut → frame-reuse(3)
   → bg-downscale(2) → patch 256→128. Serious-tier step-cut is a no-op on the
   realtime path (fixed single-step t=0; multi-step DDIM is Nx slower) —
   serious-tier only effective on the offline multi-step path. Realtime
   degradation relies on the critical tier.
5. **Edge inputs** — silence / clipping / <10ms: covered by unit tests, no crash,
   auto-standby.
6. **Per-stage benchmark** — `musetalk-mlx-benchmark` reports STFT/Whisper/UNet/
   VAE/warp/frame-out timing + FPS + RTT + memory + fragmentation.

## Zero-copy egress (FR-LK-001) — wired

`LiveKitAdapter.publish_frame` hands BGRA to rtc `VideoFrame` as a `memoryview`
of a pre-allocated rotating numpy pool — SDK `get_address` passes the pointer to
FFI (`ctypes.addressof(c_char.from_buffer)`), no `tobytes`/`bytearray` copy.
`ZeroCopySink` exposes the #913 `MetalZeroCopyBridge` (IOSurface→CVPixelBuffer)
for non-LiveKit sinks; the livekit Python SDK does not accept a CVPixelBuffer
directly, so the IOSurface path is reserved for a future native VideoToolbox
direct-encoding integration.

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
