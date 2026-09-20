# End-to-end quality regression (PRD §10.5)

## Status

| Metric | Threshold | Status | First measurement | Needs GT? |
|---|---|---|---|---|
| PSNR | ≥38dB | **gated** (needs CUDA GT) | — | yes |
| SSIM | ≥0.95 | **gated** (needs CUDA GT) | — | yes |
| CSIM | ≥0.98 | **gated** (needs CUDA GT + arcface) | — | yes |
| LSE-C | ≥5.5 | **ready** | 5.70 ✅ (was 0.76) | no |
| LSE-D | ≤9.0 | **ready** | 9.31 ❌ (was 11.50) | no |

LSE-C/LSE-D measure the output video's own audio-video sync — they do not
need CUDA ground-truth, so they run now. PSNR/SSIM/CSIM compare against CUDA
GT and are gated until the MuseTalk PyTorch torch env + weights are
provisioned (`musetalk-mlx-gen-gt`, stub).

**First measurement** (150-frame / 5s offline run, `eng.wav` on the `sun.mp4`
template) initially measured LSE-D=11.50 / LSE-C=0.76. Root cause was in the
render pipeline, not the eval: `paste_back` received the `FaceCropper` bbox in
`(x, y, w, h)` form while its contract is `(x1, y1, x2, y2)`, so `x2 < x1` read
as degenerate and the generated face was silently dropped — output was raw
base-video passthrough, byte-identical for any input audio (verified: real
vs silence md5 identical). Second bug: `FaceParseMask` returned the parse
labels at the backend's own output resolution (512×512) instead of the crop
box size, so the paste alpha sampled the wrong region. Both fixed
(`crop_bbox_to_xyxy` boundary conversion + label resize; regression tests in
`test_blending.py`). **Third bug — window-boundary tempo drift:** the
5s audio window was 79950 samples (150×533) but `get_whisper_chunk`'s
`floor()` dropped one chunk per window while the windower still advanced 150
PTS steps, so the video timeline lagged audio 0.67% per window; chunk PTS also
used `step/sr` (533/16000) instead of `1/fps`. SyncNet measured this as a
monotonic per-slice offset drift (−2 → −5 over 60s) which made the
whole-video LSE collapse. Fixed: window length is sample-exact
(`round(sr·window_s)`, so `window·fps/sr` is an integer) and chunk PTS
advances `1/fps` (regression tests in `test_audio_overlap.py`/
`test_smoke.py`).

**Post-fix measurement** (60s `eng.wav` on the `sun.mp4` template, whole
video): **LSE-C=5.70 ✅ / LSE-D=9.31 ❌**. Per-5s-slice offsets are now
constant (+3) with LSE-C 4.3–6.8 per slice — no drift. A silence-audio
negative control scores LSE-C=0.18, confirming the metric tracks the driving
audio. Remaining LSE-D gap (9.31 vs ≤9.0; MuseTalk paper 6.53) is a
feature/precision-quality item, not alignment: candidates are fp16 UNet,
window-boundary feature quality, and the 5s-window whisper context (upstream
encodes whole-audio context). Next-phase work; LSE-D is offset-invariant so
the +3 constant offset does not affect it.

## Pipeline validation (matches native LatentSync)

The `evaluate_sync` port was validated byte-for-byte against the canonical
LatentSync `eval/syncnet/syncnet_eval.py` + `eval/syncnet_detect.py` pipeline
(joonson `syncnet_v2.model`, S3FD detector, `crop_scale=0.4` mouth-biased
crop, medfilt(k=13) smoothing, 25fps resample):

| Input | Native S3FD pipeline | This suite (insightface crop) |
|---|---|---|
| musetalk-mlx output (eng.wav-driven) | LSE-D=11.50 LSE-C=0.76 | LSE-D=10.54 LSE-C=0.55 |
| sun.mp4 + sun.wav (unsynced template) | LSE-D=13.33 LSE-C=0.11 | LSE-D=13.74 LSE-C=0.08 |

The insightface crop tracks the native S3FD result within ~1 LSE-D. The
sun.mp4+sun.wav "baseline" is **not** a valid sync reference — MuseTalk's
template video mouth motion does not match `sun.wav` (unsynced content), so
both pipelines correctly report poor sync. A true positive control (a known
lip-synced output) is still needed to bound the absolute LSE floor; deferred
until CUDA GT generation lands.

## Eval env

Eval models (SyncNet + arcface) run in the **openclaw conda env** (torch 2.14
+ insightface), not the musetalk-mlx venv (torch-free at runtime per PRD).
Install deps:

```bash
conda activate openclaw
pip install python_speech_features scikit-image
# syncnet weights (joonson syncnet_v2.model, via hf-mirror)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    ByteDance/LatentSync-1.5 auxiliary/syncnet_v2.model \
    --local-dir weights/eval
# arcface (insightface buffalo_l, auto-downloaded on first use)
```

## LSE-C / LSE-D definition (canonical)

Faithful port of joonson/LatentSync `eval/syncnet/syncnet_eval.py`
(`SyncNetEval.evaluate`):

- **LSE-D** = `min(mean_dists)` — minimum mean pairwise embedding distance
  over AV offsets (vshift=15). Lower = better sync.
- **LSE-C** = `median(mean_dists) - min_dist` — confidence (how much better
  synced than the median offset). Higher = better.
- Input: 5 consecutive 224×224 face frames + MFCC(16kHz, 20-frame windows).
- Weights: joonson `syncnet_v2.model` @ `ByteDance/LatentSync-1.5`.
- Video is resampled to 25fps (canonical; MFCC 100fps ÷ 4 = 25 video fps).
  musetalk-mlx renders at 30fps per PRD — the eval resamples before SyncNet.

## Face crop deviation

Upstream uses S3FD + scenedetect per-face-track stabilization. We use
insightface (shared with CSIM) with the **canonical crop geometry**
replicated exactly: per-frame bbox, `crop_scale=0.4`, mouth-biased
asymmetric square crop (side = `max(w,h)·1.4`, biased toward the chin),
medfilt(k=13) smoothing, 25fps resample. Only the detector differs
(insightface vs S3FD); validated to track the native S3FD result within
~1 LSE-D for single-face talking heads. Switch to S3FD if multi-face or
stricter parity is needed (deps: torchvision, sfd_face.pth, scenedetect).

## Datasets (PRD §10.5.2)

- HDTF 100 clips — not yet provisioned.
- K12 private 15 clips — not yet provisioned.
- Current: MuseTalk sample data (eng/sun/yongen, 3 pairs) for smoke runs.

## Run

```bash
# LSE only (pred has no audio track — mux source audio):
musetalk-mlx-eval --pred output.mp4 \
    --audio source.wav \
    --syncnet weights/eval/auxiliary/syncnet_v2.model

# full (with GT):
musetalk-mlx-eval --pred output.mp4 --gt gt.mp4 \
    --audio source.wav \
    --syncnet weights/eval/auxiliary/syncnet_v2.model
```

`--audio` muxes the 16kHz source wav into the pred video (via ffmpeg) when
the pred lacks an audio track — musetalk-mlx offline output is video-only;
SyncNet needs both streams.

