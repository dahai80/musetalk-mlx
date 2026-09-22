# musetalk-mlx-cases

Demo case runner for **musetalk-mlx** — reproduces the [MuseTalk](https://github.com/TMElyralab/MuseTalk) "TestCases For 1.0" suite on Apple Silicon (MLX, no PyTorch runtime). YAML-driven multi-task inference: audio + base video → lip-synced output video (audio muxed via ffmpeg).

Lives inside the `musetalk-mlx` repo as a subdirectory and consumes its venv + converted weights via relative paths (`../.venv`, `../weights-mlx`).

## Cases (reproduce MuseTalk README)

Each row = input image → musetalk-mlx lip-synced output. Videos are hosted on the
[demo-v1 release](https://github.com/dahai80/musetalk-mlx/releases/tag/demo-v1) and
rendered inline (playable, same as the MuseTalk README).

<table>
<tr><th>Input</th><th>Output (musetalk-mlx)</th></tr>
<tr>
<td><img src="assets/inputs/yongen.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_yongen.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/musk.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_musk.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/monalisa.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_monalisa.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/sit.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_sit.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/man.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_man.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/sit.jpg" width="200"><br><sub>sun1 (blink amplified)</sub></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_sun1.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/sit.jpg" width="200"><br><sub>sun2</sub></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_sun2.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="assets/inputs/video1.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_video1.mp4" controls preload></video></td>
</tr>
</table>

Full result videos also regenerable locally: `results/*.mp4` (not tracked in git).

## Quick start

```bash
# 1. Parent repo must be set up first (venv + converted weights):
cd ..            # musetalk-mlx repo root
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# convert or download weights into ../weights-mlx (see parent README)

# 2. Back here, fetch the demo assets (MuseTalk data, via hf-mirror):
cd musetalk-mlx-cases
bash scripts/fetch_assets.sh      # symlinks data/video, data/audio

# 3. Run all cases:
./run.sh
# or a single case:
./run.sh --inference-config configs/inference_cases.yaml --only case_musk
```

## Layout

```
inference.py            # entry: yaml tasks → subprocess render → ffmpeg mux
render_worker.py        # render worker (subprocess, isolates C++ teardown abort)
amplify_eyes.py         # eye-motion amplification for low-blink base videos
configs/
  inference.yaml        # 2-task smoke config (yongen×yongen, yongen×eng)
  inference_cases.yaml  # full 8-case MuseTalk reproduction config
data/{video,audio}/     # assets (symlinked, not tracked)
results/                # output mp4 (regenerable, not tracked)
assets/                 # README thumbnails/gifs (tracked)
```

## Config format

```yaml
case_musk:                       # task id (arbitrary)
  video_path: "data/video/musk.mp4"   # relative to repo root
  audio_path: "data/audio/yongen.wav"
  result_name: "case_musk.mp4"        # output filename in results/
  bbox_shift: -7                 # optional: face crop y-shift (sun2 case)
```

## How it works

1. **`inference.py`** reads the YAML, runs each task via `render_worker.py` as a subprocess (3 retries on abort), then muxes source audio into the silent render via ffmpeg.
2. **`render_worker.py`** drives `MuseTalkSession`: chunked audio push (keeps the ~20s buffer under cap), ffmpeg rawvideo pipe output (abort-safe — a C++ teardown abort triggers EOF, ffmpeg still finalizes a valid file), and `os._exit` past `session.close()` (the close path hits the same abort).
3. **`amplify_eyes.py`** (optional, for sun1/sun2): simplified Eulerian magnification of the eye ROI so blinks are visible — MuseTalk only drives the mouth, blink/body motion comes from the base video.

## Known limitations (musetalk-mlx upstream)

- **C++ teardown abort**: `MuseTalkSession`'s render thread can trigger a `There is no Stream(gpu, 0)` abort at session close. Isolated to the subprocess; output is still valid. Underlying cause tracked in the parent repo.
- **Static-image motion**: MuseTalk drives only the mouth. Official cases use MuseV (CUDA) to generate motion video; without MuseV, static-image cases get a ffmpeg `zoompan` Ken Burns push (1.0→1.08) for overall motion. The person (blink, limbs) stays static — expected.
- **Long audio (60s+)**: render on a dedicated GPU to avoid other MLX processes contending and raising abort probability. `MT_BATCH=1` is forced for stability.
- **MLX version**: pinned to `0.32.0` (parent repo requirement; 0.32.2 has a decode-kernel regression on the joint compile graph).

## Developer guideline

See [`GUIDELINE.md`](GUIDELINE.md) for adding cases, configuring bbox-shift, tuning amplification, and the asset-fetch protocol.
