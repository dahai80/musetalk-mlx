# Developer Guideline — musetalk-mlx-cases

How to add cases, configure options, fetch assets, and debug the demo runner.

## Prerequisites

1. **Parent repo set up** (`musetalk-mlx`, one level up):
   ```bash
   cd ..
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. **Converted MLX weights** in `../weights-mlx` (run `musetalk-mlx-convert` from a torch env, or `musetalk-mlx-download` for a pre-converted bundle — see parent README).
3. **ffmpeg/ffprobe** on PATH (`brew install ffmpeg`).
4. **Demo assets** fetched into `data/` (see [Asset protocol](#asset-protocol)).

## Asset protocol

Demo video/audio come from the MuseTalk repo's `data/` directory. They are **not** tracked in git (large binaries). `scripts/fetch_assets.sh` symlinks them:

```bash
bash scripts/fetch_assets.sh /path/to/MuseTalk   # default: ../MuseTalk
```

If you don't have the MuseTalk clone, fetch from HuggingFace mirror:
```bash
# MuseTalk data is on hf-mirror.com (https://hf-mirror.com/TMElyralab/MuseTalk)
# Download data/video/*.mp4 and data/audio/*.wav into ./data/{video,audio}/
```

Assets used by `inference_cases.yaml`:
- `data/video/`: musk, man, monalisa, sit, sun1, sun2, video1, yongen, yongen_img, sun_amp
- `data/audio/`: yongen.wav, eng.wav, sun.wav

`sun_amp.mp4` is generated from `sun.mp4` via `amplify_eyes.py` (see [Eye amplification](#eye-amplification)).

## Adding a case

1. Drop your base video + audio into `data/video/` and `data/audio/`.
2. Add a task block to `configs/inference_cases.yaml` (or a new config):
   ```yaml
   case_mynew:
     video_path: "data/video/mynew.mp4"
     audio_path: "data/audio/myvoice.wav"
     result_name: "case_mynew.mp4"
   ```
3. Run:
   ```bash
   ./run.sh --inference-config configs/inference_cases.yaml --only case_mynew
   ```
4. Output lands in `results/case_mynew.mp4`.

### Static-image cases

MuseTalk drives only the mouth; blink/body motion comes from the base video. For a static image, first make a motion base video with a Ken Burns zoom:

```bash
ffmpeg -loop 1 -i data/video/myimg.jpg -t 5 -vf "zoompan=z='min(zoom+0.0015,1.08)':d=125:s=720x720:fps=25" \
  -c:v libx264 -pix_fmt yuv420p data/video/mynew.mp4
```

Then add it as a case as above.

## Options

### bbox_shift

Face-crop vertical offset (MuseTalk `upperbondrange`). Positive shifts the crop down, negative up. Use when the default crop misses the mouth (receding hairline, tilted head). The `sun2` case uses `-7`:

```yaml
case_sun2:
  video_path: "data/video/sun_amp.mp4"
  audio_path: "data/audio/sun.wav"
  result_name: "case_sun2.mp4"
  bbox_shift: -7
```

Applied before `MuseTalkSession.__init__` (patches `FaceCropper` in the session module) — changing `session._cropper` after init has no effect because bg precompute already ran.

### fps

Auto-detected from the base video via ffprobe. Override only if detection fails:
```bash
./run.sh  # render_worker auto-probes
# manual:
../.venv/bin/python render_worker.py --video V --audio A --out O --fps 25 --mlx-dir ../weights-mlx
```

### Env overrides

| Variable | Default | Purpose |
|---|---|---|
| `MT_MLX_DIR` | `../weights-mlx` | converted MLX weights dir |
| `MT_MLX_PYTHON` | `../.venv/bin/python` | python with musetalk-mlx installed |
| `MT_BATCH` | `2` (forced `1` by inference.py) | render batch depth; `1` = stable for long audio |

## Eye amplification

`amplify_eyes.py` — simplified Eulerian video magnification on the eye ROI (height 25%-45%, width 20%-80%). Amplifies blink motion so it's visible in the output (MuseTalk doesn't generate blinks).

```bash
../.venv/bin/python amplify_eyes.py --in data/video/sun.mp4 --out data/video/sun_amp.mp4 --alpha 2.5
```

- `--alpha`: amplification factor. 2.5 ≈ 2× yongen's blink amplitude. Higher = more visible but artifacts appear above ~4.
- Streams frame-by-frame (no full-video OOM); rolling 8-frame low-pass approximates the temporal baseline.
- Only needed for base videos with weak blinks. yongen/musk/etc. don't need it.

## Debugging

### Render abort (`There is no Stream(gpu, 0)`)

Expected at session teardown — isolated to the `render_worker.py` subprocess. If it happens mid-render (output < expected frames):
- Check GPU is not contended: `ps aux | grep -iE 'python|mlx'` — stop other MLX processes.
- `MT_BATCH=1` is already forced; if still failing, reduce audio length or render in 30s segments.
- ffmpeg stderr: `/tmp/*_ffmpeg_err.log` (worker writes there).

### Invalid output video

`inference.py` validates via ffprobe (`duration > 0`); a torn write is detected and retried. If all 3 attempts fail:
- Check disk space.
- Check ffmpeg logs.
- Run `render_worker.py` directly to see the full traceback before `os._exit`.

### Wrong mouth position

Adjust `bbox_shift` (see above). Run a single frame to check:
```bash
../.venv/bin/python -c "
from musetalk_mlx import MuseTalkSession
s = MuseTalkSession(None, 'data/video/mynew.mp4', mlx_dir='../weights-mlx')
# inspect s._cropper.upperbondrange + first crop
"
```

## Regenerating README assets

GIFs/thumbnails in `assets/` are tracked. Regenerate after re-running cases:

```bash
# thumbnails (jpg) from result videos
for f in results/case_*.mp4; do
  name=$(basename "$f" .mp4)
  ffmpeg -y -loglevel error -i "$f" -vframes 1 -vf "scale=320:-1" "assets/cases/${name}.jpg"
done

# gifs (looping, compressed)
for f in results/case_yongen.mp4 results/case_musk.mp4; do
  name=$(basename "$f" .mp4)
  ffmpeg -y -loglevel error -i "$f" -t 4 -vf "fps=6,scale=240:-1,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer" "assets/cases/${name}.gif"
done
```

Keep GIFs under ~600KB (use `fps=6 scale=240` + 64-color palette).
