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
base video frame -> 5-pt landmarks -> 256x256 crop -> VAE encode
-> UNet(t=0, audio cross-attn) -> VAE decode -> feather paste-back
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
| `pytest tests/ -v` | run tests |
| `ruff check .` | lint |

## Phase status

- [x] Phase 1 (partial): neural core in fusion-mlx v0.2.0; DWPose/Kalman TODO
- [ ] Phase 2: audio streaming polish (prefix cache), zero-copy, offline demo
- [ ] Phase 3: graph pass + ICB + LiveKit realtime
- [ ] Phase 4: optional LCM distillation

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
