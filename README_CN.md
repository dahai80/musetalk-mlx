# musetalk-mlx

MuseTalk 1.5 唇形同步数字人，运行于 Apple Silicon。K12 英语外教实时数字人的**业务层**，构建在 **fusion-mlx** MLX 底座之上。规格文档：[`../arch/musetalk-mig-mlx-v2-0918.md`](../arch/musetalk-mig-mlx-v2-0918.md)（PRD V1.1-RC2）。

## 与 fusion-mlx 的分工

| 层 | 位置 | 内容 |
|---|---|---|
| 神经核心 | `fusion-mlx` → `fusion_mlx.video.musetalk_mlx` | UNet、SD-VAE、Whisper-tiny、`MuseTalkPipeline`、权重加载器 |
| 业务 | 本仓库 | 人脸处理、实时会话、温控降级、LiveKit 适配器、工具 |

神经核心作为依赖被消费——本仓库不重复实现任何模型代码。

## 推理管线

```
16k 单声道 PCM -> 5s 滑窗（33ms 步长）-> Whisper-tiny 音频 embedding（50Hz）
底片视频帧 -> 5 点关键点 -> 256x256 crop -> VAE encode
-> UNet(t=0, 音频 cross-attn) -> VAE decode -> 羽化贴回
-> 携带音频 PTS 的输出帧 -> LiveKit
```

## 硬指标（PRD）

30 FPS · 音视频 RTT ≤ 80ms · 统一内存 ≤ 4GB · macOS 14+ · 运行时无 PyTorch。

## 安装

```bash
cd musetalk-mlx
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`fusion-mlx` 依赖通过 `file://` 指向 `~/fusion/fusion-mlx`（见 `pyproject.toml`）。

### 权重

期望目录结构（匹配 `MuseTalkPipeline.from_pretrained`）：

```
weights/
├── sd-vae-ft-mse/                  # stabilityai/sd-vae-ft-mse
├── MuseTalk/musetalkV15/unet.pth   # TMElyralab/MuseTalk
└── whisper-tiny/                   # openai/whisper-tiny
```

通过 https://hf-mirror.com 下载，或用 `musetalk-mlx-convert` 从 MuseTalk 克隆的 `models/` 布局转换。

## 命令

| 命令 | 用途 |
|---|---|
| `musetalk-mlx-convert --weights ... --out ...` | PyTorch 权重 → MLX safetensors 发行目录 |
| `musetalk-mlx-offline --weights ... --audio ... --video ... --out ...` | 离线：音频 + 底片视频 → 输出视频 |
| `pytest tests/ -v` | 运行测试 |
| `ruff check .` | lint |

## 阶段状态

- [x] Phase 1（部分）：神经核心已在 fusion-mlx v0.2.0；DWPose/卡尔曼待补
- [ ] Phase 2：音频流式打磨（前缀缓存）、零拷贝、离线 Demo
- [ ] Phase 3：图优化 + ICB + LiveKit 实时
- [ ] Phase 4：可选 LCM 蒸馏

## 目录结构

```
musetalk_mlx/
├── config.py          # 管线常量（PRD 门限）
├── face/              # 关键点（卡尔曼、待机兜底）+ 256 crop
├── pipeline/          # MuseTalkSession + 羽化融合
├── livekit/           # phase3 适配器（可选 extra）
├── utils/             # 音频滑窗 + 温控分级
└── tools/             # 权重转换 + 离线 CLI
```
