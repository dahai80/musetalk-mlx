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
底片视频帧 -> 68 点 DWPose 人脸关键点 -> 256x256 crop -> VAE encode
-> UNet(t=0, 音频 cross-attn) -> VAE decode -> face-parse 掩码贴回
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
| `musetalk-mlx-realtime --weights ... --video ... --livekit-url ... --token ...` | 实时：流式 → LiveKit（FR-LK-001/002） |
| `pytest tests/ -v` | 运行测试 |
| `ruff check .` | lint |

## DWPose / 人脸关键点

`LandmarkTracker` 消费 68 点 DWPose 人脸关键点（COCO-WholeBody `[23:91]`，
与 MuseTalk `preprocessing.get_landmark_and_bbox` 同约定），通过
`derive_face_bbox` 推导轴向对齐 crop bbox。关键点用向量化常速度卡尔曼滤波平滑；
单帧检测丢失则保持上一帧姿态，连续 `KEYPOINT_FAIL_IDLE` 帧丢失进入待机眨眼
（原样输出底片帧）。

MLX DWPose/RTMPose 推理后端**尚未在 fusion-mlx 中提供**（已在 dahai80/fusion-mlx
提 issue 跟踪）。`load_dwpose_backend()` 探测 `fusion_mlx.video.dwpose`，缺失时回退
待机眨眼，保证管线可运行。bbox 数学、卡尔曼平滑与待机逻辑由 `tests/test_landmarks.py`
覆盖，不依赖模型。

## 阶段状态

- [x] Phase 1（部分）：神经核心已在 fusion-mlx v0.2.0；DWPose bbox 数学 + 卡尔曼 + 待机眨眼已接入；MLX DWPose 后端待 fusion-mlx #909
- [x] Phase 2（业务层）：重叠音频滑窗（前缀平滑）、生产级融合贴回 + 可插拔 face-parse 掩码、零拷贝帧汇（拷贝兜底）、离线 Demo 打磨。embedding 级前缀缓存待 fusion-mlx #914；face-parse 模型待 #910；真零拷贝待 #913。
- [x] Phase 3（业务层）：完整温控降级阶梯（FR-END-003）、ReloadModel 热重载（FR-MLX-006）、LiveKit 实时适配器（FR-LK-001/002，音频继承 PTS，双向音频）、实时运行 CLI。图优化 + ICB 性能待 fusion-mlx #911/#912。
- [x] Phase 4（桩）：LCMFastSession 配置桩（默认关闭；主版本用多步 DDIM）。蒸馏训练不在本仓库范围。
- [ ] 集成测试：神经核心已验证（7 个集成测试通过）；DWPose/face-parse 权重加载被 fusion-mlx #915 阻塞 — 见 `tests/integration/INTEGRATION_PENDING.md`。

## fusion-mlx 依赖 issue

| Issue | 能力 | 阶段 |
|---|---|---|
| [#909](https://github.com/dahai80/fusion-mlx/issues/909) | DWPose/RTMPose MLX 人脸关键点 | 1 |
| [#910](https://github.com/dahai80/fusion-mlx/issues/910) | Face-parsing BiSeNet MLX | 2 |
| [#911](https://github.com/dahai80/fusion-mlx/issues/911) | Conv+GN+SiLU 图熔合 + SafeGroupNorm | 3 |
| [#912](https://github.com/dahai80/fusion-mlx/issues/912) | Metal ICB 批量编码 | 3 |
| [#913](https://github.com/dahai80/fusion-mlx/issues/913) | IOSurface↔CVPixelBuffer 零拷贝 | 3 |
| [#914](https://github.com/dahai80/fusion-mlx/issues/914) | encode_audio 前缀上下文缓存 | 2 |
| [#915](https://github.com/dahai80/fusion-mlx/issues/915) | convert_dwpose/convert_face_parsing 键名不匹配 + mel 资源未打包 | 1/2 |

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
