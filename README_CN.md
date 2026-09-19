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

MLX DWPose/RTMPose 后端已随 fusion-mlx 提供，但其解码关键点当前损坏（上游 #917），
且坐标在网络输入空间（#916 — 本地已重缩放）。`load_dwpose_backend()` 包一层坐标
换算，`LandmarkTracker` 的 bbox 合理性守卫在异常时降级为待机眨眼，避免产出坏
crop。bbox 数学、卡尔曼平滑、守卫与待机逻辑由 `tests/test_landmarks.py` 覆盖，
不依赖模型。

## 阶段状态

- [x] Phase 1（部分）：神经核心已在 fusion-mlx v0.2.0；DWPose bbox 数学 + 卡尔曼 + 待机眨眼已接入；MLX DWPose 后端待 fusion-mlx #909
- [x] Phase 2（业务层）：重叠音频滑窗（前缀平滑）、生产级融合贴回 + 可插拔 face-parse 掩码、零拷贝帧汇（拷贝兜底）、离线 Demo 打磨。embedding 级前缀缓存待 fusion-mlx #914；face-parse 模型待 #910；真零拷贝待 #913。
- [x] Phase 3（业务层）：完整温控降级阶梯（FR-END-003）、ReloadModel 热重载（FR-MLX-006）、LiveKit 实时适配器（FR-LK-001/002，音频继承 PTS，双向音频）、实时运行 CLI。图优化 + ICB 性能待 fusion-mlx #911/#912。
- [x] Phase 4（桩）：LCMFastSession 配置桩（默认关闭；主版本用多步 DDIM）。蒸馏训练不在本仓库范围。
- [ ] 集成测试：神经核心已验证（7 个集成测试通过）；#915 修复已验证（权重严格加载 + mel 打包）；真实关键点唇形对齐被 fusion-mlx #917（DWPose 输出损坏）阻塞 — 见 `tests/integration/INTEGRATION_PENDING.md`。

## 性能（M5 Max）

测量前提：fusion-mlx 服务器与其它 GPU 消费者全部停止 —— 后台 GPU 负载会把
数字抬高数倍（详见 `tests/integration/INTEGRATION_PENDING.md`）。

| 优化 | 效果 |
|---|---|
| fp16 转换 + 离线预计算（landmarks/bbox/VAE latent） | DWPose + encode 移出热循环 |
| UNet+VAE-decode 联合 `mx.compile`（单图） | 每批 2 帧的 round 178ms → 82ms（2.2 倍）；分离编译有 compiled-input 边界开销 |
| `mx.metal.set_cache_limit(1GB)` + `set_memory_limit(3GB)` | allocator cache 原本涨到 13.5GB、渲染尖峰 >1s；现 p90 304ms，RSS 17GB → 4.2GB |
| 批 2 热路径（`config.BATCH`） | 每 2 步一次 UNet+decode，RTT 安全（多等一个 33ms 步） |

隔离干净 GPU 微基准（fp16）：UNet+VAE-decode 联合 batch-2 = 82ms/round
（41ms/帧）。本机持续满载 E2E 低于此值（长期后台 GPU 负载；MLX conv2d kernel
悬崖，上游 #919 —— VAE 256x256 解码 ~58ms/帧，理论应可达 ~20ms）。30FPS 需要
上游 kernel 工作（#918/#919）；消费侧上述杠杆已用尽。

## PRD V1.1-RC2 差距补齐（本次发布）

- fusion-mlx v0.10.2 验证（#916/#917）：真实关键点链路全绿 —— 10/10 检出、bbox 合理；消费侧 `_FrameSpaceDWPose` workaround 已移除。
- 图优化 Pass 接入修复：改为编译纯 UNet 前向（`generate_faces` 内含 `mx.eval`，在 `mx.compile` 中非法）；输出与 plain 一致（cosine 1.0）。
- fp16 管线转换（`config.FP16`）+ MuseTalk-realtime 式离线预计算（每底片帧缓存 landmarks/bbox/VAE latent，`config.PRECOMPUTE`），热循环移除 DWPose + encode。

- 对 PyTorch 的分层精度 parity 测试（`tests/parity/`）：Whisper 编码器、VAE encode（cosine ≥ 0.98）、VAE decode（PSNR ≥ 38dB）、8 通道 UNet（cosine ≥ 0.98）。fixture 在 torch 环境一次性生成（`musetalk_mlx/tools/gen_parity_fixtures.py`），测试时无 torch 依赖。
- 单帧关键点丢失走 Kalman 预测（FR-END-006）：位置+速度外推，连续 5 帧丢失才进 idle-blink；含 bbox 合理性 + 离群点剔除守卫。
- 温控降级阶梯全链路接入 session（FR-END-003）：砍步数 → 帧复用(3) → 背景降采样(2) → patch 256→128（绝不直接 256→128）。
- 底片帧预加载内存池（FR-END-002），保留流式兜底。
- 通过 fusion-mlx `compile_with_custom_pass`（#911）消费图优化 Pass，探测式优雅降级。
- 帧输出接 `MetalZeroCopyBridge.array_to_cvbuffer`（#913）零拷贝，保留拷贝兜底。
- 分阶段 profiler（STFT/Whisper/UNet/VAE/warp/frame-out）、phys-footprint 内存采样、最大连续分配探测（`musetalk_mlx/utils/profiling.py`）。
- 运维工具：`musetalk-mlx-benchmark`（FPS + 分阶段预算，PRD 9.5）、`musetalk-mlx-stress`（长会话泄漏 ≤ 50MB + 碎片探测）。
- 边界输入测试：小于 10ms 音频、全静音、满幅削波。

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
| [#916](https://github.com/dahai80/fusion-mlx/issues/916) | DWPose 坐标为网络输入空间而非原始帧空间 | 1 |
| [#917](https://github.com/dahai80/fusion-mlx/issues/917) | 严格加载通过但 DWPose 输出损坏 | 1 |
| [#918](https://github.com/dahai80/fusion-mlx/issues/918) | `compile_with_custom_pass` 是空壳（模式从未真正应用） | 3 |
| [#919](https://github.com/dahai80/fusion-mlx/issues/919) | Metal conv2d fp16 吞吐在不同形状间差距达 8 倍 | 3 |
| [#920](https://github.com/dahai80/fusion-mlx/issues/920) | 默认 allocator cache 无界增长，渲染出现秒级尖峰 | 3 |

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
