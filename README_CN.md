# musetalk-mlx

[MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) 唇形同步数字人的 **MLX 移植版**，运行于 Apple Silicon，运行时无 PyTorch。K12 英语外教实时数字人的**业务层**，构建在 **fusion-mlx** MLX 底座之上。规格文档：[`../arch/musetalk-mig-mlx-v2-0918.md`](../arch/musetalk-mig-mlx-v2-0918.md)（PRD V1.1-RC2）。

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

## 演示

来自 [`musetalk-mlx-cases`](musetalk-mlx-cases/) 的唇形同步输出，在 MLX 上复现
[MuseTalk](https://github.com/TMElyralab/MuseTalk) 1.0 TestCases（运行时无 PyTorch）。
视频托管在 [demo-v1 release](https://github.com/dahai80/musetalk-mlx/releases/tag/demo-v1)，下方内联播放。

<table>
<tr><th>输入</th><th>输出（musetalk-mlx）</th></tr>
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
<td><img src="musetalk-mlx-cases/assets/inputs/sit.jpg" width="200"><br><sub>sun1（眨眼放大）</sub></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_sun1.mp4" controls preload></video></td>
</tr>
<tr>
<td><img src="musetalk-mlx-cases/assets/inputs/man.jpg" width="200"></td>
<td><video src="https://github.com/dahai80/musetalk-mlx/releases/download/demo-v1/case_man.mp4" controls preload></video></td>
</tr>
</table>

完整 case 套件（sit、sun2、video1……）及复现方法见
[`musetalk-mlx-cases/README_CN.md`](musetalk-mlx-cases/README_CN.md)。

## 安装

```bash
cd musetalk-mlx
python3.12 -m venv .venv
source .venv/bin/activate
# 先装 fusion-mlx 神经底座（editable，从其检出目录）：
pip install -e /path/to/fusion-mlx
pip install -e ".[dev]"
```

`fusion-mlx[video]>=0.10.2,<0.11` 为版本化依赖（无硬编码本地路径）；先 editable 安装 fusion-mlx 检出目录以满足版本约束。

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

## 环境变量

部署时覆盖（无需重新打包），均为 import 时读取的 `MT_*` 变量。

| 变量 | 默认 | 用途 |
|---|---|---|
| `MT_GRAPH_OPT` | `true` | fusion-mlx #911 Conv+GN+SiLU 图改写 + 联合 mx.compile |
| `MT_SMART_CONV` | `false` | fusion-mlx #919 SmartConv2d（图内测得慢 1.5x，关） |
| `MT_FP16` | `true` | fp16 管线转换（30FPS + <=4GB 预算） |
| `MT_PRECOMPUTE` | `true` | 离线预计算 landmarks+bbox+latent |
| `MT_BATCH` | `2` | 批量 UNet+decode 深度（2 在 RTT 预算内） |
| `MT_DECODE_128` | `false` | 解码前 2x2 均值池化 latent（速度优先） |
| `MT_PASTE_MULTIPROC` | `false` | 子进程做 paste 融合（关——无测得收益，增 IPC） |
| `MT_BG_POOL_MAX_FRAMES` | `240` | 底片帧池上限（8s@30fps，约 304MB）；超长底片按需读取 |

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
- [x] Phase 2（业务层）：重叠音频滑窗（前缀平滑）、生产级融合贴回 + 可插拔 face-parse 掩码、零拷贝帧出站（memoryview 直交 livekit VideoFrame，FR-LK-001）、离线 Demo 打磨。embedding 级前缀缓存待 fusion-mlx #914；face-parse 模型待 #910；native IOSurface→CVPixelBuffer 直编待 #913（livekit Python SDK 当前不直接接受 CVPixelBuffer；桥已接入非 LiveKit sink）。
- [x] Phase 3（业务层）：完整温控降级阶梯（FR-END-003）、ReloadModel 热重载（FR-MLX-006）、LiveKit 实时适配器（FR-LK-001/002，音频继承 PTS，双向音频）、实时运行 CLI。图优化 fusion-mlx #921/#924/#932 已闭（模块级 ResnetBlock2D 熔合 + MSL Conv+GN+SiLU 内核落地，默认开）；ICB（#912）仍为上游 open 项（Python/MLX 路径未消费 — 路线分叉）。
- [x] Phase 4（LCM）：MLX 架构下不适用。fusion-mlx pipe 本身就是 single-step t=0，没有多步 DDIM 可蒸馏成 1-step fast path——LCM 蒸馏在此是同义反复。先前的 `LCMFastSession` 桩 + `MT_LCM_ENABLED` flag 实例化了一个渲染热路径从不调用的死对象，暗示存在不存在的 fast path；两者已删，诚实处理。若 fusion-mlx 未来发布蒸馏权重 API，应重新引入真实 fast path（非桩）。
- [x] 集成测试：神经核心已验证（`tests/integration/` 8 个集成测试通过）；#915 修复已验证（权重严格加载 + mel 打包）；#916/#917 已在 fusion-mlx v0.10.2 修复——真实关键点链路全绿（10/10 检出、bbox 合理、`_FrameSpaceDWPose` workaround 已移除）；离线 E2E 跑通（150 帧视频、真实关键点、无 guard 兜底）；A3 LiveKit 实时已实测（29.82 FPS、RTT 33.3ms、PTS 单调）；A4 2h 压测在跑。剩余门禁：离线 PSNR ≥ 38dB / SSIM ≥ 0.95 对比 MuseTalk torch 参照——`gen_gt.py` 工具存在（subprocess 进 MuseTalk checkout，MPS），尚未跑 torch GT。见 `tests/integration/INTEGRATION_PENDING.md`。

## 性能（M5 Max）

测量前提：fusion-mlx 服务器与其它 GPU 消费者全部停止 —— 后台 GPU 负载会把
数字抬高数倍（详见 `tests/integration/INTEGRATION_PENDING.md`）。

| 优化 | 效果 |
|---|---|
| fp16 转换 + 离线预计算（landmarks/bbox/VAE latent） | DWPose + encode 移出热循环 |
| UNet+VAE-decode 联合 `mx.compile`（单图） | 每批 2 帧的 round 178ms → 82ms（2.2 倍）；分离编译有 compiled-input 边界开销 |
| **MLX 0.32.0 版本锁定** | 0.32.2 的 decode kernel 在联合编译图上有回归：实测 73.6 vs 61.6ms/round。以下数字要求 `mlx==0.32.0` + `mlx-metal==0.32.0` |
| **allocator 上限在预热后设置**（load/precompute 完成后 cache 4GB + budget 8GB） | load 之前设置上限会永久污染 allocator 水位：全程 84–92ms vs 62–67ms/round。cache 扫描 2.5/3/3.5/4/5/6GB → 61.8/58.9/59.1/57.0/56.9/58.4ms |
| 深度 2 渲染前瞻 | 排队两个 lazy round，GPU 在 paste/emit CPU 工作期间保持忙碌（单流；sync 是流级粒度）。注意：MLX lazy eval 当前会串行化 —— 提交的 round 的图只在 sync 时才物化，故深度 2 队列尚未让 GPU 计算与 paste 重叠。重叠需 eager dispatch（fusion-mlx issue 待提） |
| paste 工作线程 + `cv2.blendLinear` | warp/blend/emit（纯 cv2/numpy）移出渲染线程；blend 比numpy fp32 快 7 倍；parse-mask cache miss 延迟到主线程 |
| **子进程 paste worker**（`config.PASTE_MULTIPROC`，默认关） | paste/blend 跑在子进程（独立 GIL），不受 MLX busy-wait sync 在渲染线程上的 GIL 饿死影响。stdin/stdout 上 4 字节长度帧 + pickle 协议；IPC 只传 expand 后的 crop，不传整帧。默认关：热路径已在渲染线程内联 paste（缓存 alpha ~0.2ms，无 IPC）；worker 保留为 drain/IPC 失败回退 |
| **`DECODE_128` 速度开关**（`config.DECODE_128`，默认关） | UNet 输出 latent 2x2 均值池化（32²→16²）后再 VAE decode → 128² face patch，decode FLOPs ~1/4。速度优先、可牺牲质量场景；默认关闭，按 flag 开启 |
| 批 2 热路径（`config.BATCH`） | 每 2 步一次 UNet+decode，RTT 安全（多等一个 33ms 步） |
| **专用渲染线程 + 内联 paste**（A3） | 生产者线程泵 `scheduler.next_step`；`get_output_frame` = 纯消费者（pop out_q，忙等≤2s）。渲染线程内联 paste（缓存 alpha ~0.2ms，无 worker 队列）。修复 2:1 丢帧（`_wait_out_q(pop=False)` 等待 emit 不丢弃） |
| **异步 LiveKit publish 线程 + H264**（A3） | H264 编码（VideoToolbox 硬件）——VP8 软编 1080p 下仅 ~5fps。专用 `_publish_loop` 线程 + 有界 deque（上限 4，丢最旧）解耦 GIL 与渲染线程的 MLX readback（render_eval 140ms→55ms） |
| **零拷贝出站**（FR-LK-001，A3） | BGRA 以 `memoryview`（轮转 numpy 池）交 rtc `VideoFrame`——SDK 把指针传 FFI，无 `tobytes`/`bytearray`（旧路径每帧 3 次拷贝） |

隔离干净 GPU 微基准（fp16, MLX 0.32.0, allocator caps 开, v0.10.4），60-round burst：

| 配置 | GPU/round（b2） | GPU 上限 |
|---|---|---|
| 全 decode（默认） | 61.0ms | 32.8 FPS |
| `DECODE_128` 开 | 34.1ms | 58.7 FPS |

LiveKit E2E（A3，干净 GPU，1080p，`MT_DECODE_128=1`，本地 LiveKit server v1.9.1，`lk_receive.py` 探针）：

| 指标 | 值 |
|---|---|
| 接收端 FPS | **29.82** |
| publish 间隔 p50 | 33.3ms |
| render_eval | 55ms |
| 稳态丢帧 | 0 |
| RTT（渲染管线段） | 33.3ms ≤ 80ms |
| PTS | 严格单调 |

实测到上限的 gap 已由「渲染线程 + 异步 publish + 零拷贝」架构（A3）闭合：旧 ~19ms/帧 gap
是 paste + readback + publish GIL 争用在 pacer 线程串行。VAE decoder 完整质量 gap
（#921/#924/#932 — conv2d fp16 悬崖 + MSL Conv+GN+SiLU 熔合 + 模块级图 patterns）已在
fusion-mlx 闭合；A3 LiveKit E2E 在 `DECODE_128=1` 下实测 29.82 FPS（128²）。完整 256²
质量 30 FPS 需在干净 GPU 上用最新 fusion-mlx 重测熔合 MSL 内核（19.4 FPS full-decode 数字
早于 #921/#924 修复）。

早期「58ms decode / 8 FPS 持续」「24.4 FPS 实测」数据被 linguakids watchdog 自动重启
fusion-mlx 服务器污染，已撤回；上表 A3 LiveKit E2E 为干净 GPU 数字。

## PRD V1.1-RC2 差距补齐（本次发布）

- fusion-mlx v0.10.2 验证（#916/#917）：真实关键点链路全绿 —— 10/10 检出、bbox 合理；消费侧 `_FrameSpaceDWPose` workaround 已移除。
- 图优化 Pass 接入修复：改为编译纯 UNet 前向（`generate_faces` 内含 `mx.eval`，在 `mx.compile` 中非法）；输出与 plain 一致（cosine 1.0）。
- fp16 管线转换（`config.FP16`）+ MuseTalk-realtime 式离线预计算（每底片帧缓存 landmarks/bbox/VAE latent，`config.PRECOMPUTE`），热循环移除 DWPose + encode。

- 对 PyTorch 的分层精度 parity 测试（`tests/parity/`）：Whisper 编码器、VAE encode（cosine ≥ 0.98）、VAE decode（PSNR ≥ 38dB）、8 通道 UNet（cosine ≥ 0.98）。fixture 在 torch 环境一次性生成（`musetalk_mlx/tools/gen_parity_fixtures.py`），测试时无 torch 依赖。
- 单帧关键点丢失走 Kalman 预测（FR-END-006）：位置+速度外推，连续 5 帧丢失才进 idle-blink；含 bbox 合理性 + 离群点剔除守卫。
- 温控降级阶梯全链路接入 session（FR-END-003）：砍步数 → 帧复用(3) → 背景降采样(2) → patch 256→128（绝不直接 256→128）。**注（审计 0921 P1-7）：** serious 档砍步数（`set_ddim_steps` 15→8）在实时路径是 no-op——实时渲染循环固定单步 t=0 以守住 33ms 预算（多步 DDIM 慢 Nx 倍）。serious 档仅在离线多步路径生效。实时降级依赖 critical 档（帧复用 / 背景降采样 / patch-128）。
- 底片帧预加载内存池（FR-END-002），保留流式兜底。
- 通过 fusion-mlx `compile_with_custom_pass`（#911）消费图优化 Pass，探测式优雅降级。
- 零拷贝帧出站（FR-LK-001）：`LiveKitAdapter.publish_frame` 将 BGRA 像素缓冲以 `memoryview`（预分配轮转 numpy 池）交给 livekit rtc `VideoFrame`——SDK 的 `get_address` 把缓冲指针传给 FFI（`ctypes.addressof(c_char.from_buffer)`），无 `tobytes`/`bytearray` 拷贝（旧路径每帧 3 次拷贝）。`capture_frame` 为同步 FFI 请求，native 编码器在调用期间读取指针。`ZeroCopySink` 暴露 fusion-mlx #913 `MetalZeroCopyBridge`（IOSurface 承载 CVPixelBuffer，native 路径；CVPixelBufferCreate+memcpy 兜底）供非 LiveKit sink 消费者（离线、未来 native VideoToolbox 直编）；livekit Python SDK 不直接接受 CVPixelBuffer，故桥的 IOSurface 路径留作未来 native 编码器集成。
- 分阶段 profiler（STFT/Whisper/UNet/VAE/warp/frame-out）、phys-footprint 内存采样、最大连续分配探测（`musetalk_mlx/utils/profiling.py`）。
- 运维工具：`musetalk-mlx-benchmark`（FPS + 分阶段预算，PRD 9.5）、`musetalk-mlx-stress`（长会话泄漏 ≤ 50MB + 碎片探测）。
- 边界输入测试：小于 10ms 音频、全静音、满幅削波。
- Barge-in 打断（审计 0921 P3，K12 核心缺口）：`MuseTalkSession.interrupt()` + `LiveKitAdapter.interrupt()` 清 pending/inflight/out_q 及音频前缀（被截断语音的 embedding 尾会污染下一轮）；`consumed` 保持单调（输出 PTS 不倒退）；`_last_frame` 保留为 standby（不黑屏）；paste 队列轻清（不走 2s join）。仅显式 API——VAD 自动触发为后续项。
- Deadline 驱动 push 模型（审计 A-1）：`realtime_infer.py` pacer 用 `next_deadline += period` + overrun 重同步，替代 `sleep(period*0.5)` 空转；jitter 仪表（publish 间隔 p50/p95/max、overrun、pts_drift）每 10s log + 写 `results/realtime_pacing.json`。PRD 生产态「禁 Python 主循环」过渡达标（最终态 = LiveKit 队列化，Phase 3 后续）。
- Session 拆分（审计 A-1）：从 `session.py` 抽出 `BackgroundStore`（bg 池/缓存/预计算）与 `RenderScheduler`（round 提交/finish/inflight/温控应用）；公共 API 不变，审计注释逐字随迁，profiler stage 字符串不动。

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
| [#918](https://github.com/dahai80/fusion-mlx/issues/918) | `compile_with_custom_pass` 是空壳 — **v0.10.3 已修；musetalk 拓扑匹配 0 处** | 3 |
| [#919](https://github.com/dahai80/fusion-mlx/issues/919) | Metal conv2d fp16 吞吐悬崖 — **v0.10.3 已修；SmartConv2d 数值正确但图内慢 1.5×，默认关** | 3 |
| [#920](https://github.com/dahai80/fusion-mlx/issues/920) | 默认 allocator cache 无界增长 — **v0.10.3 已修** | 3 |
| [#921](https://github.com/dahai80/fusion-mlx/issues/921) | Metal conv2d fp16 kernel 悬崖 — **已闭**：MuseTalk 实时循环 GPU 侧达 30 FPS | 3 |
| [#924](https://github.com/dahai80/fusion-mlx/issues/924) | MSL fused Conv+GN+SiLU kernel — **已闭**：`fusion_mlx/custom_kernels/fused_conv_gn_silu.py` 落地（conv_stats + gn_affine_silu 内核） | 3 |
| [#927](https://github.com/dahai80/fusion-mlx/issues/927) | `set_ddim_steps` 运行时 DDIM 步数 setter — **v0.10.5 已闭**：`pipe.set_ddim_steps(n)` 设 pipe 的 `_ddim_steps` 状态；`_run_unet(steps=None)` 消费它。musetalk-mlx 实时热循环保持单步（steps=1, t=0）以守 33ms 预算；serious 档步削（15→8）在离线多步路径经 `_apply_ddim_steps` 生效 | 3 |
| [#928](https://github.com/dahai80/fusion-mlx/issues/928) | 公共 API 契约 — musetalk-mlx 访问私有属性（`_dtype`/`UNET_TIMESTEP`/`apply_pe`/`unet`）；版本锁是兜底非契约 — **v0.10.5 已闭，迁移到 `pipe.dtype`/`pipe._run_unet`** | 3 |
| [#932](https://github.com/dahai80/fusion-mlx/issues/932) | `apply_patterns` 模块级图熔合 — **已闭**：整块 ResnetBlock2D 熔合（GN→SiLU→Conv 对），默认开（`FUSION_MUSETALK_GRAPH_PATTERNS=1`），在 SmartConv2d 包装前运行 | 3 |

## 运维 runbook

### 单机部署（Apple Silicon，macOS 14+）

1. **前置依赖**：Xcode 18.0 命令行工具、Python 3.11 venv、`brew install ffmpeg`。温控监测需 PyObjC：`pip install pyobjc`。缺失时 session 打 warning，仅跑 normal 档（无温控降级）。
2. **权重**：放在 `weights/`（MuseTalk `unet.pth` ~3.2GB、`whisper-tiny`、`sd-vae-ft-mse`、`weights/eval`）。torch-free 用户须获取预转换 MLX safetensors（见下文权重分发），`musetalk-mlx-convert` 需 torch 环境，torch-free 下无法自建。
3. **fusion-mlx**：从 checkout 可编辑安装（`pip install -e /path/to/fusion-mlx[video]`），锁定 `>=0.10.2,<0.11`。起停服务用 `/path/to/fusion-mlx/start.sh start|stop`。
4. **离线**：`musetalk-mlx-offline --weights weights --audio in.wav --video base.mp4 --out out.mp4`。输出经 ffmpeg mux 源音频（审计 B1）。
5. **实时（LiveKit）**：`musetalk-mlx-realtime --weights weights --video base.mp4 --livekit-url wss://... --token <jwt>`。token 仅供 connect/重连持有，close 时丢弃。
6. **自启（launchd）**：把实时 CLI 包进 `~/Library/LaunchAgents/io.musetalk.mlx.plist`，`KeepAlive=true` 让崩溃后自动拉起。

### 监控 / 告警

- 日志为纯 `logging`（stderr）。生产环境把 stderr 转发到结构化 sink（Loki/Cloudwatch），对 `ERROR`/`WARNING` 速率告警。
- 关键告警信号：`bg pool hit byte budget`、`LiveKit room disconnected`、`paste worker thread exited`、`set_ddim_steps unavailable`、`thermal` 档位转换、`NSProcessInfo unavailable`。
- `musetalk-mlx-stress` 每采样报 RSS + MLX active/cache/peak 内存；把其 JSON 输出接入 2h 泄漏预算告警（阈值 50MB 漂移）。

### 回滚

- fusion-mlx 上界 `<0.11`：0.11 发布须手动验证后才能 bump。回滚用 `pip install 'fusion-mlx[video]<0.11'` 后重启。
- `MuseTalkSession.reload(mlx_dir)` 热替换权重不重启；在窗口边界 drain，swap 中发 standby 底片帧（不黑屏）。不能替代版本回滚，仅用于权重刷新。
- **reload 后掉帧（预期行为，非故障）**：reload 成功后会同步重建 bg latent 缓存（逐底片帧 DWPose + VAE encode，约 150ms/帧）。缓存重建完成前，cache-miss 走实时 encode 兜底路径，帧率约 6 FPS；输出保持 standby 底片帧——不黑屏、不丢音频。掉帧随缓存填充自愈（审计 0921 P1-6）。reload 后数秒内自行恢复的 fps 告警不必告警处理。

### 权重分发（P0 缺口 — 未交付）

`weights/` ~3.2GB，当前手工放置。商用发布需满足以下之一：
- 预转换 MLX safetensors 发布到 HuggingFace（镜像走 https://hf-mirror.com），配 `musetalk-mlx-download` 拉取脚本，或
- 随安装包分发的签名 bundle。

此项为发布阻塞（审计 B5）；`musetalk-mlx-convert` 存在但需 torch 环境，torch-free 终端用户当前无法自建。

## 威胁模型

- **LiveKit token**：短期 JWT，仅在进程内存中供 connect + 重连看门狗持有；绝不入日志，`close()` 时丢弃。每会话轮换；勿在 launchd plist 内嵌长效 token —— 启动时从 Keychain 或 secrets manager 读取。
- **权重**：首次加载信任。`torch.load` 用 `weights_only=True`（`eval/syncnet.py`）；`MT_SYNCNET_UNSAFE_LOAD=1` 仅为开发显式 opt-out。商用 bundle 应在加载前签名校验（尚未实现）。
- **用户输入（音频/视频）**：音频经 `librosa` 解码（无代码执行）；视频经 `imageio`/`cv2`。不可信媒体应在入库前扫描 —— 管线不沙箱化解码。多租户部署应每会话跑在 seatbelt/容器内。
- **IPC**：`paste_proc.py` 用 pickle 经 pipe 传输，但子进程由 `sys.executable -m` spawn（可信父进程）。无外部输入跨越 pickle 边界。
- **subprocess**：所有 shell 调用（`ffmpeg`/`ffprobe`）为 list 形式，无 `shell=True`。

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

## 致谢

本项目是 [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk)（1.5 唇形同步数字人）
面向 Apple Silicon 的 **MLX 移植版**。神经架构（12 通道音频条件 UNet、SD-VAE、
Whisper-tiny、DWPose 关键点、face-parse 贴回）沿用原版 MuseTalk 设计；模型、权重与推理
逻辑在 [MLX](https://github.com/ml-explore/mlx) 上重新实现，运行时无 PyTorch。

- 原版 MuseTalk 仓库：<https://github.com/TMElyralab/MuseTalk>
- 神经底座：[fusion-mlx](https://github.com/dahai80/fusion-mlx)
- 演示素材（底片视频/音频）源自 MuseTalk `data/` 数据集。
