# 运行与配置

安装与纯 CV 快速开始见 [项目首页](../README.md)。所有命令通过 `python3 -m vive_correct <命令> --help` 查看完整参数。

## 输入要求

纯 CV 选帧只需矫正左目视频；进入几何审核还需要时间戳和标定。基础目录如下：

```text
<episode>/
├── videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4
├── data/chunk-000/episode_000000/observation.images.headband_left_rgb_rectified.parquet
└── calibrations/rectify_params.yaml
```

识别不回退到原始左目视频，视频分辨率须与标定一致。后续输入按阶段准备：

- **轴向预览**：可同步的头部、对应腕部位姿及相机外参。
- **通过审核／导出／重建**：完整的 [六个目标 Parquet](../README.md#3-按需重建六个文件)，审核时即校验其结构。
- **完整深度 RRD**：额外准备深度、掩码、元数据及眼动输入，见 [RRD 说明](../ego_data/README_rerun.md#完整-rrd-的输入)。

`eye_rgb` 仅用于 RRD 眼动展示。`--initial-mount` 指定矫正左相机到头部的外参，不改变识别输入相机，详见 [相机参数](../calibrations/README.md)。

## 配置与优先级

`run --profile` 支持三种预设：

- `cv`：OpenCV 检测，无模型调用，默认 `--stage features`。
- `astra`：默认模型路径，通过已登录的 Codex CLI 调用仓库配置的 `gpt-6-astra`，默认 `--stage all`。
- `legacy`：API 模型路径，需配置 `openai.local.json` 或环境变量。

显式命令行参数覆盖预设。检测与模型默认值见以下配置：

- [run.yaml](config/run.yaml)：模型通道、并发与搜索参数。
- [cv.yaml](config/cv.yaml)：CV 检测、选帧预算与跟踪参数。
- [estimate.yaml](config/estimate.yaml)：标签尺寸与几何阈值。
- [batch.yaml](config/batch.yaml)：批处理与复核参数。

`scripts/run_cv.sh` 固定使用 `cv`，默认数据级并发为 4；直接运行 `run --profile cv` 的并发取配置预设。批量运行建议显式指定 `--jobs`。

审核服务的并发与缓存由独立 CLI 参数控制，见 [并发与响应](../docs/review_rebuild.md#并发与响应)。

常用参数：`--episodes-file FILE` 限定样本；`--dry-run` 预览 `all/features` 计划，不调用模型或生成检测结果；`--initial-mount FILE` 指定外参。CV 扫描与调参见 [纯 CV 指南](../docs/cv_detection.md)。

## 可选模型路径

```bash
python3 -m vive_correct run --dataset /path/to/dataset \
  --out outputs/model --profile astra
```

流程为：生成候选 → 像素四角与 PnP 几何检查 → 模型身份和侧别复核 → 必要时补扫 → 安装检查与交付。模型框仅限定搜索区域，不作为最终角点。双手可以来自不同帧，不能直接计算同时刻相对位姿。

使用 `--transport api` 可切换通道并保留当前 profile 的搜索参数。`--jobs` 控制数据处理并发，`--model-jobs` 控制模型并发；Codex 请求保存在 `codex_requests/`。

`--fixed-candidates` 关闭补扫与提前停止，适合 [模型对比实验](../docs/experiments.md)；`--tracker-guided-frames 0` 关闭 Tracker 局部补扫。模型结果仍须按使用场景人工核查。

## 阶段、续跑与产物

纯 CV 默认仅选帧；显式 `--stage all` 才增加整段身份跟踪。CV 检测不自动审核、修正或导出 RRD。模型路径默认执行估计、扫描、复核、安装检查和交付；可加 `--rerun` 导出，或加 `--open-rerun` 导出并打开。

```text
<out>/
├── run_plan.json、feature_plan.json   # 运行计划与输入绑定，按路径生成
├── features/                         # CV 候选、图片和人工记录
├── estimates/                        # 估计与可选 CV 跟踪
├── search/、search_refined/           # 模型路径粗扫与补扫
├── qwen_reviews/                     # 模型复核（沿用历史目录名）
├── batch/delivery/index.html         # 静态展示入口
└── rerun/                            # 录制索引与日志
```

相同命令可续跑；输入、检测配置或计算代码变化后使用新输出目录。已进入人工审核的特征结果不会被自动重扫覆盖。审核依据包及修正文件目录见 [产物与排查](../docs/review_rebuild.md#产物与排查)。

已有模型结果可单独更新展示或录制：

```bash
python3 -m vive_correct run --out outputs/model --stage gallery
python3 -m vive_correct run --out outputs/model --stage rerun --open-rerun
```

缺少任一侧有效原点时不导出模型路径录制。报告字段见 [字段与接口](../docs/observations_schema.md)。
