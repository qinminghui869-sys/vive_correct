# 运行与配置

安装和日常命令见 [项目首页](../README.md)。完整参数用 `python3 -m vive_correct <命令> --help`。

<a id="inputs"></a>

## 输入要求

默认数据集为 `/mnt/nas/synnas/脑电/KS3`，可用 `--dataset` 指定其他目录。输出默认写入 `outputs/ks3`。输入应包含矫正后的头带左目视频；眼部或头部其他相机不能直接替代。

单条旧数据仍按原来的目录读取。一条里只有 `episode_000000` 时，样本根目录就是这条数据：

```text
<episode>/
├── videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4
├── data/chunk-000/episode_000000/observation.images.headband_left_rgb_rectified.parquet
└── calibrations/rectify_params.yaml
```

一个任务里有多条时，每条样本是 `data/chunk-000/episode_NNNNNN/`。视频在任务根目录，文件名与该条编号一致：

```text
<dataset>/
├── videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4
├── videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000001.mp4
├── data/chunk-000/episode_000000/
├── data/chunk-000/episode_000001/
└── calibrations/rectify_params.yaml
```

标定优先读取 `calibrations/chunk-000/episode_NNNNNN/rectify_params.yaml`，不存在时读取任务级 `calibrations/rectify_params.yaml`。各条标定不能互相替代。

纯 CV 选帧只需要矫正左目视频。进入几何审核还需要上面对应的时间戳和标定。识别不回退到原始左目，视频分辨率须与标定一致。后续阶段再补这些输入：

- **轴向预览**：可同步的头部、对应腕部位姿和相机外参。
- **通过审核、导出、重建**：完整的 [六个目标 Parquet](../docs/review_rebuild.md#rebuild-and-backup)。审核提交时即校验结构。
- **完整深度 RRD**：深度、掩码、元数据和眼动输入，见 [RRD 说明](../ego_data/README_rerun.md#rrd-inputs)。

`eye_rgb` 只用于 RRD 眼动展示。`--initial-mount` 指定矫正左相机到头部的外参，不改变识别相机，见 [相机参数](../calibrations/README.md)。

<a id="configuration"></a>

## 配置与优先级

`run --profile` 有三种预设，显式命令行参数覆盖预设：

- `legacy`：默认。`transport=api`，读取 `vive_correct/openai.local.json`，默认 `--stage all`。该文件优先于环境变量和 `.env`；`--model`、`--model-config` 仍优先于该文件。
- `astra`：已登录的 Codex CLI，模型为 `gpt-6-astra`。必须显式 `--profile astra`。
- `cv`：OpenCV 选帧，默认 `--stage features`，模型默认关闭。

配置文件：

- [run.yaml](config/run.yaml)：模型通道、并发和搜索。
- [cv.yaml](config/cv.yaml)：检测、选帧预算和跟踪。
- [estimate.yaml](config/estimate.yaml)：标签尺寸和几何阈值。
- [batch.yaml](config/batch.yaml)：批处理与复核。

`scripts/run_cv.sh` 使用 `--profile cv`，并在用户参数之后关闭模型复核。默认 `--stage features --jobs 4`；写在后面的 `--stage` 或 `--jobs` 会覆盖这两个默认值。直接运行 `run --profile cv` 时，并发取配置预设。审核服务的并发见 [并发与响应](../docs/review_rebuild.md#concurrency)。

`--episodes-file FILE` 限定样本。`--dry-run` 只预览计划。CV 扫描和调参见 [CV 选帧](../docs/cv_detection.md)。

本地模型配置是 `vive_correct/openai.local.json`。没有该文件时，复制 [openai.example.json](openai.example.json) 后填写密钥。该文件已被忽略，不要提交。

<a id="model-pipeline"></a>

## 独立模型候选流程

此流程用模型生成并复核候选，与 CV 选帧后的 [最终结果 AI 预审](../docs/ai_review.md) 是两条路径。日常审核走首页流程。

```bash
python3 -m vive_correct run --dataset /path/to/dataset --out outputs/model
```

顺序是：生成候选、像素四角与 PnP 检查、模型身份和侧别复核、必要时补扫、安装检查、交付。模型框只限定搜索区域。双手可以来自不同帧，不能当作同时刻相对位姿。

默认读取 `openai.local.json`。`--profile astra` 或 `--transport codex` 改走 Codex，请求保存在 `codex_requests/`。`--jobs` 是数据并发，`--model-jobs` 是模型并发。`--fixed-candidates` 关闭补扫和提前停止，用于 [模型对比](../docs/experiments.md)。`--tracker-guided-frames 0` 关闭 Tracker 局部补扫。

## 阶段、续跑与产物

CV 默认只选帧。`--stage all` 才增加整段身份跟踪，仍不自动审核、修正或导出。模型路径默认做到交付；加 `--rerun` 导出，加 `--open-rerun` 导出并打开。

```text
<out>/
├── run_plan.json、feature_plan.json    # 计划与输入绑定
├── features/                           # CV 候选、图片、AI 建议、人工记录
├── rejected_episodes.json              # 整条不通过
├── estimates/                          # 估计与可选跟踪
├── search/、search_refined/             # 模型路径粗扫与补扫
├── qwen_reviews/                       # 模型复核（历史目录名）
├── batch/delivery/index.html           # 静态展示
└── rerun/                              # 模型路径录制
```

相同命令可续跑。输入、检测配置或计算代码变化后换新输出目录。已进入人工审核的特征不会被自动重扫。审核依据包见 [产物与排查](../docs/review_rebuild.md#outputs-and-troubleshooting)。

已有模型结果可单独刷新：

```bash
python3 -m vive_correct run --out outputs/model --stage gallery
python3 -m vive_correct run --out outputs/model --stage rerun --open-rerun
```

任一侧没有有效原点时，不导出模型路径录制。报告字段见 [字段与接口](../docs/observations_schema.md)。
