# 输入与运行配置

安装及日常命令见 [项目首页](../README.md)。本页说明两种模式的输入、配置和续跑规则；完整 CLI 参数使用 `python3 -m vive_correct <命令> --help` 查看。

## 输入要求

### 公共数据

每个样本以 `<episode>/` 为根。审核、导出和重建需要左右 Vive 位姿、左右 21 关节 point，以及 `action/state`，即 [六个目标 Parquet](../docs/review_rebuild.md#重建-parquet)。视频与位姿使用真实时间戳，分辨率必须与相机标定一致。

### 矫正左目 `headband`

```text
<episode>/
├── videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4
├── data/chunk-000/episode_000000/observation.images.headband_left_rgb_rectified.parquet
└── calibrations/rectify_params.yaml
```

选帧只需矫正左目视频，不回退到原始左目。几何审核还需要标定、可同步的头部和腕部位姿；相机到头部外参用 `--initial-mount FILE` 指定，默认采用仓库安装近似。完整深度录制的额外输入见 [Rerun 说明](../ego_data/README_rerun.md#完整-rrd-的输入)。

### 眼动固定参考系 `eye-reference`

```text
<episode>/
├── videos/chunk-000/observation.images.eye_rgb/episode_000000.mp4
└── data/chunk-000/episode_000000/observation.images.eye_rgb.parquet
```

使用原始眼动视频，运行时统一去畸变；不读取 `eye_rgb_rectified`。选帧前检查公共数据及 `observation.hand_skeleton.parquet`，缺少完整手点或骨架的样本整条跳过。无需头部位姿、左目外参和深度输入。

默认眼动内参为 `calibrations/neon_scene_camera_917093.json`，在选帧时用 `--eye-calibration FILE` 覆盖。两种模式的参数格式和坐标约定见 [相机标定](../calibrations/README.md)。

## 模式与阶段

`--coordinate-mode headband` 为默认，`--coordinate-mode eye-reference` 显式启用眼动固定参考系。模式和相机上下文随特征报告保存，`review`、`repair`、`rebuild` 自动继承；`--initial-mount` 仅用于左目模式。

推荐 `--profile cv`：OpenCV 检测，无模型调用，默认阶段为 `features`。两种模式均支持选帧、审核、导出和重建；眼动模式的 `run --stage` 限定为 `features/gallery/rerun`。左目模式可额外使用 `--stage all` 进行图像身份跟踪；眼动模式整段位姿由参考锚点与源 Vive 流传播，无需图像身份跟踪。

`scripts/run_cv.sh` 固定使用 CV，支持以上两种坐标模式，默认 `--jobs 4`。直接运行 `run --profile cv` 的并发取配置预设，批量任务建议显式指定。

## 常用配置

显式 CLI 参数优先于配置文件：

- `--episodes-file FILE`：限定样本，JSON 内容为相对数据集根目录的路径列表，例如 `["scene/episode_a", "scene/episode_b"]`。
- `--feature-search fast|full`：选帧覆盖与预算，详见 [CV 选帧](../docs/cv_detection.md#选择样本与扫描模式)。
- `--dry-run`：仅输出 `features/all` 阶段计划，不写检测结果、不调用模型。
- `--cv-config FILE`：覆盖部分检测配置，默认值见 [cv.yaml](config/cv.yaml)。
- 标签尺寸和几何阈值见 [estimate.yaml](config/estimate.yaml)；运行预设见 [run.yaml](config/run.yaml)。

审核服务有独立的 [并发与缓存参数](../docs/review_rebuild.md#并发与响应)。

## 续跑与产物

同一配置使用相同命令续跑；不同模式使用不同输出目录。输入、标定、检测配置或计算代码变化后，应使用新目录重新选帧审核。已进入人工审核的结果不被自动重扫覆盖。

`feature_plan.json` 保存 CV 选帧计划和输入绑定，`feature_progress.json` 保存进度，`features/` 保存候选与人工记录，`batch/delivery/index.html` 提供静态预览。审核依据包、修正文件及录制目录见 [产物与排查](../docs/review_rebuild.md#产物与排查)。

## 可选模型路径

模型路径仅适用于左目模式，不属于推荐的纯 CV 流程：

```bash
python3 -m vive_correct run --profile astra   --dataset /path/to/dataset --out outputs/model
```

`astra` 通过已登录的 Codex CLI 调用仓库配置的模型；`legacy` 使用 API 配置。模型路径默认执行 `all`，生成候选后进行几何检查、身份与侧别复核、必要补扫和展示。模型框不作为最终角点，模型复核不代替人工审核。

`--transport api` 切换调用通道，`--model-jobs` 控制模型并发，`--fixed-candidates` 固定搜索预算。模型产物包括 `run_plan.json`、`estimates/`、`search/` 和 `qwen_reviews/`；参数及历史比较见 [模型实验](../docs/experiments.md)。
