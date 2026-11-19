# Vive Correct

使用 OpenCV 识别腕部标签，经人工审核后校准整段 Vive 位姿和已有 21 关节手点，输出 Parquet 与 Rerun 录制（RRD）。纯 CV 流程无需模型密钥。

**选择模式 → 选帧 → 人工审核 → 按需导出或重建**。两种模式共用审核、版本化依据包和文件重建流程。

## 选择模式

- **矫正左目 `headband`（默认）**：从矫正左目视频定位标签，结合头部位姿和相机外参，在源跟踪世界中校准腕部与手点。左右手可分别选择参考帧，审核界面分别展示左右图。
- **眼动固定参考系 `eye-reference`**：从原始眼动视频统一去畸变后定位标签，以双手共同参考帧的眼动相机为原点，保持原 Standing 世界的轴方向；只平移 Vive 和手点，原旋转字段不变。审核界面只显示一张图，在同帧中依次调整左手、右手；无需头部位姿和左目外参。

按目标坐标系选择模式；每种模式使用独立输出目录。模式在选帧时确定，后续审核、导出和重建自动继承。输入文件与标定要求见 [运行配置](vive_correct/README.md#输入要求)。

## 环境准备

需要 Python 3.10+。以下命令在项目根目录执行；建议使用 16 个逻辑 CPU、32 GiB 内存，资源较少时降低并发。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

眼动模式当前处于几何流程验证阶段，真实手点贴合尚未验收。仅平移原点不会修复原手点的安装方向误差；已知限制见 [验证记录](docs/eye_reference.md#当前限制)。

## 快速开始

### 1. 选帧

选择一条命令，将 `/path/to/dataset` 替换为数据目录：

```bash
# 矫正左目
bash scripts/run_cv.sh /path/to/dataset outputs/headband   --coordinate-mode headband --jobs 4

# 眼动固定参考系
bash scripts/run_cv.sh /path/to/dataset outputs/eye_reference   --coordinate-mode eye-reference --jobs 4
```

左目默认快速选帧，可加 `--feature-search full` 扩大搜索；眼动模式固定从 10 秒开始、每隔 5 帧、最多 100 帧。加 `--dry-run` 可仅查看计划。候选仍需人工确认，选帧不会自动审核或导出。

### 2. 审核

在另一终端启动审核。以下以 `outputs/headband` 为例；选择眼动模式时，将后续命令中的输出目录统一替换为 `outputs/eye_reference`。

```bash
python3 -m vive_correct review --out outputs/headband
```

打开 <http://127.0.0.1:18763>，检查物理手侧、标签四角和设备轴向，填写审核人后点击 **通过审核**。未完成时点 **仅暂存，不通过**。可边选帧边审核，新结果不会覆盖当前编辑。

界面操作见 [人工审核指南](docs/user_guide.md)。

### 3. 导出或重建

审核后，点击页面的 **导出 RRD 文件**，完成后查看：

```bash
python3 -m vive_correct view outputs/headband
```

需要修正 Parquet 时执行，无需先导出 RRD：

```bash
python3 -m vive_correct rebuild --out outputs/headband
```

默认另存六个修正文件，源数据不变。仅在明确需要覆盖源文件时增加 `--in-place`；写入范围、备份和恢复见 [重建说明](docs/review_rebuild.md#重建-parquet)。

## 使用约定

- 标签物理尺寸为 **50 × 30 mm**；标签中心近似 Vive 原点，设备轴向和安装关系仍需人工核查。
- 图像左右不代表物理手侧；自动候选、重投影误差或功能测试通过不代表实测精度达标。
- 相同配置可续跑；更换模式、输入、标定、检测配置或计算代码时使用新输出目录，避免复用失效依据。
- 眼动模式只支持纯 CV 选帧和审核后的位姿传播，不估计相机平移，不生成未知编码深度点云。

## 文档

- **操作**：[人工审核](docs/user_guide.md) · [导出、重建与排错](docs/review_rebuild.md)
- **配置**：[输入与运行参数](vive_correct/README.md) · [相机标定](calibrations/README.md) · [CV 选帧与调参](docs/cv_detection.md)
- **技术**：[校准模型](docs/hand_pipeline.md) · [字段与接口](docs/observations_schema.md) · [Rerun 与深度输入](ego_data/README_rerun.md) · [开发说明](docs/development.md)
- **验证**：[眼动模式验证记录](docs/eye_reference.md) · [CV 历史验证](docs/cv_validation.md)

研究工具与历史实验：[手部安装标定](docs/hand_calibration.md) · [模型对比](docs/experiments.md)。
