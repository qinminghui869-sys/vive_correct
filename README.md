# Vive Correct

基于矫正左目视频识别腕部标签，经人工审核后修正整段腕部位姿和手部关节点，输出 Parquet 与 Rerun 录制（RRD）。默认使用 OpenCV，无需模型服务或密钥。

流程：**选帧 → 人工审核 → 导出／重建**。不通过的数据记录到 `rejected_episodes.json`，跳过后续处理。

## 环境准备

需要 Python 3.10+；输入目录结构见 [输入要求](vive_correct/README.md#输入要求)。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

已有可用环境可跳过安装。使用虚拟环境时，每个终端都需激活。

## 快速开始

### 1. 选帧与审核

终端 A：将 `/path/to/stove` 替换为数据目录，默认处理其下全部样本。

```bash
bash scripts/run_cv.sh /path/to/stove outputs/stove_review --jobs 4
```

终端 B：使用同一输出目录启动审核，可边选边审。

```bash
python3 -m vive_correct review --out outputs/stove_review --port 18775
```

打开 [审核界面](http://127.0.0.1:18775)，检查物理左右手、标签四角和 XYZ 轴，填写审核人：

- **通过审核**：保存重建依据，随后可导出或重建。
- **不通过**：记录极端案例及可选原因，整条数据退出后续流程。
- **仅暂存，不通过**：保存未完成编辑，仍处于待审核状态，不等同于整条“不通过”。

具体操作见 [人工审核指南](docs/user_guide.md)。

### 2. 按需导出与查看

审核通过后，可在页面点击“导出 RRD 文件”，也可批量执行：

```bash
python3 -m vive_correct repair --out outputs/stove_review --export-jobs 1
python3 -m vive_correct view outputs/stove_review
```

### 3. 按需重建六个文件

无需先导出 RRD：

```bash
# 生成独立修正文件，保留源数据
python3 -m vive_correct rebuild --out outputs/stove_review

# 可选：备份后覆盖源数据
python3 -m vive_correct rebuild --out outputs/stove_review --in-place
```

目标为左右腕部位姿、`action`、`observation.state` 和左右手点六个 Parquet；完整范围、备份和元数据要求见 [重建说明](docs/review_rebuild.md#重建与备份)。

### 只处理指定样本

创建非空 JSON 数组，填写相对本次数据目录的样本路径，并在选帧命令后加 `--episodes-file /path/to/episodes.json`。完整搜索用 `--feature-search full`；预览计划用 `--dry-run`。参数细节见 [选帧指南](docs/cv_detection.md)。

## 数据边界

- 修正原点是 **50 × 30 mm 标签中心**；标签须与腕部刚性绑定，不代表已标定的 Vive 内部原点。
- 图像左右不代表物理左右手。自动检测及几何检查不能替代人工确认或精度验收。
- 输入固定为矫正左目；默认外参是安装近似，使用前核对 [相机参数](calibrations/README.md)。
- 仅 `rebuild --in-place` 覆盖源文件；不修改 `hand_skeleton`，不自动更新 `meta/`。

## 文档导航

- 操作：[人工审核](docs/user_guide.md) · [导出、重建与排错](docs/review_rebuild.md) · [CV 选帧](docs/cv_detection.md)
- 参考：[输入与配置](vive_correct/README.md) · [字段与接口](docs/observations_schema.md) · [校准模型](docs/hand_pipeline.md) · [RRD 输入](ego_data/README_rerun.md)
- 开发：[检查、架构与约束](docs/development.md)
- 历史研究（不作为当前能力承诺）：[安装标定](docs/hand_calibration.md) · [模型实验](docs/experiments.md) · [CV 验证](docs/cv_validation.md)
