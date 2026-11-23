# Vive Correct

从矫正左目视频中选取腕部标签参考帧，经人工审核后修正腕部位姿与手部关节点，输出 Parquet 和 Rerun 录制（RRD）。默认使用 OpenCV；可选 AI 检查最终标签框与 XYZ 轴，最终决定始终由人工提交。

**OpenCV 选帧 → 可选 AI 预审 → 人工审核 → 导出／重建**

## 环境准备

需要 Python 3.10+。在仓库根目录执行，后续终端均使用此环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

选帧需要矫正左目视频；审核还需要时间戳、姿态及标定，见 [输入要求](vive_correct/README.md#输入要求)。先核对 [相机与轴定义](calibrations/README.md)；自定义配置须在选帧、审核、导出和重建中使用同一 `--initial-mount FILE`。

## 快速开始

以下命令在仓库根目录运行。将 `/path/to/dataset` 替换为数据集目录，所有阶段使用同一输出目录。

### 1. 选帧

```bash
bash scripts/run_cv.sh /path/to/dataset outputs/review --jobs 4
```

默认处理全部样本，无需模型服务或密钥。启用拼接图 AI 预审见 [AI 配置与运行](docs/ai_review.md)。

### 2. 人工审核

另开终端启动服务，可边选边审：

```bash
python3 -m vive_correct review --out outputs/review --port 18775
```

打开 [审核页面](http://127.0.0.1:18775)，核对物理左右手、四角和 XYZ 轴，填写审核人后提交：**通过**进入后续处理，**不通过**排除整条数据，**暂存**保留草稿。操作细节见 [审核指南](docs/user_guide.md)。

### 3. 导出或重建

仅处理已通过且依据有效的数据。两个操作独立，按需执行：

```bash
# 导出并查看 RRD
python3 -m vive_correct repair --out outputs/review --export-jobs 1
python3 -m vive_correct view outputs/review

# 生成独立修正 Parquet，保留源数据
python3 -m vive_correct rebuild --out outputs/review
```

覆盖源数据须显式使用 `rebuild --in-place`；六文件范围、备份和元数据要求见 [重建说明](docs/review_rebuild.md#重建与备份)。

## 使用边界

- 修正原点是 **50 × 30 mm 标签中心**，标签须与腕部刚性绑定；不等同于已标定的 Vive 内部原点。
- 图像左右不等于物理左右手。AI 建议、PnP 重投影通过均不能替代人工确认或精度验收。
- 默认相机外参是安装近似。源文件只在显式覆盖时修改，`hand_skeleton` 和 `meta/` 不自动更新。

## 文档导航

- 操作：[输入与配置](vive_correct/README.md) · [CV 选帧与调参](docs/cv_detection.md) · [AI 预审](docs/ai_review.md) · [人工审核](docs/user_guide.md) · [导出、重建与排错](docs/review_rebuild.md)
- 技术：[字段与接口](docs/observations_schema.md) · [校准模型](docs/hand_pipeline.md) · [RRD 输入](ego_data/README_rerun.md) · [开发与验证](docs/development.md)
- 历史研究（不作为当前能力承诺）：[安装标定](docs/hand_calibration.md) · [模型实验](docs/experiments.md) · [CV 验证](docs/cv_validation.md)
