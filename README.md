# Vive Correct

从矫正左目视频选取腕部标签参考帧，经人工审核后修正腕部位姿与手部关节点，导出 Parquet 或 Rerun 录制（RRD）。默认使用 OpenCV 选帧，AI 预审可选。

修正原点为与腕部刚性绑定的 **50 × 30 mm 标签中心**；默认相机外参是安装近似。它们不代表已独立标定的 VIVE 内部原点或物理轴。

## 安装

需要 Python 3.10+，在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

输入格式见 [输入与配置](vive_correct/README.md#inputs)。使用自定义外参时，各阶段传入同一 `--initial-mount FILE`，见 [相机参数](calibrations/README.md)。

<a id="quick-start"></a>

## 快速开始

只查看原始数据、点云或 RGB 手点叠加，无需审核，直接使用 [原始 RRD 导出](ego_data/README_rerun.md#raw-export)。以下流程用于人工审核后的矫正。

以下命令均在仓库根目录执行。替换数据路径，各阶段使用同一输出目录：

```bash
DATASET="/path/to/dataset"
OUT="outputs/ks3"
bash scripts/run_cv.sh "$DATASET" "$OUT"
python3 -m vive_correct review --out "$OUT" --port 18775
```

打开 [审核页面](http://127.0.0.1:18775)，核对物理左右手、标签四角及 XYZ，填写审核人后提交。操作见 [人工审核](docs/user_guide.md)；启用可选模型检查见 [AI 预审](docs/ai_review.md)。

审核通过后按需导出：

```bash
python3 -m vive_correct repair --out "$OUT" --export-jobs 1
python3 -m vive_correct view "$OUT"
python3 -m vive_correct rebuild --out "$OUT"
```

## 输出说明

| 命令 | 产物 | 是否需要审核 |
| --- | --- | --- |
| `raw` | 原始数据 RRD 与同名统计 JSON | 否 |
| `repair` | 矫正前后对照 RRD，索引 `repair_manifest.json` | 是 |
| `rebuild` | 修正 Parquet，索引 `rebuild_manifest.json` | 是 |
| `view` | 打开已有 RRD | 不生成数据 |

RRD 根据输入显示点云或 RGB 手点叠加，规则见 [RRD 输入](ego_data/README_rerun.md#rrd-inputs)。默认只写输出目录；覆盖源数据须显式执行 `rebuild --in-place`，范围、备份和恢复约定见 [导出与重建](docs/review_rebuild.md)。

## 文档导航

- 使用：[CV 选帧](docs/cv_detection.md) · [AI 预审](docs/ai_review.md) · [人工审核](docs/user_guide.md) · [导出、重建与排错](docs/review_rebuild.md)
- 配置：[输入与运行配置](vive_correct/README.md) · [相机参数](calibrations/README.md) · [RRD 输入与回退](ego_data/README_rerun.md)
- 技术：[字段与接口](docs/observations_schema.md) · [校准模型](docs/hand_pipeline.md) · [手部安装标定](docs/hand_calibration.md)
- 开发：[开发与检查](docs/development.md#checks) · [Smoke 测试](docs/smoke_test.md) · [模型对比](docs/experiments.md) · [CV 历史验证](docs/cv_validation.md)
