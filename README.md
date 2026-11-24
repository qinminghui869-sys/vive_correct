# Vive Correct

从矫正左目视频选取腕部标签参考帧，人工审核后修正腕部位姿与手部关节点，输出 Parquet 和 Rerun 录制（RRD）。默认用 OpenCV 选帧。可选 AI 预审最终标签框和 XYZ 轴。通过或排除始终由人工提交。

## 环境

需要 Python 3.10+。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

选帧需要矫正左目视频；审核与重建还需要时间戳、姿态及标定，见 [输入要求](vive_correct/README.md#inputs)。自定义外参时，各阶段传入同一 `--initial-mount FILE`，约定见 [相机与轴定义](calibrations/README.md)。

<a id="quick-start"></a>

## 快速开始

以下命令均在仓库根目录执行。`DATASET` 换成实际数据路径，各阶段使用同一个 `OUT`。不写 `--out` 时，过程文件默认进入 `outputs/ks3`。

```bash
DATASET="/path/to/dataset"
OUT="outputs/ks3"
bash scripts/run_cv.sh "$DATASET" "$OUT"
```

上面的命令只做 OpenCV 选帧。如需 AI 预审，首次运行时改用：

```bash
bash scripts/run_cv.sh "$DATASET" "$OUT" --ai-final-review
```

首次使用 AI 时，按 [配置说明](vive_correct/README.md#configuration) 创建 `vive_correct/openai.local.json`。

另开终端启动审核，可与选帧同时进行：

```bash
python3 -m vive_correct review --out "$OUT" --port 18775
```

打开 [审核页面](http://127.0.0.1:18775)。核对物理左右手、四角和红 X / 绿 Y / 蓝 Z，填写审核人后提交。**通过**进入后续处理，**不通过**排除整条，**暂存**保留草稿。页面操作见 [审核指南](docs/user_guide.md)。

只处理已通过且依据仍有效的数据：

```bash
python3 -m vive_correct repair --out "$OUT" --export-jobs 1
python3 -m vive_correct view "$OUT"
python3 -m vive_correct rebuild --out "$OUT"
```

`repair` 导出 RRD，`rebuild` 另存修正 Parquet，两者都不改源文件。覆盖源数据必须显式使用 `rebuild --in-place`，范围见 [原地重建](docs/review_rebuild.md#rebuild-and-backup)。

## 使用边界

- 修正原点是 **50 × 30 mm 标签中心**，且标签与腕部刚性绑定。它不是已标定的 Vive 内部原点。
- 画面左右不是物理左右手。AI 建议和 PnP 重投影通过都不能代替人工确认。
- 默认相机外参是安装近似。只有 `rebuild --in-place` 修改源文件，范围见 [原地重建](docs/review_rebuild.md#rebuild-and-backup)。

## 文档

- 操作：[输入与配置](vive_correct/README.md) · [CV 选帧](docs/cv_detection.md) · [AI 预审](docs/ai_review.md) · [人工审核](docs/user_guide.md) · [导出与重建](docs/review_rebuild.md)
- 技术：[字段与接口](docs/observations_schema.md) · [校准模型](docs/hand_pipeline.md) · [RRD 输入](ego_data/README_rerun.md) · [开发与验证](docs/development.md)
- 专项工具：[手部安装标定](docs/hand_calibration.md) · [模型对比实验](docs/experiments.md)
- 验证记录：[Smoke 测试](docs/smoke_test.md) · [CV 历史验证](docs/cv_validation.md)（不代表当前精度）

## 验证

安装开发依赖后运行 `bash scripts/check.sh`，包含文档链接检查、静态检查和单元测试。完整步骤见 [开发与验证](docs/development.md#checks)。
