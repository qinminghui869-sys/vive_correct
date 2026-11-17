# Vive Correct

从矫正左目视频识别腕部 Vive 标签，得到标签中心在相机系中的位置；经人工审核后，用固定世界平移修正整段腕部轨迹，并可用 Rerun 查看。

## 安装

需要 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

纯 CV 流程不需要模型密钥。默认模型路径使用 `gpt-6-astra`（需已登录的 Codex CLI）。

## 推荐流程（纯 CV）

```bash
# 1. 快速选帧：有界扫描 + 审核材料
bash scripts/run_cv.sh /path/to/dataset outputs/cv --jobs 4

# 2. 浏览器审核（默认 http://127.0.0.1:18763）
python3 -m vive_correct review --out outputs/cv

# 页面「通过审核」保存依据，「导出 RRD 文件」按需生成录制
# 3. 导出完成后查看
python3 -m vive_correct view outputs/cv
```

打开 `outputs/cv/batch/delivery/index.html` 看特征图。默认跳过身份跟踪，每条最多检测 120 帧、扫描软时限 30 秒；双侧严格外观候选齐全则提前结束，缺手仍可人工补标。只读矫正左目，不调用模型。

需要完整补扫时加 `--feature-search full`；需要逐帧身份跟踪时再加 `--stage all`。不同模式使用独立输出目录。

常用选项：`--episodes-file` 限样本、`--dry-run` 只预览计划、`--port` 换审核端口、`--initial-mount` 指定头相机外参。检测参数见 [纯 CV 指南](docs/cv_detection.md)。

需要重建六个源 Parquet 时，先通过审核，再执行 `python3 -m vive_correct rebuild --out outputs/cv --in-place`。覆盖前自动保留初始 `.bak`，支持失败回滚和重复执行；完整说明见 [审核与重建指南](docs/review_rebuild.md)。

## 模型识别（可选）

```bash
python3 -m vive_correct run --dataset /path/to/dataset --out outputs/demo
```

默认 `--profile astra`。`--profile legacy` 走 API；`--open-rerun` / `--rerun` 导出录制。配置与产物见 [运行指南](vive_correct/README.md)。

## 手部原始数据

```bash
python3 open_manus.py /path/to/episode   # MANUS / RGB / 点云预览（不应用标签修正）
python3 open_file.py /path/to/episode    # 手动调腕部
```

详见 [Rerun 说明](ego_data/README_rerun.md)。结果另存，不改源数据。

## 结果含义

- 原点是 **50×30 mm 标签中心**，不是已标定的 Vive 内部原点。
- 无安装方向参考时只输出中心，不生成设备坐标轴。
- 重投影误差、检出率 ≠ 实测位置精度。
- 固定世界平移仅对齐参考时刻，不能保证手腕转动后标签中心仍与修正原点重合。
- 默认头相机外参是安装近似；同类左目设备可用 `--initial-mount` 指定外参。当前不支持切换到眼动相机完成识别与重建。
- 自动结果默认未人工核验；交叉后的物理左右手需审核确认。

## 文档

- [运行指南](vive_correct/README.md)：输入、配置、模型路径、阶段与产物。
- [纯 CV 指南](docs/cv_detection.md)：检测、调参和验证。
- [审核与重建](docs/review_rebuild.md)：日常审核、RRD 导出、备份覆盖和排查。
- [报告接口](docs/observations_schema.md)：坐标、字段与质量状态。
- [手部流程](docs/hand_pipeline.md)：已有手点与 MANUS 重建的区别。
- [开发说明](docs/development.md)：模块职责、数据约束与检查命令。
- [模型对比实验](docs/experiments.md)：固定候选池复核对比。
