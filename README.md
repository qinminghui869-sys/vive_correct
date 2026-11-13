# Vive Correct

从头戴相机视频估计腕部标签中心，复核左右手身份，将位置偏移应用到 Vive 轨迹和手部骨架，并使用 Rerun 查看结果。

## 安装

Python 3.10+。在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp vive_correct/openai.example.json vive_correct/openai.local.json
```

在本地配置中填写模型服务的 `api_key`、`base_url` 和 `model`。也可通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 设置。复核过程会向该服务发送选中的图像。

## 运行

```bash
python3 -m vive_correct run \
  --dataset /path/to/episode \
  --out outputs/demo \
  --open-rerun
```

`--dataset` 支持单条 episode 或数据集目录。`--open-rerun` 导出并打开录制；`--rerun` 仅导出。桌面查看器需要图形环境。

已有结果可直接打开或重新导出：

```bash
python3 -m vive_correct view outputs/demo
python3 -m vive_correct run --out outputs/demo --stage rerun --open-rerun
```

修复报告位于 `outputs/demo/batch/delivery/index.html`，录制和失败原因位于 `outputs/demo/rerun/`。重新导出需要源数据和完整工作报告。数据、密钥及生成结果不包含在仓库中。

## 查看原始手点

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/raw_demo --open-rerun
```

直接将 Parquet 中的左右手点叠加到深度点云，不应用原点修正或手点插值。红色为拇指，`--show-labels` 可显示节点编号；同步方式见 [检查说明](ego_data/README_rerun.md#原始手点检查)。

修复已确认的骨架左右分组错误：

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/hands_fixed \
  --repair-hands --open-rerun
```

按屈指通道验证手侧后重新生成手点，源文件不变。适用条件与输出见 [骨架修复说明](ego_data/README_rerun.md#修复骨架左右分组)。

## 精度与限制

- 输出位置是 50×30 mm 标签中心，标签到 Vive 内部原点的偏移尚未标定。
- 默认头相机外参是安装近似。其他安装应通过 `--initial-mount` 提供自己的外参。
- Rerun 对每侧应用固定世界平移，保留源旋转和手指姿态；它不是逐帧视觉校准。
- 标签角点误差、深度歧义和手部安装误差均可能导致骨架不贴合。较小的重投影误差不等于三维位置准确。
- 缺少任意一侧有效原点时不导出录制；缺失头部跟踪的帧会跳过并记录。

RGB 中彩色骨架为平移后的结果，灰色为源骨架。原始数据不修改。

## 文档

- [输入、配置和运行阶段](vive_correct/README.md)
- [Rerun 与手动对齐](ego_data/README_rerun.md)
- [报告字段](docs/observations_schema.md)
- [开发与测试](docs/development.md)
