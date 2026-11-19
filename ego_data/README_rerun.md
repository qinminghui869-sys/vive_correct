# Rerun 与深度输入

## 查看已有结果

```bash
python3 -m vive_correct view outputs/cv
# 或指定一份录制
python3 -m vive_correct view /path/to/recording.rrd
```

此命令只打开已有录制，不重新识别或导出。目录入口读取 `rerun/manifest.json`；审核录制实际位于 `repairs/<运行>/rerun/`。青色表示左手，粉色表示右手，灰色为修正前骨架。录制生成步骤见 [审核指南](../docs/review_rebuild.md#导出与查看)。

## 完整 RRD 的输入

已有深度视频后无需重新计算双目深度，但仅有深度视频不足以完整渲染，还需要：

- 深度时间戳 Parquet、`valid_mask_ffv1.mkv` 有效掩码及深度元数据。
- 矫正左目 RGB 视频，用于着色和图像显示。
- 头部、左右腕部位姿及手点；审核导出使用同一份修正 Parquet。
- `eye_gaze`、`eye_rgb` 视频及其时间戳，供当前渲染器的眼动视图读取。

世界对齐还依赖 [相机到头部外参](../calibrations/README.md) 和头部欧拉角约定。缺外参的独立查看模式不能视为点云与跟踪数据已对齐。审核导出在深度输入不足时回退到 RGB；选中的深度元数据损坏时不静默切换文件。

### 深度元数据

目录：`annotation/chunk-000/observation.images.depth_headband_left_rectified/episode_000000/`。以下两种文件名由渲染、深度精修和手动调腕部共用；同时存在时优先第一种：

1. `episode_000000_meta.json`
2. `fast_foundation_stereo_video_meta.json`

渲染读取五个数值：

- `intrinsics.left.fx/fy/cx/cy`：焦距和主点，单位像素，须对应实际深度分辨率，用于反投影、投影和相机视锥。
- `max_depth_m`：有效深度上限，单位米。

宽高从视频读取，时间从 Parquet 读取；右目内参、双目基线和深度模型参数不参与渲染。当前固定按毫米编码解码：

```text
Z = 深度像素值 / 1000
X = (u − cx) Z / fx
Y = (v − cy) Z / fy
```

排除零深度和掩码无效区域。`depth_unit`、`depth_encoding`、`invalid_value`、`scale` 不会自动改变解码尺度。

## 原始 MANUS 查看

```bash
python3 open_manus.py /path/to/episode
```

显示原始局部骨架、RGB 和可用的世界骨架／点云，不应用标签校准。需要重建手点或指定帧时：

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/manus_preview \
  --manus --repair-hands --show-labels --frames 90 300 900 --open-rerun
```

省略 `--repair-hands` 直接读取已有 21 关节；`--rgb-only` 仅查看 RGB 与原始 MANUS。缺深度自动回退，侧别不确定时标记 `unverified_source_preview`。

输出为 `raw.rrd`、`repaired.rrd` 或 `unverified.rrd` 及同名 JSON。21 关节编号：腕 0，拇指 1–4，食指 5–8，中指 9–12，无名指 13–16，小指 17–20。局部骨架转换与安装假设见 [手部流程](../docs/hand_pipeline.md#独立-manus-重建)。

## 同步与独立工具

跟踪时间戳稳定排序，非有限值丢弃，冲突作为缺测屏障。原始最近邻查看容差为 5 ms；启用跟踪插值时位置线性插值、旋转取最短弧，区间不超过 50 ms，不外推。诊断写入录制 JSON 的 `timestamp_diagnostics`；审核深度精修规则见 [校准模型](../docs/hand_pipeline.md#同步与字段保真)。

`python3 open_file.py /path/to/episode` 打开手动调腕部工具，需要完整深度和系统 Tk。它使用固定世界偏移，区别于浏览器的腕部本地刚性校准。

单独转换已有标签报告时使用：

```bash
python3 ego_data/vive_origin_alignment.py \
  --episode /path/to/episode --report /path/to/observations.json \
  --calibration /path/to/camera_mount.json --output outputs/wrist_alignment.json
```

浏览器审核已保存本地校准，RRD 直接复用。独立手部安装拟合见 [标定说明](../docs/hand_calibration.md)；这些工具均另存结果，不覆盖源数据。
