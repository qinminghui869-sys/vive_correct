# Rerun 查看与对齐

## 打开修复结果

```bash
python3 -m vive_correct view /path/to/refined.rrd
# 或：python3 -m vive_correct view outputs/cv
```

读取已有录制，不重新识别。录制路径由 `<out>/rerun/manifest.json` 索引；人工审核导出文件位于 `<out>/repairs/<运行>/rerun/`。青色左手、粉色右手；灰色为修正前骨架。

## 检查 MANUS 与 RGB

```bash
python3 open_manus.py /path/to/episode
```

显示原始局部骨架、RGB，以及可用的世界骨架与点云。核查侧别，**不**应用标签位置修正。

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/manus_preview \
  --manus --repair-hands --show-labels --frames 90 300 900 --open-rerun
```

- 省略 `--repair-hands`：直接读原始 21 关节
- `--rgb-only`：只看 RGB + 原始 MANUS
- 缺深度自动回退；侧别不确定时标记 `unverified_source_preview`，不猜左右

输出为 `raw.rrd` / `repaired.rrd` / `unverified.rrd` 及同名 JSON。世界骨架编号：腕 0，拇 1–4，食 5–8，中 9–12，无名 13–16，小 17–20。

## 同步

跟踪时间戳稳定排序；非有限值丢弃，冲突作缺测屏障。原始查看容差 5 ms；修复导出同步到深度时间，位置线性插值、旋转最短弧，区间 ≤50 ms，不外推。统计见录制 JSON 的 `timestamp_diagnostics`。

## 手动调腕部

```bash
python3 open_file.py /path/to/episode
```

需完整深度与系统 Tk。世界轴：X 左、Y 上、Z 前；固定世界偏移作用于整段腕部与同侧骨架。缺深度时用上面的 MANUS 入口。

## 单独转换标签位置

```bash
python3 ego_data/vive_origin_alignment.py \
  --episode /path/to/episode --report /path/to/observations.json \
  --calibration /path/to/camera_mount.json --output outputs/wrist_alignment.json
```

审核流程保存深度校验后的偏移，RRD 复用该结果；旧导出路径在转换时做校验。数据流见 [手部流程](../docs/hand_pipeline.md)，标定见 [手部安装标定](../docs/hand_calibration.md)，默认外参见 [相机参数](../calibrations/README.md)。源数据不修改。
