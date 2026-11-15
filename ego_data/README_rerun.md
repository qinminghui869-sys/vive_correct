# Rerun 查看与对齐

## 打开修复结果

```bash
python3 -m vive_correct view /path/to/refined.rrd
```

此命令读取已有录制，不重新识别。自动流程的录制保存在 `<out>/rerun/`。青色为左手，粉色为右手，灰色为位置修正前的骨架；已重建侧别时，灰色也来自重建后的手点。时间轴用于切换帧，实体树可隐藏 `source` 对照。

## 检查 MANUS 与 RGB

```bash
python3 open_manus.py /path/to/episode
```

显示原始 50 节点的两组局部骨架、RGB，以及可用的世界坐标骨架和深度点云。等价入口为 `open_file.py /path/to/episode --tool manus`。该入口核查侧别，但不应用标签位置修正；查看最终修复结果应打开对应 `.rrd`。

需要限定帧或输出目录时：

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/manus_preview \
  --manus --repair-hands --show-labels --frames 90 300 900 --open-rerun
```

- 省略 `--repair-hands`：直接读取原始 21 关节文件。
- `--rgb-only`：只查看 RGB 与原始 MANUS 局部节点。
- 缺深度资源时自动回退；主 RGB 缺失时选择可用相机，来源写入录制说明。
- 侧别不确定时保留原始预览并标记 `unverified_source_preview`，不猜测左右手。

输出为 `raw.rrd`、`repaired.rrd` 或 `unverified.rrd`，同名 JSON 记录同步与显示状态。`hands/repair.json` 保存侧别证据和源文件哈希。回退视图标记 `rgb_manus_only`，没有点云配准或 Vive 修正。

原始 MANUS 的 0–24、25–49 是数据分组，不预设物理侧别。世界骨架的 21 点编号为：腕部 0，拇指 1–4，食指 5–8，中指 9–12，无名指 13–16，小指 17–20。当前没有 MANO 网格。

## 同步与数据兼容

跟踪时间戳稳定排序；非有限时间戳丢弃，相同重复记录合并，冲突记录作为缺测屏障。空流、单行流可读取。视频保持原帧索引，不按时间重排。

原始查看按最近记录匹配手点，容差 5 ms；只对头部做有限间隔插值。修复导出将手点和位姿同步到深度时间，位置线性插值、旋转走最短弧，区间不超过 50 ms，不外推、不跨缺测屏障。清理统计见录制 JSON 的 `timestamp_diagnostics`。

## 手动调整腕部

```bash
python3 open_file.py /path/to/episode
```

选择手侧，输入世界 XYZ 偏移，或点击 RGB 目标并确认深度，再保存并生成录制。世界轴为 X 左、Y 上、Z 前；固定世界偏移应用到整段腕部和同侧骨架。该工具需要完整深度资源和系统 Tk。缺深度时使用上面的 MANUS 查看入口。

## 单独转换标签位置

```bash
python3 ego_data/vive_origin_alignment.py \
  --episode /path/to/episode --report /path/to/observations.json \
  --calibration /path/to/camera_mount.json --output outputs/wrist_alignment.json
```

该独立命令转换双手 PnP 位置；自动 Rerun 导出还会执行深度平面校验。渲染时通过 `--wrist-calibration` 传入结果，并保持相同样本及相机外参。

数据映射及新流程见 [DAG 与坐标说明](../docs/hand_pipeline.md)，可选的旋转和平移标定见 [手部安装标定](../docs/hand_calibration.md)。源数据不修改，默认安装外参的适用范围见 [相机参数](../calibrations/README.md)。
