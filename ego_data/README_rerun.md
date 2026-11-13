# Rerun 与腕部对齐

自动修复到 Rerun 的统一命令见[项目首页](../README.md)，输入、产物和质量限制见[修复流程](../vive_correct/README.md)。

## 查看录制

```bash
python3 -m vive_correct view /path/to/recording.rrd
```

Rerun 左侧显示点云、腕部及骨架，右侧显示 RGB、眼动和标定说明。青色为左手、粉色为右手；自动流程还显示灰色原始腕部位置和修正连线。时间轴可播放或定位帧。

`view` 也可接收输出目录或 `review_index.html`，直接读取完整录制，不调用模型。自动导出在 RGB 中叠加 21 点和彩色骨架，并用灰色显示源骨架作对照；可以在实体树中隐藏各侧的 `source`。统计 JSON 的 `source_hand_mount` 记录源腕点相对 Vive 的局部偏移及一致性，仅用于诊断，不是实测标定。

## 手动对齐

在项目根目录打开任意 episode：

```bash
python3 open_file.py /path/to/episode
```

选择左腕或右腕，输入世界 XYZ 偏移，或点击 RGB 目标并确认深度；预览后点击“保存并生成 Rerun”。世界坐标为 X 左、Y 上、Z 前。固定世界位移应用到整段腕部和该侧手点。

内置样本保持 `ego_data/outputs/wrist_alignment_A.json` 和 `l3_A_wrists_aligned_all.rrd`。其他样本使用 `场景_episode名` 独立命名，预览选择距第 300 帧最近且可同步的帧。显示参数保存在 `l3_render_settings.json`。

## 单独转换已有原点

需要接入自己的脚本时：

```bash
python3 ego_data/vive_origin_alignment.py \
  --episode /path/to/episode --report /path/to/observations.json \
  --calibration /path/to/camera_mount.json --output outputs/wrist_alignment.json
```

报告必须属于同一个样本，且包含双手 `origin_camera_m`。转换结果交给 `render_rerun.py --wrist-calibration`，同时传入同一 `--dataset`、`--calibration` 和帧范围。自动流程已完成这些步骤，一般无需分别运行。

默认相机安装近似见 [calibrations](../calibrations/README.md)。手动工具需要系统安装 Tk（Ubuntu：`sudo apt install python3-tk`）。
