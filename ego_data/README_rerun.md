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

## 原始手点检查

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/raw_demo --open-rerun
```

直接读取 `observation.hand_left_points.parquet` 和 `observation.hand_right_points.parquet` 的世界坐标，不应用 Vive 原点修正，不插值手点。以深度时间戳选最近记录，超过 5 ms 则隐藏；`raw.json` 记录各帧的时间差及有效性，Rerun 的 `sync` 实体也可查看。

深度点云通过头部姿态与相机外参转换到世界系。仅头部姿态使用不超过 50 ms 间隔的插值；原始双腕姿态也按最近记录显示。默认相机外参仍是安装近似，可用 `--initial-mount` 替换。

青色为左手，粉色为右手，红色突出拇指。`--show-labels` 显示节点编号（密集视图可能重叠）。节点 0 为手腕，1–4 为拇指，5–8 为食指，9–12 为中指，13–16 为无名指，17–20 为小指。先检查 Vive 与实物位置，再检查腕点及手指方向。`--frames 90 300 900` 可只导出指定帧；`python3 -m vive_correct view outputs/raw_demo` 可重新打开完整录制。

## 修复骨架左右分组

```bash
python3 -m vive_correct raw \
  --dataset /path/to/episode --out outputs/hands_fixed \
  --repair-hands --open-rerun
```

程序将原始 50 节点的中指、无名指弯曲角与 `hand_pose` 的左右屈指通道比较，确认分组后重新生成 21 点，绑定同侧 Vive。动作不足或左右证据不明确时停止，不猜测手侧。已检查的四条 desk 样本均为 `node00–24` 对应右手、`node25–49` 对应左手；原手点文件使用了相反分组。

仅支持当前 L3 安装假设：`Tracker(x,y,z) = MANUS(y,z,x) + [0,-0.085,-0.058] m`，Vive 姿态按 `xyz` 外旋、角度制解析。程序先验证该假设能重建源手点，其他安装参数不匹配时停止。此步骤修正骨架来源，不重新标定安装变换、相机外参或 Vive 原点。

修复文件另存于 `<out>/hands/`，`repair.json` 记录分组证据、源文件哈希和有效行数；不修改原数据及其 meta。`repaired.rrd` 为修复后的点云预览，`repaired.json` 记录显示同步情况。生成手点时插值同侧 Vive 姿态；播放时仍直接选取最近手点记录，不再插值手点。不能通过交换两个世界坐标文件名完成这项修复。
