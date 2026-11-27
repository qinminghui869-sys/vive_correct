# Rerun 与深度输入

## 查看已有结果

```bash
python3 -m vive_correct view outputs/cv
# 或指定一份录制
python3 -m vive_correct view /path/to/recording.rrd
```

`view` 只打开已有录制。目录入口读 `rerun/manifest.json`；审核录制在 `repairs/<运行>/rerun/`。左右手以实体名称为准，灰色用于修正前对照。生成步骤见 [导出与查看](../docs/review_rebuild.md#export-and-view)。

<a name="rrd-inputs"></a>

## 导出模式与输入

`raw` 和 `repair` 自动选择下列模式；`--rgb-only` 可在 `raw` 中强制使用 RGB：

| 输入情况 | 显示内容 | 时间线 / JSON mode |
| --- | --- | --- |
| 深度、内参、掩码齐全 | RGB、有效点云、世界坐标手点 | 深度 / `depth_masked` |
| 深度和内参齐全，缺掩码 | RGB、按距离过滤的预览点云、世界坐标手点 | 深度 / `depth_unmasked_preview` |
| 缺深度视频、时间戳或可用参数 | RGB 叠加手关节点与骨架；保留原有 3D 手部视图 | RGB / `rgb_manus_only` 或 `rectified_rgb_corrected_wrists` |

`raw` 使用原始手点；`repair` 使用审核后修正 Parquet，保留修正前后对照，不重复施加矫正。模式不改变源数据或审核状态。支持多 episode、逐条标定和实际分辨率；含中文、空格或 `+` 的路径须加双引号。

点云需要深度视频及时间戳、矫正左目 RGB、内参和深度尺度。当前世界视图还读取头部、腕部、手点以及眼动数据（`eye_gaze`、`eye_rgb` 视频及时间戳）。缺掩码时仅过滤无效值和距离范围，日志及录制注明预览：有损深度中的无效区域不能据此可靠恢复。RRD 的 RGB 回退只决定显示内容，不能使缺少有效深度的原点校准通过；原点输入规则见 [校准模型](../docs/hand_pipeline.md#depth-origin)。

RGB 叠加需要矫正左目内参、头部位姿、相机到头部外参及世界坐标 21 关节。关节点经世界→相机→像素投影，连接可见骨节；相机后方、画面外及缺测点不显示，上一帧叠加会清除。只有局部 MANUS 节点或非矫正 RGB 时，不猜测映射，原因写入 JSON 的 `hand_projection`。外参仍使用指定的安装标定，精度见 [相机参数](../calibrations/README.md)。

录制位置与发布流程见 [导出与查看](../docs/review_rebuild.md#export-and-view)。`render.log` 记录实际模式及时间线；同名 JSON 中的 `mode`、`frame_timeline`、`validity_mask` 或 `hand_projection` 用于排错。

<a name="depth-metadata"></a>

### 深度元数据

目录：`annotation/chunk-000/observation.images.depth_headband_left_rectified/<episode>/`。单条旧数据的 `<episode>` 是 `episode_000000`；多条数据集使用对应编号。以下两种文件名由渲染、原点计算和手动调腕部共用；同时存在时优先第一种：

1. `episode_000000_meta.json`
2. `fast_foundation_stereo_video_meta.json`

独立元数据存在时优先读取，损坏则明确报错。仅当上述文件均不存在，渲染与原点计算从 `meta/info.json` 的深度特征读取毫米单位、无效值和最大距离，从逐 episode（其次根目录）`rectify_params.yaml` 读取内参，并校验分辨率。不推测缺失单位或内参。

旧版元数据读取：

- `intrinsics.left.fx/fy/cx/cy`：焦距和主点，单位像素，须对应实际深度分辨率，用于反投影、投影和相机视锥。
- `max_depth_m`：有效深度上限，单位米。

宽高从视频读取，时间从 Parquet 读取；右目内参、双目基线和深度模型参数不参与渲染。深度按毫米解码为米，反投影公式见 [坐标计算](../docs/hand_pipeline.md#depth-origin)。

排除零深度、声明的无效值及超出上限的深度；有掩码时再排除掩码无效区域。旧版按毫米解码，新格式显式校验毫米单位。

<a name="raw-export"></a>

## 原始 MANUS 查看

无需选帧或人工审核。在仓库根目录执行，替换单条 episode 路径：

```bash
EPISODE="/path/to/dataset/data/chunk-000/episode_000004"
OUT="outputs/episode_000004_raw"
python3 -m vive_correct raw --dataset "$EPISODE" --out "$OUT" --manus --open-rerun
```

单条旧数据的路径规则见 [输入要求](../vive_correct/README.md#inputs)。省略 `--frames` 导出整条；`--open-rerun` 在成功后打开查看器。之后可单独查看：

```bash
python3 -m vive_correct view "$OUT/raw.rrd"
```

| 可选参数 | 用途 |
| --- | --- |
| `--frames 30 31 32` | 只导出指定帧，帧号从 0 开始 |
| `--rgb-only` | 强制 RGB 手点叠加，不生成点云 |
| `--manus` | 在点云模式中额外显示 50 个原始局部节点 |
| `--show-labels` | 显示关节编号 |
| `--repair-hands` | 核查 MANUS 分组并尝试重建 21 关节，另存结果 |
| `--initial-mount FILE` | 指定相机到头部外参 |

默认输出 `raw.rrd` 和同名 JSON；使用 `--repair-hands` 后，根据侧别核查结果输出 `repaired.rrd` 或 `unverified.rrd`。导出日志为 `render.log`，查看器日志为 `viewer.log`。重新导出时可使用新输出目录保留旧录制。

`raw` 不应用审核后的 VIVE 矫正；矫正结果使用 [repair 流程](../docs/review_rebuild.md#export-and-view)。饮料店 `pour_beverage` 格式从 `meta/info.json` 和逐 episode 标定读取参数，缺掩码时生成预览点云。已有 RRD 不随代码更新，需重新导出。

21 关节编号：腕 0，拇指 1–4，食指 5–8，中指 9–12，无名指 13–16，小指 17–20。局部骨架转换与安装假设见 [手部流程](../docs/hand_pipeline.md#manus-reconstruction)。快捷入口 `python3 open_manus.py /path/to/episode` 调用同一原始查看流程。

## 同步与独立工具

跟踪时间戳稳定排序，非有限值丢弃，冲突作为缺测屏障。原始最近邻查看容差为 5 ms；启用跟踪插值时位置线性插值、旋转取最短弧，区间不超过 50 ms，不外推。诊断写入录制 JSON 的 `timestamp_diagnostics`；审核原点深度规则见 [校准模型](../docs/hand_pipeline.md#synchronization)。

`python3 open_file.py /path/to/episode` 打开手动调腕部工具，需要完整深度和系统 Tk。它使用固定世界偏移，区别于浏览器的腕部本地刚性校准。

单独转换已有标签报告时使用：

```bash
python3 ego_data/vive_origin_alignment.py \
  --episode /path/to/episode --report /path/to/observations.json \
  --calibration /path/to/camera_mount.json --output outputs/wrist_alignment.json
```

该命令必须读取同步深度，不再使用报告中的 PnP 位置。浏览器审核生成的有效深度校准可由 RRD 复用。独立手部安装拟合见 [标定说明](../docs/hand_calibration.md)；这些工具均另存结果，不覆盖源数据。
