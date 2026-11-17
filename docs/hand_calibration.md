# 手部安装标定

位置修正 ≠ 手部拟合。`hand_calibration.py` 在 Tracker 局部系修正旋转和平移，保持骨长；禁止用缩放或镜像伪造贴合。必须使用独立参考关节，不能把最近桌面点当关节真值。

输入 JSON 含 `episode`、完整 `camera_calibration`、`wrist_calibration_sha256`、`reference_method`，以及 `hands.left` / `hands.right` 对应点：

```json
{
  "frame": 100,
  "split": "train",
  "source_tracker_m": [0.01, -0.08, -0.05],
  "target_tracker_m": [0.02, -0.07, -0.04]
}
```

示例仅说明格式。源点为已核实侧别、尚未加 Vive 修正的手点（Tracker 局部系）；独立参考世界点先减该帧 Vive 偏移，再转回同一局部系。相机与 Tracker 须同一时刻、同一外参。

至少 3 个训练帧、2 个不重叠验证帧，训练与验证各 ≥6 个对应点；默认验证最大误差 ≤20 mm（可配置，不是当前数据精度声明）。

```bash
python3 ego_data/hand_calibration.py \
  --samples /path/to/correspondences.json \
  --output outputs/hand_mount.json --max-error-mm 20
```

在 `render_rerun.py` 中同时传入 `--hand-points-dir`、`--wrist-calibration` 和 `--hand-calibration`，不要传 `--raw-hands`。须匹配同一 episode、相机配置与腕部修正；未通过验证的标定会被拒绝。
