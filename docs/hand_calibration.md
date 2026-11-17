# 手部安装标定

本文适用于独立的手部安装拟合，不是浏览器标签校准的必经步骤；两条路径见 [手部流程](hand_pipeline.md)。`hand_calibration.py` 在 Tracker 局部系拟合旋转和平移，保持骨长，不使用缩放或镜像。参考关节须独立测量，不能用最近桌面点代替真值。

## 输入与验收

输入 JSON 含 `episode`、完整 `camera_calibration`、`wrist_calibration_sha256`、`reference_method`，以及 `hands.left` / `hands.right` 对应点数组；下例为数组中的一项：

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

## 生成与使用

```bash
python3 ego_data/hand_calibration.py \
  --samples /path/to/correspondences.json \
  --output outputs/hand_mount.json --max-error-mm 20
```

在 `render_rerun.py` 中同时传入 `--hand-points-dir`、`--wrist-calibration` 和 `--hand-calibration`，不要传 `--raw-hands`。须匹配同一 episode、相机配置与腕部修正；未通过验证的标定会被拒绝。
