# 手部安装标定

位置修正不等于手部拟合。新增 `hand_calibration.py` 支持在 Tracker 局部系中修正旋转和平移，保持骨长；禁止用缩放或镜像伪造贴合。必须使用独立参考关节，不能把最近桌面点当作关节真值。

输入 JSON 包含 `episode`、完整 `camera_calibration`、`wrist_calibration_sha256`、`reference_method`，以及 `hands.left` / `hands.right` 对应点列表。每个对应点包含：

```json
{
  "frame": 100,
  "split": "train",
  "source_tracker_m": [0.01, -0.08, -0.05],
  "target_tracker_m": [0.02, -0.07, -0.04]
}
```

示例坐标只说明格式，不是可用标定值。源点为已核实侧别、尚未加 Vive 修正的手点，转换回原始 Tracker 局部系。独立参考世界点先减去该帧 Vive 世界偏移，再转换回同一 Tracker 局部系。相机及 Tracker 位姿必须使用同一时刻和同一外参。

至少使用 3 个训练帧、2 个不重叠验证帧，训练和验证各至少 6 个对应点；默认验证点最大误差不超过 20 mm。该阈值是可配置验收条件，不是对当前数据精度的声明。

```bash
python3 ego_data/hand_calibration.py \
  --samples /path/to/correspondences.json \
  --output outputs/hand_mount.json --max-error-mm 20
```

随后在 `render_rerun.py` 中同时传入 `--hand-points-dir`、`--wrist-calibration` 和 `--hand-calibration outputs/hand_mount.json`，不要传 `--raw-hands`。标定必须匹配同一 episode、相机配置和腕部修正文件；未通过验证的标定会被拒绝。当前 40 条样本没有独立关节真值，尚不能宣称稳定贴合或 MANO 拟合完成。
