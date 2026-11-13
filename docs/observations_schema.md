# 位姿交付接口

文件：`delivery/results/<样本>/observations.json`。用于读取指定视频帧中，腕部标签参考坐标系相对矫正左目相机的位姿。不是完整轨迹，也不是已标定的 Vive 世界系位姿。

## 坐标契约

- 相机系：OpenCV，X 向右、Y 向下、Z 向前，长度单位米。
- 左右手各自使用设备固定参考系，方向由对应安装参考帧确定；随手腕运动旋转，不按画面左、上或相机前向重新定义。
- 原点：标签中心；Vive 内部原点是否标定以状态字段为准。
- `T_camera_local`：按行保存的 4×4 矩阵，将局部坐标变换到相机坐标。前三列分别为局部 XYZ 轴在相机系中的单位向量，第四列前三项为原点坐标。
- 矩阵已包含方向修正，不要再次应用历史旋转。不同帧的左右手结果不能作为同时刻相对位姿。

## 顶层字段

- `schema_version`：数据结构标识，供客户端兼容性检查。
- `episode`、`video_key`：源样本路径与视频字段。源路径是追溯信息，换机器后不保证可访问。
- `origin`：原点定义。
- `camera_coordinates`、`pose_transform`：坐标与变换说明。
- `camera`：`K` 为 3×3 内参；`resolution` 为 `[宽, 高]` 像素；`distortion` 说明图像已矫正。
- `axis_length_m`：绘图轴长，当前 0.04 米；不是单位轴向量的长度。
- `mode`：`both_hands_same_frame`、`separate_frames`、`single_hand` 或 `no_usable_label`。
- `mount_calibration`：`accepted` 与 `reason`，表示安装标定结论。
- `observations`：选中帧列表；无有效观测时为空。

## 每帧字段

- `frame_index`：从 0 开始的帧索引。
- `timestamp_s`：源视频 parquet 对应的时间戳，单位秒；用于同步，不应直接用帧号除以 FPS 替代。
- `stickers`：按物理手侧 `left`、`right` 保存；没有观测的手侧不出现。

## 每只手字段

- `local_axes_convention`：`device_fixed_anchor` 表示设备固定参考轴，`unresolved` 表示没有可用安装方向；其他来源约定不能自动当作设备参考。
- `origin_camera_m`：标签中心在相机系中的 `[x,y,z]`，单位米。安装方向未确认时仍保留通过几何拟合的原点。
- `axes_generated`：是否生成完整参考坐标轴；不等于物理轴向已经验证。
- `axes_failure_reason`：未生成原因，例如 `installation_reference_missing`（缺少安装参考）、`use_original_device_reference`（需提供统一换轴前的参考）；成功时为 `null`。
- `T_camera_local`：最终位姿矩阵；`axes_generated=false` 时为 `null`，不得用单位旋转替代。
- `quality_status`：`verified`、`conditional_motion`、`low_resolution_preview` 或 `unknown`；`verified` 仅表示观测检查通过，不代表物理标定通过。
- `orientation_status`：沿用原方向状态，例如 `not_direction_reviewed`、`user_constraint_applied`、`reconstructed_pending_direction_review`、`model_uncertain`；`axes_not_generated` 表示仅有原点；未知状态不能当作已验证。
- `rms_px`：角点重投影均方根误差，单位像素；不等于位置误差或物理轴向准确度。
- `stability_passed`：稳定性检查结论。
- `temporal_consistency_passed`：本次短时跟踪与位姿连续性检查结论。
- `baseline_retained`：是否沿用历史观测；`true` 不表示本次重新确认。
- `plane_ambiguous`：是否存在平面多解歧义。
- `orientation_180_unresolved`：是否仍存在 180° 方向歧义。
- `physical_axes_verified`：物理轴向是否已验证。
- `true_vive_origin_calibrated`：Vive 内部原点是否已标定。

验证与歧义字段为 `true` / `false` / `null`；`null` 表示缺失或未知，不得当作通过。`axes_generated`、`baseline_retained` 为布尔值。缺失的误差为 `null`。历史观测保留原质量状态，所以 `quality_status` 与本次时序结论可能不同。

## 读取示例

```python
import json
import numpy as np

with open(report_path) as f:
    report = json.load(f)

for observation in report['observations']:
    for side, pose in observation['stickers'].items():
        origin_m = np.asarray(pose['origin_camera_m'], dtype=float)
        if not pose['axes_generated']:
            continue  # 可以使用原点；没有可用的 XYZ 轴或完整姿态。
        T = np.asarray(pose['T_camera_local'], dtype=float)
        x_axis, y_axis, z_axis = T[:3, 0], T[:3, 1], T[:3, 2]
        # 使用 observation['timestamp_s'] 同步源数据。
        # 依用途检查质量、方向和标定状态，不凭单一 verified 字段放行。
```

## 使用限制

几何拟合通过但安装方向未确认时保留原点；标签身份仍需正常复核。仅原点结果不参与完整位姿时序检查、双手相对变换或安装标定。

完整位姿中的 `origin_camera_m` 与矩阵平移列一致。读取 `T_camera_local` 前先检查 `axes_generated`，不得用单位旋转补全缺失姿态。详细角点、复核记录和诊断保存在内部工作报告中。

固定轴跟踪结果及停止原因保存在内部报告 `device_tracks` 中。参考方向沿用用户约束，不代表 Vive 内部原点或安装关系已完成实测标定。
