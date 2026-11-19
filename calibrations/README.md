# 相机标定

标定与坐标模式配套使用，不能只替换内参切换相机。两种模式均要求视频分辨率与标定一致。

## 矫正左目 `headband`

内参来自每条数据的 `calibrations/rectify_params.yaml`。头部位姿和相机到头部外参共同确定相机世界位姿。

[left_camera_to_head_vive.json](left_camera_to_head_vive.json) 是来自 L3-001 的安装近似，未经独立标定。核心约定：

- `T_head_rectified_camera`：矫正左相机 → 头部 Vive，4 × 4 矩阵，作用于齐次列向量，平移单位米。
- `T_world_camera = T_world_head × T_head_rectified_camera`。
- 源欧拉角按文件中的顺序和单位解码；默认是小写 `xyz` 外旋、角度制，区别于浏览器人工校准的内禀 `XYZ`。

其他设备使用 `--initial-mount FILE`，选帧、审核、导出和重建保持同一份外参。该参数不会切换识别相机。

## 眼动固定参考系 `eye-reference`

显式启用 `--coordinate-mode eye-reference` 后，默认读取 [Neon 917093 内参](neon_scene_camera_917093.json)，并对原始 `observation.images.eye_rgb` 统一去畸变。去畸变使用原 K、原分辨率；选帧、审核与 PnP 使用相同图像坐标。该模式不使用 `T_head_rectified_camera` 或头部位姿。

使用其他眼动设备时，在选帧命令中传 `--eye-calibration FILE`。JSON 须包含 3 × 3 的 `camera_matrix`、OpenCV `dist_coefs` 数组和 `[宽, 高]` 的 `resolution`，实际视频分辨率必须一致。审核、重建与导出继承保存的标定路径、哈希和参数；修改标定后使用新输出目录重新选帧审核。

相机内参本身不能确定深度：本模式结合已知 50 × 30 mm 标签求解 PnP。参考帧眼动相机定义新原点，输出保持原 Standing 世界轴向（RUB），原 Vive 旋转字段不变，标签中心近似 Vive 原点；设备轴向和 `[0,−0.085,−0.058] m` 安装偏移仍需独立物理标定才能验收精度。几何和安装关系见 [校准模型](../docs/hand_pipeline.md#眼动固定参考系)。

## 其他标定文件

`oak_calib_*.json` 及其他 `neon_scene_camera_*.json` 保留设备原始内参供核查。Neon 眼动相机内参不能替代头带矫正左目内参；切换识别相机必须显式选择相应模式。默认左目深度渲染的元数据文件名与必需字段见 [RRD 说明](../ego_data/README_rerun.md#深度元数据)。眼动模式不使用这些深度输入。
