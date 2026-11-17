# 相机参数

标签识别使用每条数据的 `calibrations/rectify_params.yaml`，要求矫正左目视频分辨率与内参一致。头部到世界的位姿和相机到头部的外参共同确定相机世界位姿。

## 默认外参

[left_camera_to_head_vive.json](left_camera_to_head_vive.json) 是来自 L3-001 的安装近似，未经独立标定。核心约定：

- `T_head_rectified_camera`：矫正左相机 → 头部 Vive，4 × 4 矩阵，作用于齐次列向量，平移单位米。
- `T_world_camera = T_world_head × T_head_rectified_camera`。
- 源欧拉角按文件中的顺序和单位解码；默认是小写 `xyz` 外旋、角度制，区别于浏览器人工校准的内禀 `XYZ`。

其他设备使用 `--initial-mount FILE`，选帧、审核、导出和重建保持同一份外参。该参数不会切换识别相机。

## 其他标定文件

`oak_calib_*.json`、`neon_scene_camera_*.json` 保留设备原始内参供核查。Neon 眼动相机内参不能替代头带矫正左目内参；深度渲染的两种元数据文件名与必需字段见 [RRD 说明](../ego_data/README_rerun.md#深度元数据)。
