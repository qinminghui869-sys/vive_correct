# 相机参数

标签识别使用每条数据的 `calibrations/rectify_params.yaml`，要求矫正左目视频分辨率与内参一致。头部到世界的位姿和相机到头部的外参共同确定相机世界位姿。

## 默认外参

[left_camera_to_head_vive.json](left_camera_to_head_vive.json) 是来自 L3-001 的安装近似，未经独立标定。核心约定：

- `T_head_rectified_camera`：矫正左相机 → 头部 Vive，4 × 4 矩阵，作用于齐次列向量，平移单位米。
- `T_world_camera = T_world_head × T_head_rectified_camera`。
- 源欧拉角按文件中的顺序和单位解码；默认是小写 `xyz` 外旋、角度制，区别于浏览器人工校准的内禀 `XYZ`。

其他设备使用 `--initial-mount FILE`，选帧、审核、导出和重建保持同一份外参。该参数不会切换识别相机。

## PnP 建议轴

挂载配置的 `pnp_axis_convention` 只决定“按四角校正”的建议方向，不改变相机外参、PnP 四角、中心或源姿态：

- `label_z_normal`（省略字段时默认）：从标签平面取最接近图像左／上、Z 远离相机的正交右手系。
- `wrist_y_normal`：在上述基准后右乘 `Rx(-90°)`，即 `X_new=X_old`、`Y_new=-Z_old`、`Z_new=Y_old`。

[new_codex_review_mount.json](../configs/new_codex_review_mount.json) 用于已确认该定义的 `L3_test_data/new` 批次，保留默认相机外参。AI 基准与人工校正共用此转换，不能重复旋转；其他批次需先确认轴定义。`pnp_axis_scope` 仅记录适用数据路径，不自动选择或限制数据集。

这些是几何方向约定，不是设备物理轴的独立标定。预览响应字段见 [轴向预览](../docs/observations_schema.md#只读轴向预览)。

## 其他标定文件

`oak_calib_*.json`、`neon_scene_camera_*.json` 保留设备原始内参供核查。Neon 眼动相机内参不能替代头带矫正左目内参；深度渲染的两种元数据文件名与必需字段见 [RRD 说明](../ego_data/README_rerun.md#深度元数据)。
