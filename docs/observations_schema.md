# 字段与接口

本文定义坐标、人工审核记录和预览接口。操作步骤见 [审核指南](review_rebuild.md)，整段变换公式见 [校准模型](hand_pipeline.md)。模型交付报告与人工审核依据包是不同 schema，不能仅凭同名 `observations.json` 混用。

## 坐标约定

- 图像：左目模式为矫正左目，眼动模式为使用原 K 统一去畸变后的 `eye_rgb`；左上角为原点，x 向右、y 向下，单位像素。四角按轮廓循环排列，q0 不保证是左上角。
- 相机：OpenCV 右手系，X 右、Y 下、Z 前，位置单位米。
- 标签原点：四角对角线交点对应的标签中心，不默认等于 Vive 内部原点。
- `T_camera_local`：按行存储的 4 × 4 矩阵，将局部齐次列向量变换到相机系；前三列为 XYZ 单位轴，平移列与 `origin_camera_m` 一致。
- `frame_index` 从 0 开始；`timestamp_s` 来自源 Parquet，不以帧号除 FPS 替代。左目模式允许左右手不同帧；眼动模式通过审核时必须同帧，整段输出使用该参考帧的固定相机坐标系。

矩阵已包含方向修正，不能重复施加历史旋转。`axes_generated=false` 时 `T_camera_local` 为 `null`，可使用有效原点，但不可用单位旋转补全姿态。

## 人工审核记录：schema 3

`features/.../human_review.json` 保存 `selected.left/right`、审核人、输入绑定及版本信息。每侧记录参考帧、四角、备注、物理手侧、状态，以及：

```json
{
  "axis_angles_deg": [0, 0, 0],
  "axis_convention": "intrinsic_XYZ_degrees"
}
```

角度必须是三个有限数，采用本地内禀 XYZ，`C = Rx(α) Ry(β) Rz(γ)`。左目模式右乘源腕部旋转，零角度恢复源方向；眼动模式右乘标签几何基底，表示标签到设备的安装方向约束，再联合拟合共同外参；零角度是零安装修正。手动输入与四角校正使用相同字段。schema 2 加载时保留选择并默认零角度，必须重新审核。

眼动模式仍使用 `selected.left/right` 分别存储两侧角点、角度和状态，不因界面共用画面而合并两侧数据。模式由 `features.json.camera_input` 及输入绑定确定；`GET /api/episode` 返回 `coordinate_mode` 供界面选择布局。旧记录缺少眼动上下文时按原左目流程读取。

### 依据包

`features/.../approval.json` 指向 `<out>/approved/<版本>/`。使用前须校验审核修订、输入、计算代码、标定快照和文件哈希。

- `human_review.json`：审核快照。
- `observations.json`：参考帧、真实时间戳、四角、PnP 结果及最终姿态；`human_review_sha256`、`review_binding`、`input_signature` 用于追溯。
- `wrist_alignment.json`：每侧本地校准、同步源姿态、深度结果和外参。
- `camera_mount.json`：左目模式为外参快照；眼动模式沿用文件名，内容为眼动 `camera_input` 标定上下文。帧图与 `*_quad.png` 保存审核图像、四角及轴向叠图；眼动审核图像已经统一去畸变，叠图还包含可用手点。
- `bundle.json`：依据索引、审核人、版本、原始六文件及依据文件校验值。
- `corrected_manifest.json`：导出或重建时生成，位于当前依据包目录，记录 `corrected_*/data/` 下的修正文件、哈希、行数和坐标组统计；`corrected_groups` 中的 `*_skipped` 为跳过数量。

### 校准与位姿字段

- `model="wrist_local_rigid"`：`offset_local_m` 为腕部本地平移，`rotation_local` 为右乘本地旋转。`offset_m` 仅是参考时刻世界偏移，不可作为整段固定平移。历史 `model="world"` 走兼容路径。
- `pnp_origin_camera_m`：PnP 原始中心；`origin_camera_m`：沿标签中心射线并经可用深度精修后的中心。
- `rotation_camera_label`：用于中心求解的 PnP 标签平面旋转，不等于设备物理方向。
- `T_camera_local`、`axis_projection`、`axis_length_m`、`camera_K`：最终姿态、投影、绘图轴长（0.04 m）与内参。投影含 `origin` 和按 X/Y/Z 排列的 `endpoints`；不可投影点为 `null`。
- `calibration_sample`：同步源头部与腕部位姿、`T_world_camera`、`rotation_camera_source`、`rotation_camera_corrected`、本地校准和深度结果。
- `local_axes_convention="source_wrist_with_local_calibration"`、`orientation_status="human_axis_review"`：源腕部叠加人工本地旋转。`physical_axes_verified=false` 表示未进行独立物理标定。

### 眼动固定参考系字段

- 选帧报告 `features.json` 使用 `selection_policy="joint_frame_only"`；`reference_candidates` 每项含 `frame_index`、`hands`、`label_count`、`score`、`evidence` 和 `other_hand_visibility_verified`。`hands` 仅保存实际检出的标签，缺失侧不填造角点；`dual_label`／`single_label_pending_other_hand` 是标签候选线索，不表示物理侧别或精度已确认。`reference_frame`、`reference_frame_image` 指向首选共同帧及其去畸变图。
- `GET /api/episode` 增加帧级 `reference_candidates`，候选 ID 为 `frame:<帧号>`；原 `candidates` 保留兼容。旧眼动报告仅将左右同帧的已有特征作为共同候选，不拼接单侧候选池。

- `observations.json` 增加 `coordinate_mode="eye-reference"` 和 `camera_input`；审核后的 `observations` 只有一个共同帧，`stickers` 同时包含 `left/right`。
- `camera_input` 保存 `video_key`、`calibration_path`、`calibration_sha256`、`K`、`distortion`、`resolution`、`rectification` 和 `offset_local_m`。其中 `distortion` 是原始镜头参数，参与图像去畸变；PnP 使用去畸变后的图像和 K，不重复施加原畸变。
- `wrist_alignment.json` 使用 `version=5`、`model="eye_reference"`、`extrinsic_model="eye_origin_standing_axes"`、`world_frame="eye_origin_standing_axes"`、`world_axes="RUB"`、`orientation_policy="preserve_source"`。`reference_frame`、`reference_timestamp_s` 是共同参考时刻；顶层 `T_reference_source` 仅平移原点，旋转块为单位矩阵；`camera_origin_source_m` 保存参考相机在源世界的位置，`T_camera_source` 保存拟合的相机外参，`T_camera_reference` 仅用于从输出世界投影到参考相机；`wrists.left/right` 保留同值以兼容旧接口。`shared_fit` 记录方法、权重、两侧残差、阈值、安装近似和源欧拉角解析状态。
- `samples` 保存输出世界中的 `reference_pose`、相机中的 `camera_pose`、拟合前相机中的 `reviewed_reference_pose`、源腕部位姿及安装方向修正；`state_fields` 保存各 `action/state` 文件的左右字段前缀映射；`hand_validation` 保存骨架分组一致性结果，不代表物理侧别已独立标定。
- 预览使用 `local_axes_convention="source_vive_camera_projection"`；`T_world_local` 保存方向未变的输出位姿，`T_camera_local` 仅供相机绘图；`calibration_sample` 不含头部世界外参，包含 `T_reference_source`，深度状态为 `pnp_only`。`hand_projection` 为可用的 21 关节像素投影，`wrist_camera_m` 为腕节点参考相机坐标；无法生成手点预览时提供 `hand_preview_error`。
- 修正腕部及 point 文件写入 `coordinate_frame`、`unit`；混合坐标的 `action/state` 仅写 `wrist_coordinate_frame`、`wrist_field_mapping`、`wrist_position_unit`。另记录 `reference_frame`、`invalid_value` 和 `extrinsic_status`。无效目标位置为 NaN，源旋转列逐值保留，原 null 掩码保留，不能按左目模式保留源世界坐标。

相机模式、标定和安装上下文参与依据绑定与缓存身份。跨模式或修改标定后不得复用旧审核依据。

## 浏览器接口

接口仅服务本机会话。POST 请求使用 `Content-Type: application/json` 和页面下发的 `X-Review-Token`。数据 `id` 由持久化目录分配，只追加，不随新结果或服务重启重排。

### 只读轴向预览

`POST /api/axis-preview?id=0`：

```json
{
  "side": "left",
  "frame_index": 100,
  "quad": [[120, 262], [180, 262], [180, 298], [120, 298]],
  "axis_angles_deg": [0, 0, 0],
  "axis_convention": "intrinsic_XYZ_degrees"
}
```

数值仅示意格式。成功返回单侧位姿、投影与 `calibration_sample`，并提供四角校正建议：

- `corner_correction`：`axis_angles_deg`、`axis_convention`、`rotation_camera_target`、`geometry_convention`、`plane_ambiguous`、`physical_axes_verified`。
- `geometry_convention="X_left_Y_up_Z_forward"`：在同一个 PnP 平面上选择最接近图像左／上及远离相机方向的正交右手系，不是设备朝向识别。
- 建议无法生成时 `corner_correction=null`，原因在 `corner_correction_error`；有效的手动轴向预览仍可返回。

眼动预览额外提交 `selected: {left: {...}, right: {...}}`，每侧包含帧号、四角和安装角度；响应的 `shared_fit.provisional` 表示另一侧尚未标注，`consistent` 表示残差是否在阈值内。缓存键包含两侧所有观测和角度；任一侧修改会使两侧预览失效。旧单侧请求仍提供临时预览，不能作为双侧审核依据。

四角或同步姿态无效时返回 HTTP 400 与 `error`。接口不写人工记录；仅点击“四角校正”才将建议角度应用到草稿。通过审核时重新计算，不信任客户端投影。

### 列表与任务

- `GET /api/status`：选帧进度 `scan`、列表版本 `list_version`、任务 `jobs`、整体运行标志及发现异常；前端每秒查询。
- `GET /api/episodes`：轻量列表，版本变化时重新获取；列表状态不替代完整依据校验。
- `GET /api/episode?id=…`：加载完整条目并校验依据。
- `POST /api/review?id=…`：保存人工记录，校验 `binding` 与 `revision`。
- `POST /api/approve?id=…`、`POST /api/export?id=…`：分别提交审核和导出，HTTP 202 仅表示任务已接受；完成或失败由 `jobs` 查询。

新结果不覆盖草稿；旧请求响应必须按数据、手侧、帧和编辑版本隔离。同条排队或处理中的数据不能重复提交或修改。

## 模型交付报告

文件位于 `delivery/results/<样本>/observations.json`，描述选中帧上标签相对矫正左目相机的位姿，不是完整轨迹。

### 顶层与每帧

- `schema_version`、`episode`、`video_key`：结构版本和源信息；源路径换机器后不保证可访问。
- `origin`、`camera_coordinates`、`pose_transform`：原点、坐标和变换说明。
- `camera`：3 × 3 的 `K`、像素分辨率 `[宽, 高]` 和 `distortion`；`axis_length_m` 是绘图轴长，不是旋转矩阵列长。
- `mode`：`both_hands_same_frame`、`separate_frames`、`single_hand` 或 `no_usable_label`。
- `mount_calibration`：安装标定的 `accepted` 与 `reason`。
- `identity`、`reviewer_type`：身份来源；`cv_image_half_unverified` / `cv_only` 表示图像半区假设，未经物理侧别确认。
- `observations[]`：`frame_index`、`timestamp_s` 和 `stickers.left/right`；缺失手侧不出现，无有效观测时列表为空。

### 每只手与质量状态

位置和矩阵遵循上面的坐标约定。设备固定轴用 `local_axes_convention="device_fixed_anchor"`；缺少安装方向时为 `unresolved`，仍可保留有效中心。其他轴向约定不能自动当作设备参考。

- `identity_source`：身份来源；历史缺失值为 `unknown`。
- `axes_generated`、`axes_failure_reason`：是否生成完整轴及失败原因；生成不等于物理验证。
- `quality_status`：`verified`、`conditional_motion`、`low_resolution_preview` 或 `unknown`；`verified` 仅表示观测检查通过。
- `orientation_status`：方向审核状态；`axes_not_generated` 表示只有原点，其他状态保留来源含义，未知值不可当作通过。
- `rms_px`：角点重投影 RMS，单位像素，不是三维位置误差。
- `stability_passed`、`temporal_consistency_passed`：稳定性与本次时序检查。
- `plane_ambiguous`、`orientation_180_unresolved`：平面多解和 180° 方向歧义。
- `physical_axes_verified`、`true_vive_origin_calibrated`：独立物理轴向与 Vive 内部原点标定状态。
- `baseline_retained`：沿用历史观测，不表示本次重新确认。

验证或歧义字段的 `null` 表示未知，不是通过；缺失误差也为 `null`。历史质量状态与本次时序结论可能不同。只含原点的结果不得用于完整姿态检查、双手相对变换或安装标定。内部 `device_tracks` 保存固定轴跟踪与停止原因。
