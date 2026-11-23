# 字段与接口

本文只定义坐标、审核记录和预览接口。操作见 [审核指南](user_guide.md)，整段变换见 [校准模型](hand_pipeline.md#标签刚性绑定)。模型交付报告和人工依据包同名 `observations.json`，schema 不同，不能混用。

## 坐标约定

- 图像：矫正左目，左上角为原点，x 向右、y 向下，单位像素；四角按轮廓循环排列，q0 不保证是左上角。
- 相机：OpenCV 右手系，X 右、Y 下、Z 前，位置单位米。
- 标签原点：四角对角线交点对应的标签中心，不默认等于 Vive 内部原点。
- `T_camera_local`：按行存储的 4 × 4 矩阵，将局部齐次列向量变换到相机系；前三列为 XYZ 单位轴，平移列与 `origin_camera_m` 一致。
- `frame_index` 从 0 开始；`timestamp_s` 来自源 Parquet，不以帧号除 FPS 替代。左右手来自不同帧时，不能直接求同时刻相对位姿。

矩阵已包含方向修正，不能重复施加历史旋转。`axes_generated=false` 时 `T_camera_local` 为 `null`，可使用有效原点，但不可用单位旋转补全姿态。

## 人工审核记录：schema 3

`features/.../human_review.json` 保存 `selected.left/right`、审核人、输入绑定及版本信息。每侧记录参考帧、四角、备注、物理手侧、状态，以及：

```json
{
  "axis_angles_deg": [0, 0, 0],
  "axis_convention": "intrinsic_XYZ_degrees"
}
```

角度必须是三个有限数，采用本地内禀 XYZ，`C = Rx(α) Ry(β) Rz(γ)`，右乘源腕部旋转。绕当前轴按钮使用 `C_new = C × R_axis(Δ)` 合成，再转为等效内禀 XYZ 角；手动输入与四角校正使用相同字段；零角度表示恢复源方向。schema 2 加载时保留选择并默认零角度，必须重新审核。

## AI 建议记录

`features/.../ai_review.json` 与人工记录独立；当前 `schema_version=1`、`prompt_version=4`：

- `status`：`running`、`complete` 或 `error`；加载时还可能返回 `stale` 或 `unavailable`。`complete` 仅表示复核结束。
- `original`、`selected`：原选择和建议选择；`adjusted` 表示是否调整，`human_verified` 始终为 `false`。
- `rounds`、`issues`、`error`：各轮判断、待人工处理原因及失败信息；`calls` 不超过 2。
- `input_signature`、`calibration`、`model`、`endpoint`：输入与计算版本绑定、配置路径及模型来源。
- 加载时附加 `valid` 和 `revision`；版本失效时不能继续以该建议提交审核。

人工记录的可选 `ai_reference` 指向采用的建议修订；人工修改帧、四角或方向后清除该关联。模型响应约束与拼接图布局见 [AI 预审](ai_review.md#输入与处理边界)。

## 整条不通过记录

`<out>/rejected_episodes.json` 是整条排除的唯一状态来源，与单侧 `selected.*.status="rejected"` 不同。结构如下，路径仅为示例：

```json
{
  "schema_version": 1,
  "entries": [{
    "episode": "/data/stove/sample",
    "video": "/data/stove/sample/videos/chunk-000/observation.images.headband_left_rgb_rectified/episode_000000.mp4",
    "features_file": "/output/features/sample/features.json",
    "reviewer": "reviewer-id",
    "reason": "标签严重遮挡",
    "rejected_at": "2026-11-19T08:00:00+00:00",
    "review_revision": null
  }]
}
```

`episode` 是去重键；路径由服务端生成。原因可为空，`review_revision` 表示拒绝时已有人工记录的版本。接口返回 `status="excluded"`、`approved=false` 与 `exclusion` 记录；已排除条目不能用普通保存或通过接口恢复。清单原子写入并同步落盘，先记录排除再使旧输出索引失效。

## 依据包

`features/.../approval.json` 指向 `<out>/approved/<版本>/`。使用前须校验审核修订、输入、计算代码、标定快照和文件哈希。

- `human_review.json`：审核快照。
- `observations.json`：参考帧、真实时间戳、四角、PnP 结果及最终姿态；`human_review_sha256`、`review_binding`、`input_signature` 用于追溯，`exclusions_file` 指向当前排除清单。
- `wrist_alignment.json`：每侧本地校准、同步源姿态、深度结果和外参。
- `camera_mount.json`：外参快照；原帧与 `*_quad.png` 保存四角及轴向叠图。
- `bundle.json`：依据索引、审核人、版本、原始六文件及依据文件校验值。
- `corrected_manifest.json`：导出或重建时生成，位于当前依据包目录，记录 `corrected_*/data/` 下的修正文件、哈希、行数和坐标组统计；`corrected_groups` 中的 `*_skipped` 为跳过数量。

### 校准与位姿字段

- `model="wrist_local_rigid"`：`offset_local_m` 为腕部本地平移，`rotation_local` 为右乘本地旋转。`offset_m` 仅是参考时刻世界偏移，不可作为整段固定平移。历史 `model="world"` 走兼容路径。
- `pnp_origin_camera_m`：PnP 原始中心；`origin_camera_m`：沿标签中心射线并经可用深度精修后的中心。
- `rotation_camera_label`：用于中心求解的 PnP 标签平面旋转，不等于设备物理方向。
- `T_camera_local`、`axis_projection`、`axis_length_m`、`camera_K`：最终姿态、投影、绘图轴长（0.04 m）与内参。投影含 `origin` 和按 X/Y/Z 排列的 `endpoints`；不可投影点为 `null`。
- `calibration_sample`：同步源头部与腕部位姿、`T_world_camera`、`rotation_camera_source`、`rotation_camera_corrected`、本地校准和深度结果。
- `local_axes_convention="source_wrist_with_local_calibration"`、`orientation_status="human_axis_review"`：源腕部叠加人工本地旋转。`physical_axes_verified=false` 表示未进行独立物理标定。

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

- `corner_correction`：`axis_angles_deg`、`axis_convention`、`rotation_camera_target`、`geometry_convention`、`pnp_axis_convention`、`baseline_local_rotation`、`plane_ambiguous`、`physical_axes_verified`。
- `pnp_axis_convention`：配置选择的 [PnP 轴定义](../calibrations/README.md#pnp-建议轴)；`baseline_local_rotation` 为标签基准到建议轴的本地右乘矩阵。
- `geometry_convention`：默认 `X_left_Y_up_Z_forward`；批次转换后为 `X_left_Y_negative_label_normal_Z_tangent`。均不表示已识别设备朝向。
- 建议无法生成时 `corner_correction=null`，原因在 `corner_correction_error`；有效的手动轴向预览仍可返回。

四角或同步姿态无效时返回 HTTP 400 与 `error`。接口不写人工记录；仅点击“四角校正”才将建议角度应用到草稿。通过审核时重新计算，不信任客户端投影。

### 列表与任务

- `GET /api/status`：选帧进度 `scan`、列表版本 `list_version`、任务 `jobs`、整体运行标志及发现异常；前端每秒查询。
- `GET /api/episodes`：轻量列表，版本变化时重新获取；列表状态不替代完整依据校验。
- `GET /api/episode?id=…`：加载完整条目并校验依据。
- `POST /api/reject?id=…`：整条不通过；提交 `binding`、`revision`、非空 `reviewer` 和可选 `reason`，成功返回 HTTP 200 及当前审核状态。无需有效图像或几何；版本冲突、空审核人、处理中返回 HTTP 400，缺少会话令牌返回 HTTP 403。
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
