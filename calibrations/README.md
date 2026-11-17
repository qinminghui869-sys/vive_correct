# 相机参数

- `left_camera_to_head_vive.json`：默认头相机安装近似（矫正左相机 → 头部 Vive）。米制位置，`xyz` 外旋、角度制。来自 L3-001，未经独立标定；其他设备用 `--initial-mount`。
- `oak_calib_*.json` / `neon_scene_camera_*.json`：设备原始内参，供核查。

标签估计读取每条 episode 的 `calibrations/rectify_params.yaml`，并校验矫正左目分辨率。Neon 内参不用于头带相机。
