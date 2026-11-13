# 相机参数

`left_camera_to_head_vive.json` 是默认头相机安装近似，定义从矫正左相机到头部 Vive 的变换。位置单位为米，姿态按 `xyz` 外旋、角度制解析。该文件来自 L3-001 的安装配置，未经过独立标定；其他设备应通过 `--initial-mount` 指定自己的外参。

`oak_calib_19443010C100CC4800.json` 和 `neon_scene_camera_917093.json` 是设备原始内参，供核查使用。

标签估计读取每条 episode 的 `calibrations/rectify_params.yaml`，并校验矫正左目视频的分辨率。Neon 内参不用于头带相机。
