# 修复流程

统一入口为 `python3 -m vive_correct run`，完整命令见[项目首页](../README.md)。旧入口 `python3 vive_correct/run_dataset.py` 仍可使用。

## 输入与配置

每条 episode 需要矫正左目视频及时间戳、`calibrations/rectify_params.yaml`、头部和双腕姿态。骨架显示使用左右手 21 点 parquet；修复图片允许缺失骨架，Rerun 导出还需要双手点、深度视频、有效深度掩码、深度内参、眼动视频与 gaze 数据。

模型配置依次读取命令参数、环境变量 `OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL`、`.env`、JSON。默认 JSON 为 `vive_correct/openai.local.json`，不存在时读取示例配置。密钥只放本地配置或环境变量；复核会向配置服务发送视频帧及局部图。

常用参数：

- `--initial-mount FILE`：头相机外参。默认使用内置 A 版安装近似；新安装应提供自己的外参。
- `--axis-baseline FILE`：按报告哈希和样本绑定的参考观测；保留已有原点与明确方向约束，继续搜索缺失手。
- `--baseline-only`：要求所选样本已有双手基线，跳过模型配置和联网，仍执行扫描、几何诊断及交付。
- `--rerun` / `--open-rerun`：修复后导出 / 导出并打开整段录制。只有双手原点齐全的样本可以导出；失败原因保存在 Rerun 索引及各样本的 `render.log`。
- `--jobs` / `--model-jobs`：CPU / 模型并发，默认 12 / 3。

其余参数见 `python3 -m vive_correct run --help`。几何与时序参数在 `config/estimate.yaml`，扫描和同步参数在 `config/batch.yaml`。

## 阶段与产物

默认 `--stage all` 执行估计、扫描、复核、安装检查和交付。`estimate`、`scan` 用于诊断；`finish` 使用已有人工复核文件，所需参数见帮助。

已有工作结果可以独立重新生成展示或 Rerun，不调用模型：

```bash
python3 -m vive_correct run --out outputs/demo --stage gallery
python3 -m vive_correct run --out outputs/demo --stage rerun --open-rerun
```

重新导出需要完整工作报告与源数据。使用自定义外参时，重新导出也要传入相同的 `--initial-mount`。

```text
<out>/
├── estimates/、search/、qwen_reviews/  # 扫描与复核诊断
├── batch/
│   ├── results/                      # 完整工作报告
│   ├── delivery/index.html           # 图片与精简报告入口
│   └── bilateral_results.zip         # 交付包
└── rerun/
    ├── manifest.json                 # 成功、失败及跳过帧
    └── <编号_样本>/
        ├── wrist_alignment.json      # 原点来源、同步方式、位置偏移
        ├── recording.rrd             # 完整成功后才发布
        ├── recording.json            # 导出统计
        └── render.log
```

接口错误会保留已完成结果，并返回非零退出码。Rerun 导出不会把缺失手补成零偏移，也不覆盖内置手动校准文件。

## 几何与质量边界

相机系为 OpenCV：X 右、Y 下、Z 前，单位米。重投影 RMS 上限 6 px。标签身份、几何质量和安装方向分别判断；模型不生成轴变换，低误差不代表物理标定通过。

只有绑定的设备方向参考才能生成完整轴矩阵；否则 `axes_generated=false`、`T_camera_local=null`。左右手允许来自不同参考帧，不能当成同时刻相对位姿。选中参考帧向前后最多跟踪 15 帧，遇到边界匹配、光流或位姿连续性失败即停止，诊断写入 `device_tracks`，不是完整视频轨迹。

Rerun 使用修复后的标签中心计算每侧固定世界位移，同时平移腕部及该侧骨架。线性位置插值与旋转 SLERP 的相邻样本间隔最多 50 ms，不外推；缺失头部跟踪的帧跳过并记录。青色为左手，粉色为右手，灰色为原始腕部位置。

单条估计仍可用于诊断：

```bash
python3 -m vive_correct estimate --episode /path/to/episode --out outputs/estimate
```

`--frame N` 指定帧；`--detector ai --frame N` 调用模型提供候选角点。自动候选尚未确认物理左右手，不应直接用于 Rerun 修复。

[交付字段](../docs/observations_schema.md)
