# 开发说明

## 模块

- `vive_correct/run_dataset.py`：数据集发现、并行任务、缓存与运行入口。
- `vive_correct/hand_repair.py`：屈指通道验证骨架手侧，校验既有安装假设并生成独立 Parquet。
- `vive_correct/raw_view.py`：原始 Parquet 手点与深度点云查看，不调用模型或修复流程。
- `vive_correct/rerun_export.py`：校验报告、转换原点、调用渲染器并发布完整录制。
- `ego_data/vive_origin_alignment.py`、`tracking_sync.py`：原点到世界偏移、有限间隔时间插值。
- `estimate/`：标签检测、几何求解、稳定性检查和绘图。
- `batch/automatic.py`：模型复核、补扫、观测选择和交付编排。
- `batch/qwen_review.py`、`identity_guard.py`：标签身份与独立设备定位。
- `batch/selection_policy.py`：基线保留、时序诊断与候选排序。
- `batch/pose_alignment.py`、`mount_calibration.py`：时间同步、安装拟合与验证。
- `batch/hand_overlay.py`：21 点骨架读取、同步与投影。
- `batch/bilateral_gallery.py`、`delivery.py`：全局图、网页、精简报告及 ZIP。
- `batch/workflow.py`：基于已有复核文件的批次处理，不调用模型。

## 开发约定

- 算法参数放在 `vive_correct/config/*.yaml`，运行参数通过 CLI 提供。
- 图像左右半区仅用于搜索；物理手侧必须单独确认。模型只判断身份，不生成轴变换。
- 安装参考缺失时保留有效原点并标记无轴，不伪造旋转矩阵；安装标定跳过此类观测。
- 基线绑定源报告哈希、样本和帧，不跨安装传播方向约束。
- `batch/device_axes.py` 绑定逐手参考，禁止统一换轴或按画面重选方向；跟踪使用固定局部角点与相邻姿态。相机系采用 OpenCV，骨架投影使用同步头部位姿与外参。
- 完整诊断留在工作目录，交付字段遵循[接口说明](observations_schema.md)。骨架状态写入总索引的 `hand_mapping`。
- `--stage gallery` 仅更新展示与交付，不修改源位姿或调用模型。
- 计算逻辑改变后使用新输出目录。原始数据不修改，密钥和图像请求载荷不写入日志。

## 测试

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=ego_data python3 -m unittest discover -s ego_data/tests -v
python3 -m vive_correct run --help
```

数值逻辑变更需验证固定样本；展示变更检查图像、报告哈希及 ZIP。

## 代码检查

```bash
python3 -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
```

`.github/ci-template.yml` 提供 GitHub Actions 配置。将其移至 `.github/workflows/tests.yml` 可启用；提交该变更需要 GitHub 凭据具备 `workflow` 权限。CI 不依赖本地数据、密钥和录制结果。单元测试不请求模型服务；完整流程需要自行准备数据和模型配置。

历史实验与本地验证记录保存在忽略的 `outputs/project_archive/`，不作为仓库使用说明。
