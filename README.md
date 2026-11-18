# Vive Correct

从矫正左目视频识别腕部标签，经浏览器人工审核后，校准整段腕部位姿和已有手点，并导出 Rerun 录制（RRD）。默认推荐纯 CV 流程，无需模型密钥。

流程：**选帧 → 人工审核 → 按需导出 RRD／重建 Parquet**。选帧与审核可并行，导出由人工触发。

## 环境准备

需要 Python 3.10+。以下命令均在项目根目录执行；数据应满足 [输入要求](vive_correct/README.md#输入要求)。

建议 **16 个逻辑 CPU、32 GiB 内存**，适合同时选帧、审核和单任务导出。**8 个逻辑 CPU、16 GiB 内存**也可运行，但多任务同时计算可能需要等待；满负载内存峰值尚未测量。

环境已配置时可跳过安装，直接使用可用的 `python3`；新环境建议创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 快速开始

首次操作请看 [人工审核上手指南](docs/user_guide.md)：以 desk 数据演示启动、检查、修改、审核和导出，附真实界面截图。

### 1. 选帧与审核

将 `/path/to/dataset` 替换为数据目录。两个终端使用同一输出目录，可边选边审，也可先选完再审核。使用虚拟环境时，在每个新终端执行 `source .venv/bin/activate`。

```bash
# 终端 A：选帧
bash scripts/run_cv.sh /path/to/dataset outputs/cv --jobs 4
```

```bash
# 终端 B：启动审核
python3 -m vive_correct review --out outputs/cv
```

打开 `http://127.0.0.1:18763`，核对物理左右手、标签四角和 XYZ 轴，再点击 **通过审核**。未完成的编辑可 **仅暂存**。新结果自动加入列表，不覆盖当前编辑。

### 2. 按需导出与查看

审核成功后点击 **导出 RRD 文件**，等待完成，再执行：

```bash
python3 -m vive_correct view outputs/cv
```

### 3. 按需重建六个文件

通过审核后执行，无需先导出 RRD：

```bash
# 生成独立修正文件，保留源数据
python3 -m vive_correct rebuild --out outputs/cv

# 需要替换源数据时执行；覆盖前自动保存初始备份
python3 -m vive_correct rebuild --out outputs/cv --in-place
```

重建范围为每条数据 `data/chunk-000/episode_000000/` 下的六个 Parquet：

- `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet`：左右腕部位置和方向。
- `action.parquet`、`observation.state.parquet`：其中的左右腕部字段。
- `observation.hand_left_points.parquet`、`observation.hand_right_points.parquet`：左右手关节点坐标。

只处理审核依据有效的数据，保留时间戳及其他字段；结果见 `outputs/cv/rebuild_manifest.json`。不修改 `observation.hand_skeleton.parquet`，不自动更新数据集 `meta/`。

### 只处理指定样本

新建 `episodes.json`，填写相对数据集根目录的路径，例如：

```json
["stove/L3-003-2026-10-30-17-24-33", "workbench/L3-003-2026-10-27-11-12-25"]
```

在选帧命令后加 `--episodes-file episodes.json`；省略则处理目录下全部样本。相同配置可续跑；输入、检测配置或计算代码变化时使用新输出目录。

坐标轴调整、批量导出、源文件重建和排错见 [审核与重建指南](docs/review_rebuild.md)。

## 使用边界

- 原点为 **50 × 30 mm 标签中心**，不是已标定的 Vive 内部原点；标签须与腕部设备刚性绑定。
- 自动候选的图像左右不代表物理左右手；几何检查通过也不代表实测精度达标。
- 默认头相机外参是安装近似，使用其他设备前应核对 [相机参数](calibrations/README.md)。当前识别输入固定为矫正左目，不能通过替换内参切换为眼动相机。
- 审核和导出只写输出目录；源文件覆盖须显式执行 `rebuild --in-place`。

## 文档导航

- [纯 CV 选帧](docs/cv_detection.md)：样本清单、扫描模式、检测调参与评估。
- [审核与重建](docs/review_rebuild.md)：坐标轴校准、RRD 导出、源文件覆盖与恢复。
- [运行与配置](vive_correct/README.md)：输入要求、可选模型路径及阶段产物。
- [字段与接口](docs/observations_schema.md)：坐标约定、审核 schema 和预览接口。
- [手部校准模型](docs/hand_pipeline.md)：刚性变换、缺失值处理与独立 MANUS 路径。
- [Rerun 与深度输入](ego_data/README_rerun.md)：原始数据查看、两种深度元数据名称及所需字段。
- [开发说明](docs/development.md)：模块职责、并发与数据约束、检查入口。

研究与历史记录：[手部安装标定](docs/hand_calibration.md)、[模型对比实验](docs/experiments.md)、[CV 历史验证](docs/cv_validation.md)。
