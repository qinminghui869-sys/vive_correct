# Vive Correct

从头戴相机视频识别腕部标签，计算标签中心在相机坐标系中的位置。支持左右手复核、自动补扫，以及 Vive 轨迹和手部骨架的 Rerun 展示。

## 安装

需要 Python 3.10+。纯 CV 模式不需要模型密钥或 Codex 登录。默认识别模型为 `gpt-6-astra`，通过已安装并登录的 Codex CLI 调用。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 运行

在项目根目录执行：

```bash
python3 -m vive_correct run --dataset /path/to/dataset --out outputs/demo
```

`--dataset` 可指定单条样本或数据集目录。默认开启自动补扫，完成后生成标签中心、复核记录和图片报告。

常用选项：

- `--episodes-file episodes.json`：只处理清单中的样本，如 `["desk/sample_01"]`。
- `--dry-run`：离线预览参数和样本清单，不调用模型或写入结果。
- `--open-rerun`：导出并打开录制，需要图形环境；`--rerun` 仅导出。
- `--profile legacy`：使用 API 模式；配置方法见[运行指南](vive_correct/README.md#模型与参数)。

用浏览器打开 `outputs/demo/batch/delivery/index.html` 查看报告。已导出 Rerun 录制时，可用以下命令打开：

```bash
python3 -m vive_correct view outputs/demo
```

相同命令可恢复未完成的运行。更换模型、参数、输入或代码后，请使用新的输出目录。

## 纯 OpenCV 模式

```bash
bash scripts/run_cv.sh samples outputs/cv_full --jobs 4
```

脚本强制纯 CV `--stage all`，顺序执行连续跟踪与几何求解、特征图选帧及审核材料生成；完整日志保存在 `outputs/cv_full/run.log`。只读取校正左目视频，不回退原始左目。

打开 `outputs/cv_full/batch/delivery/index.html` 查看特征图，点击其中链接查看独立的 `tracked.html` 跟踪报告。每侧保留帧序号、四角、标签中心、无标注原帧、诊断图和人工审核模板。

不调用模型，使用白色／黑色外壳检查、四角精修和连续光流关联，每手保留一个最佳观测。特征扫描首轮前10秒每5帧、后段每15帧；仍缺任意一侧时先补扫后段每5帧，再按需全段逐帧补扫。连续跟踪以5帧可靠双侧关联初始化身份；初始化后逐帧验收几何，不再要求稳定窗口。首次可靠双手窗口必须未交叉；身份不确定时输出缺失。如需优先导出每段的左右候选特征图，使用 Python 入口并指定 `--profile cv --stage features`；同帧双手优先，缺失输入明确标记。参数、诊断字段和人工验收方法见[纯 CV 指南](docs/cv_detection.md)。

## 查看手部数据

```bash
python3 open_manus.py /path/to/episode
```

查看 MANUS 原始节点、RGB 和可用点云；自动核查侧别，缺深度时回退为局部节点视图。此入口不应用标签位置修正。查看修复效果应使用上面的 `view` 命令打开已生成的录制。

手动调整腕部使用 `python3 open_file.py /path/to/episode`。操作选项见 [Rerun 说明](ego_data/README_rerun.md)。所有处理结果另存，源数据不修改。

## 结果含义

- 输出原点是 **50 × 30 mm 标签的中心**，标签到 Vive 内部原点的偏移尚未标定。
- 缺少安装方向参考时只输出中心，不生成设备坐标轴。
- 重投影误差和检出率不能代替实测位置精度。
- 默认相机外参是安装近似；其他安装应通过 `--initial-mount` 提供外参。
- 自动导出先核查手侧、校验标签区域深度，再计算每只手的固定世界偏移；未通过项保留未验证标记。
- 可选手部刚体标定需独立参考和验证；当前结果不代表 MANO 拟合或全部样本稳定贴合。

## 文档

- [运行指南](vive_correct/README.md)：输入、配置、阶段与产物
- [报告字段](docs/observations_schema.md)：坐标、质量与标定状态
- [模型对比实验](docs/experiments.md)：固定候选池的复核对比
- [手部流程 DAG](docs/hand_pipeline.md)：文件、坐标映射和修复分支
- [开发说明](docs/development.md)：模块与测试
