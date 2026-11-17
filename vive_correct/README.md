# 运行指南

安装与日常命令见 [项目首页](../README.md)。检测入口：`python3 -m vive_correct run`。全部命令见 `python3 -m vive_correct --help`。

## 输入

每条样本至少需要：

```text
<episode>/
├── videos/.../observation.images.headband_left_rgb_rectified/episode_000000.mp4
├── data/.../observation.images.headband_left_rgb_rectified.parquet
└── calibrations/rectify_params.yaml
```

视频分辨率须与标定一致。安装检查需要头部与双腕姿态；骨架叠图需要左右手 21 点 parquet。Rerun 完整导出还需要深度、掩码、深度内参、眼动视频与 gaze。

纯 CV 只接受矫正左目；缺失或解码失败直接报错，不回退原始左目。

## 配置档

| Profile | 说明 |
|---------|------|
| `cv` | 纯 OpenCV：连续跟踪 + 特征选帧；不调模型 |
| `astra`（默认） | Codex CLI 调用 `gpt-6-astra` |
| `legacy` | API（Qwen）；需 `openai.local.json` 或环境变量 |

预设在 [config/run.yaml](config/run.yaml)，命令行参数优先。几何阈值见 [config/estimate.yaml](config/estimate.yaml)，批处理见 [config/batch.yaml](config/batch.yaml)，CV 检测见 [config/cv.yaml](config/cv.yaml)。头相机外参见 [相机参数](../calibrations/README.md)。

### Astra 默认行为（摘要）

粗扫步长 15、补扫 3、Tracker 局部搜索 6 帧；首轮评审 4 候选/批，每样本最多 24 个几何合格候选；身份置信阈值 0.8；CPU 并发 10、模型并发 4（可用 `--jobs` / `--model-jobs`）。

`legacy` 使用更旧的搜索参数。只换通道、保留 Astra 搜索参数时用 `--transport api`。Codex 请求保存在 `codex_requests/`。

## 常用控制

```bash
python3 -m vive_correct run --help
```

- `--episodes-file FILE`：JSON 样本清单，保持顺序
- `--dry-run`：只预览计划（默认 `all` 阶段），不联网、不写结果
- `--profile cv` / `--model-review off`：纯 CV；详见 [纯 CV 指南](../docs/cv_detection.md)
- `--fixed-candidates`：关闭补扫与提前停止（对比实验）
- `--tracker-guided-frames N`：局部搜索帧数；`0` 关闭
- `--initial-mount FILE`：头相机外参
- `--stage …`：分阶段运行（见下）

## 模型识别流程（Astra / legacy）

1. 检查输入与模型连接，从视频生成候选
2. 像素边缘拟合四角 → PnP 求标签中心 → 11 帧窗口稳定性
3. 模型判断身份、可见性与物理左右手
4. 未得稳定双手时补扫一次（加密 + Tracker 区域）
5. 按质量选候选，再独立复核 Tracker 归属
6. 输出观测、缺失手侧、停止原因

模型框只限定搜索区域，不是角点或中心。双手可来自不同帧，不能当同时刻相对位姿。身份仍是模型判断，使用前按用途人工检查。

## 阶段与产物

模型路径默认 `--stage all`：估计 → 扫描 → 复核 → 安装检查 → 交付。纯 CV 的 `all` 为跟踪 + 特征 + 审核材料，不自动审核/修复/导出 Rerun。

```text
<out>/
├── run_plan.json              # 参数、样本、输入签名、代码哈希
├── estimates/                 # 初始估计 / CV 跟踪
├── features/                  # CV 特征图与审核模板
├── search/、search_refined/   # 粗扫与补扫（模型路径）
├── qwen_reviews/              # 模型评审
├── batch/
│   ├── results/               # 工作报告
│   ├── delivery/index.html    # 图片与精简报告入口
│   └── bilateral_results.zip
└── rerun/                     # 录制与转换日志
```

相同命令可恢复未完成运行。模型、参数、输入或代码变化后请换新输出目录。

```bash
python3 -m vive_correct run --out outputs/demo --stage gallery   # 只更新展示
python3 -m vive_correct run --out outputs/demo --stage rerun --open-rerun
```

缺少任一侧有效原点时不导出录制。交付字段见 [报告接口](../docs/observations_schema.md)。

## 审核与重建

人工审核、独立 RRD 导出、六文件备份覆盖及失败排查见 [审核与重建指南](../docs/review_rebuild.md)。普通检测与导出不改源文件；只有显式执行 `rebuild --in-place` 才覆盖源文件。
