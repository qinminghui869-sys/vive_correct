# 纯 CV 指南

不调用模型：白色／黑色外壳外观检查 → 四角精修 → 连续光流关联；每手保留一个最佳观测，供人工审核与后续位置修正。

## 一键检测

```bash
bash scripts/run_cv.sh samples outputs/cv_full --jobs 4
```

强制 `--profile cv --stage all --model-review off`，顺序跑：连续跟踪与几何 → 特征选帧 → 审核材料。日志写入 `<out>/run.log`。

```bash
# 限样本 / 覆盖参数 / 短片冒烟（须用单独输出目录）
bash scripts/run_cv.sh samples outputs/cv_full40 \
  --episodes-file outputs/cv_original40.json --jobs 4
bash scripts/run_cv.sh samples outputs/cv_smoke --max-frames 120 --jobs 2

# 只出特征图（不做身份跟踪）
python3 -m vive_correct run --profile cv --stage features \
  --dataset samples --out outputs/cv_features --jobs 8
```

`--dry-run` 只检查运行计划。配置变化、截断结果与完整结果不要混用同一输出目录。

### 输入与输出

```text
# 输入（缺矫正视频则报错，不回退原始左目）
<episode>/videos/.../headband_left_rgb_rectified/...
<episode>/data/.../headband_left_rgb_rectified.parquet
<episode>/calibrations/rectify_params.yaml

# 输出
<out>/run.log
<out>/estimates/<场景>/<样本>/          # 跟踪与几何
<out>/features/<场景>/<样本>/           # 特征图、原帧、四角、审核模板
<out>/batch/delivery/index.html         # 特征入口 → tracked.html
```

退出码 0 不表示每条双侧正确或已验收。双手可来自不同帧，不能算同时刻相对位姿。

## 流程要点

**特征扫描（三阶段）**

1. 前 10 秒每 5 帧，之后每 `--coarse-stride` 帧（默认 15）
2. 仍缺任一侧 → 后段每 5 帧补扫
3. 仍缺 → 全段逐帧；仍失败则报告缺失，不降外观门槛

同帧双手优先：若两侧都达到各自最佳分的 `pair_quality_ratio`（默认 0.70），选同帧组合。`features.json` 记录 FPS、抽样阶段与诊断。

**身份跟踪**

- 首个可靠双手窗口假设未交叉（画面左→物理左）；相机不得镜像
- 初始化需连续 `tracking_window` 帧（默认 5）可靠双侧关联
- 之后逐帧几何验收，不再要求稳定窗口；短暂缺失最多 3 帧光流桥接（桥接帧不作观测）
- 身份不确定时输出缺失，不凭半区重新命名

**外观门槛（摘要）**

白标签 + 中性黑外壳。正式候选须过内层黑盒门槛（环带黑色比例、四边、凸包包围）；特征选帧另限标签内黑像素与最小角点夹角。诊断字段含 `shell_black_fraction`、`box_*`、`enclosed_fraction` 等。完整默认值见 [config/cv.yaml](../vive_correct/config/cv.yaml)。

未跟踪特征图仅作画面半区提示，标记 `image_half_unverified`；与跟踪观测不能混作真值。

## 调参

用 `--cv-config override.yaml` 只写修改项。一次只改一类，在调参集上看效果后再评留出集。

| 问题 | 优先调整 |
|------|----------|
| 彩色反光误检 | 降低 `max_saturation` / `max_chroma` |
| 白覆盖不够 | 提高 `min_white_fraction` |
| 灰白背景误检 | 提高 `min_contrast`、`min_dark_fraction`、`min_side_dark` |
| 阴影漏检 | 查 `brightness_quantiles` / `side_contrast`，调 `thresholds`；勿先放宽色度或短边 |
| 四角偏移 | `refine_radius_scale`、`max_line_residual`、`max_refine_shift` |
| 太慢 / 远距小标签 | `proposal_max_width`（默认 960） |
| 身份不稳 | `association_fraction`、`association_margin`（放宽会增交换风险） |

已有候选池可收紧门槛重筛（输入签名须一致）：

```bash
python3 -m vive_correct.cv_features --source outputs/cv_box40_results \
  --out outputs/cv_box40_report --no-rescan-missing --jobs 8
```

离线评分与标注约定见下方「验证」；历史跑批记录见 [cv_validation.md](cv_validation.md)。

## 人工审核与重建

检测结果默认未经人工确认。标签坐标以矫正左目左上角为原点，x 向右、y 向下；四角按轮廓循环排列，q0 不一定是左上角。中心取对角线交点。

使用 [审核与重建指南](review_rebuild.md) 完成确认、RRD 导出和可选的备份覆盖。物理左右手须结合视频上下文确认，不能只看图像半区。

## 验证

```bash
python3 -m vive_correct.cv_evaluation prepare --dataset samples --out outputs/cv_evaluation
python3 -m vive_correct.cv_evaluation score --dataset samples \
  --annotations outputs/cv_evaluation/annotations.json \
  --results outputs/cv_run/batch/delivery/summary.json \
  --out outputs/cv_evaluation/scored
```

`prepare` 按场景划分调参/留出并列出采样帧。只评 `human_reviewed: true` 的帧；IoU≥0.5 一对一匹配。不得把旧检测结果当人工真值。验收需同时看留出集误检/错侧、覆盖与角点误差，并公开未确认项与样本量。
