# 纯 CV 指南

默认采用快速选帧，不调用模型、不运行整段身份跟踪。输出待人工确认的特征图、原帧、四角和候选池；之后的审核、RRD 与六文件重建流程不变。

## 一键检测

```bash
# 默认快速模式；只处理 JSON 清单中的数据
bash scripts/run_cv.sh /path/to/dataset outputs/cv_fast \
  --episodes-file /path/to/episodes.json --jobs 4

# 完整选帧：关闭快速预算与提前结束，恢复原有加密和逐帧补扫
bash scripts/run_cv.sh samples outputs/cv_full --feature-search full --jobs 4

# 完整选帧之前额外执行整段身份跟踪
bash scripts/run_cv.sh samples outputs/cv_tracked --stage all --feature-search full
```

脚本默认 `--profile cv --stage features --model-review off`，显式 `--stage` 可以覆盖默认值，日志写入 `<out>/run.log`。直接运行 `python3 -m vive_correct run --profile cv` 也默认选择 `features`；模型配置档仍默认 `all`。

清单为相对数据集根目录的路径数组，例如 `["desk/episode_a", "dining/episode_b"]`；省略 `--episodes-file` 则处理全部数据。

`--dry-run` 预览计划，`--max-frames 120` 可做短片冒烟。模式、参数、代码或输入变化时换新输出目录，避免把旧缓存当新结果。只有四条数据时，提高数据级 `--jobs` 到 4 以上不会增加实际并发。

### 输入与输出

```text
# 输入（缺矫正视频则报错，不回退原始左目）
<episode>/videos/.../headband_left_rgb_rectified/...
<episode>/data/.../headband_left_rgb_rectified.parquet
<episode>/calibrations/rectify_params.yaml

# 输出
<out>/run.log
<out>/estimates/<场景>/<样本>/          # 仅 --stage all/estimate/scan 生成
<out>/features/<场景>/<样本>/           # 特征图、原帧、四角、审核模板
<out>/batch/delivery/index.html         # 特征入口；显式跟踪后另有 tracked.html
```

退出码 0 不表示每条双侧正确或已验收。双手可来自不同帧，不能算同时刻相对位姿。

## 流程要点

**默认快速扫描**

- 每条最多检测 120 帧、扫描软时限 30 秒；单次检测或视频 I/O 不会被强行中断，图片落盘时间也可能使总耗时略超预算。
- 前 10 秒优先采样，并为后段保留预算；长视频自动增大采样步长，采样计划覆盖后段。达到时限可能提前结束，不能保证遍历整段视频；不进行全片逐帧兜底。
- 双侧候选均通过原有严格外观检查时提前结束；已有合格同帧组合优先，否则允许不同帧。仍须人工确认物理侧别和几何，不保证找到全片最高分。
- 未找到双手也输出审核材料，保留已有候选，可在审核工具中浏览整段视频补标。预算耗尽不等于源视频没有标签。

预算可通过 `--cv-config` 指定部分配置：

```yaml
feature_fast_max_samples: 120
feature_fast_timeout_s: 30.0
```

**完整扫描（`--feature-search full`）**

前 10 秒每 5 帧、后段每 `--coarse-stride` 帧（默认 15）扫描；缺手时后段每 5 帧补扫，再全段逐帧兜底。首轮不会因发现双手而立即停止，仍未检出则报告缺失，不降低外观门槛。

两种模式都优先满足 `pair_quality_ratio`（默认 0.70）的同帧组合。`features.json` 记录 `search_mode`、`scan_budget`、`stop_reason`、检测/解码帧数及各阶段耗时；日志定期显示进度。停止原因包括 `bilateral_candidates`、`sample_budget`、`time_budget`、`scan_complete`、`error`。

**可选身份跟踪**

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
