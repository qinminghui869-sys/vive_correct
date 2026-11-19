# 纯 CV 选帧与检测

默认用 OpenCV 快速选择标签参考帧，不调用模型、不运行整段身份跟踪。结果是候选材料，物理左右手与几何仍需人工确认。输入文件见 [运行指南](../vive_correct/README.md#输入要求)，后续操作见 [审核与重建](review_rebuild.md)。

以下参数示例默认使用左目模式。眼动模式在选帧时增加 `--coordinate-mode eye-reference`，使用原始眼动视频统一去畸变，详见 [模式选择与启动](../README.md#选择模式)；不支持 `--stage all` 的图像身份跟踪路径。

## 选择样本与扫描模式

```bash
# 快速模式：只处理清单中的样本
bash scripts/run_cv.sh /path/to/dataset outputs/cv \
  --episodes-file /path/to/episodes.json --jobs 4

# 完整选帧：使用独立输出目录
bash scripts/run_cv.sh /path/to/dataset outputs/cv_full \
  --feature-search full --jobs 4

# 完整选帧，并额外运行整段身份跟踪
bash scripts/run_cv.sh /path/to/dataset outputs/cv_tracked \
  --stage all --feature-search full --jobs 4
```

清单格式见 [常用配置](../vive_correct/README.md#常用配置)，路径相对本次数据集根目录填写。

省略 `--episodes-file` 处理全部样本；加 `--dry-run` 只预览计划。`--max-frames 120` 将视频限制为前 120 帧，仅适合短片冒烟测试，区别于快速模式在全时段内最多检测 120 帧。

脚本默认 `--profile cv --stage features --model-review off`，显式 `--stage` 可覆盖，日志追加到 `<out>/run.log`。相同配置可续跑；输入、代码或检测参数变化时换新输出目录。有效数据并发不超过样本数。

### 左目快速模式（默认）

- 每条最多检测 120 帧，扫描软时限 30 秒。单次检测、视频 I/O 与结果落盘不会被强行中断，总耗时可能超出该时限。
- 先采样首尾，再递归填补最大时间间隔，尽早覆盖前、中、后段；保持原采样集合和预算，不保证逐帧遍历全片。
- 找到合格双侧候选后可提前结束；左右可来自不同帧。
- 预算耗尽不代表视频中没有标签，已有候选可供人工补标。

可用 `--cv-config override.yaml` 覆盖预算：

```yaml
feature_fast_max_samples: 120
feature_fast_timeout_s: 30.0
```

### 左目完整模式

`--feature-search full` 恢复完整补扫：前 10 秒每 5 帧、后段每 `--coarse-stride` 帧扫描，缺手再对后段每 5 帧补扫，最后逐帧兜底。首轮不因发现双手立即停止，也不降低外观门槛。

### 眼动固定参考系：标签共同帧

- 从视频 **10 秒**开始，按帧号递增 **每隔 5 帧**抽样，最多 **100 帧**。首帧为 `ceil(10 × FPS)`；30 FPS 视频检查 300、305、…、795 帧，遇片尾结束。短于起始时间时不回退到前段。
- `fast` 和 `full` 均使用上述固定采样，不使用左目预算、30 秒软时限或补扫策略。`--max-frames` 仍限制可访问的原视频范围，不能当作采样数量。
- 仅依据独立高质量标签，要求左右候选位于画面不同半区；跨中线标签不分配侧别，不使用青色灯、安装座或双腕上下文。
- 双标签帧优先，其次单标签帧；同类按较弱标签质量排序。单标签帧保留为共同参考帧候选，另一半手部是否可见必须人工确认，缺失四角保持为空。两个标签都在同半区时只保留该半区的最佳标签。
- 只搜索、排序整帧候选，不跨帧组合左右结果。没有高质量标签时保持待处理，可人工换帧。

`feature_plan.json.sampling_policy` 记录实际采样及配对参数；`features.json` 记录 `selection_policy="joint_frame_only"`、实际采样帧号和 `reference_candidates`。审核列表按帧展示，一次选择同时加载两侧。图像半区不是物理手侧，仍需人工确认。缺少完整手点或骨架的样本整条 `skipped`。

左目仍按 `pair_quality_ratio` 优先选同帧组合，参数见 [cv.yaml](../vive_correct/config/cv.yaml)。

## 产物与判断

- `features/<场景>/<样本>/`：特征图、原帧、四角和候选池。
- `features.json`：扫描模式、预算、停止原因、实际采样帧号（`sampled_frame_indices`）、检测层与选帧层拒绝计数（`detector_rejections`／`feature_rejections`）和阶段耗时。`decoded_frames` 仅统计显式读取，不含视频跳转时解码器内部读取。
- `batch/delivery/index.html`：静态特征展示；显式跟踪后另有 `tracked.html`。
- `estimates/`：显式估计或跟踪阶段的结果。

停止原因包括 `bilateral_candidates`、`sample_budget`、`time_budget`、`scan_complete` 、`video_ends_before_10s` 和 `error`。单条发布后即可在浏览器审核，无需等整批结束。退出码 0 不代表双侧检出正确或已通过审核。

未跟踪候选的左右归属仅为图像位置提示，必须人工确认物理手侧。两种模式均记录 `image_half_unverified`。左目原始候选可能来自不同帧，不能作为同时刻相对位姿；眼动候选始终以共同帧为单位。四角坐标约定见 [字段与接口](observations_schema.md#坐标约定)。

## 检测与可选跟踪

检测结合白色标签和中性黑色外壳，检查环带、逐边黑色比例与凸包包围；选帧还限制标签内部黑像素和最小角点夹角。外观相似不等于 VIVE 身份正确。

重复白块只在同一黑色外壳的凸包内计数，避免把背景桌布误当键盘。左目中，外壳与手套相连、边界超出搜索区域时，若标签与黑色包围检查仍通过，可保留为 `review_candidate`，并提示 `body_extent_unresolved`；此类候选不触发双侧提前结束，需人工核对设备身份与四角。

显式启用跟踪时，首个可靠双手窗口假设双手未交叉且相机未镜像；初始化需连续 `tracking_window` 帧可靠关联。之后逐帧几何验收，短缺失最多用 3 帧光流桥接，桥接帧不作观测；身份不确定时输出缺失，不按半区重新命名。

## 漏检排查

按 `features.json` 的记录判断原因，再决定是否调参：

1. 查看 `error` 和 `stop_reason`。输入缺失或解码失败先修复输入；`time_budget`／`sample_budget` 表示预算用尽，不表示没有标签。
2. 查看 `sampled_frame_indices`，确认标签清晰的时段是否采样。未覆盖时可人工换帧；左目还可在新输出目录使用 `--feature-search full`。
3. 查看 `detector_rejections` 与 `feature_rejections`。前者是像素检测拒绝原因，后者是选帧过滤原因；计数对应候选或轮廓，不是漏检手数，也不能单独判断某个标签。
4. 检查候选图中的四角与黑色包围。`body_extent_unresolved` 已保留为待复核候选；其他拒绝需结合原帧判断，再按下节一次调整一类参数。

旧报告没有诊断字段时，使用当前代码和新输出目录重跑；不能通过重载页面补出未保存的检测结果。

## 调参与评估

`--cv-config` 只需填写覆盖项。一次调整一类参数，先用调参集，再评留出集：

- 彩色反光误检：检查 `max_saturation`、`max_chroma`；灰白背景误检：检查 `min_contrast`、`min_dark_fraction`、`min_side_dark`。
- 白色覆盖不足：检查 `min_white_fraction`；阴影漏检：先检查 `brightness_quantiles`、`side_contrast`、`thresholds`。
- 四角偏移：检查 `refine_radius_scale`、`max_line_residual`、`max_refine_shift`。
- 检测耗时或远距小标签：检查 `proposal_max_width`；身份不稳：检查 `association_fraction`、`association_margin`，放宽关联可能增加错侧。

左目模式的已有候选池可在输入签名一致时收紧门槛重筛（该独立工具不支持眼动模式）：

```bash
python3 -m vive_correct.cv_features --source outputs/cv \
  --out outputs/cv_refiltered --no-rescan-missing --jobs 4
```

离线评估：

```bash
python3 -m vive_correct.cv_evaluation prepare \
  --dataset /path/to/dataset --out outputs/cv_evaluation
python3 -m vive_correct.cv_evaluation score --dataset /path/to/dataset \
  --annotations outputs/cv_evaluation/annotations.json \
  --results outputs/cv/batch/delivery/summary.json \
  --out outputs/cv_evaluation/scored
```

`prepare` 按场景划分调参／留出集并生成标注模板；只评分 `human_reviewed: true` 的帧，以 IoU ≥ 0.5 一对一匹配。不能将旧检测结果当人工真值。验收同时报告留出集误检、错侧、覆盖率、角点误差、样本量与未确认项；历史结果见 [验证记录](cv_validation.md)。
