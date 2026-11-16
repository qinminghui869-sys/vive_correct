# 纯 CV 白标签检测与调参

## 运行

```bash
python3 -m vive_correct run --profile cv --dataset samples --out outputs/cv_run --jobs 4
```

`--profile cv` 强制关闭模型，即使同时给出 `--model-review on` 也不会调用模型。普通 `--model-review off` 使用同一连续跟踪流程。为了保留旧固定候选池消融实验，`--model-review off --fixed-candidates` 仍走旧的半区候选流程；需要新流程时显式选择 `--profile cv`。

输出 `batch/delivery/index.html`、`summary.json`，以及 `estimates/<场景>/<样本>/sticker_pose_report_new.json`。每手最多一个最佳观测，包含原图、诊断叠图、原始像素四角、标签中心、质量分数和身份来源。两手可能来自不同时间，不能据此计算同时刻相对位置。

`--max-frames N` 限制 CV 流程处理的连续帧数；`--stride` 不跳过身份跟踪帧。`--jobs` 控制按视频并发。重新运行同一命令可恢复结果，代码、输入或配置变化需要新的输出目录。

本入口用于标签观测，不直接执行手部轨迹修复或 Rerun 导出；这些仍需现有独立身份核验流程。旧 Astra/API 入口保持可用。

## 全数据特征图与同帧优先

只生成候选特征图（不要求通过 11 帧稳定窗口或求解姿态）：

```bash
python3 -m vive_correct run --profile cv --stage features --dataset samples \
  --out outputs/cv_box40_results --episodes-file outputs/cv_original40.json --jobs 8 --coarse-stride 15
```

扫描覆盖整段视频，默认每 15 帧取候选；缺某侧时，自动以 5 帧、1 帧间隔补扫尚未检测的帧。`--max-frames` 仍可限制处理范围。此模式只允许使用校正左目视频。缺失、无法解码或选中帧读取失败时，在 ERROR 日志及 features.json 中记录错误，绝不回退原始左目视频；批处理返回非零退出码。原始视频生成的旧候选缓存也禁止复用。特征坐标对应校正视频像素，不直接视为几何求解结果。不会因缺时间戳或相机标定而跳过特征图。普通 CV `--stage all` 也会在姿态估计后生成这些特征图；姿态结果仍单独保存，不将特征候选升格为有效姿态。

每段保存 `features/<场景>/<样本>/left_feature.jpg`、`right_feature.jpg`、带左右轮廓的原始帧、`features.json`、可追溯的全部通过门槛候选 `candidate_pool.json`，总览仍为 `batch/delivery/index.html`。报告区分：

- `appearance_candidate`：通过当前外观门槛，尚未核验真实物理侧别。
- `review_candidate`：仅通过稍宽的候选门槛（饱和度 0.30、色度 25、白色比例 0.68、对比度 22），记录未通过的标准规则。
- 未检出／缺视频：输出明确的诊断占位图，没有虚构四角；不计入双侧特征覆盖率。

特征候选按画面半区提供左右提示，双手交叉时可能与物理侧别不同，所有特征图均标记 `image_half_unverified`。需要物理侧别关联和几何有效观测时，仍使用完整连续跟踪流程。两类结果不能混作人工真值。

每手分别计算最佳质量；若某同帧双手候选各自达到该手独立最佳分数的 `pair_quality_ratio`（默认 0.70），优先选其中较弱一手质量最高的同帧组合。否则保留各手独立最佳帧。连续跟踪的有效观测也采用同样的同帧偏好。

## 黑色包围特征与参数

白标签本体不应大部分是黑色，所以外壳比例的分母明确排除标签本身：

- `shell_black_fraction`：以标签中心为基准放大至 `shell_scale=1.8` 的四边形，减去标签本体后，实际黑像素所占比例。
- `surround_black_fraction`：从 1.8 倍到 `context_scale=2.8` 倍四边形之间的外围区域黑像素比例。
- `enclosed_fraction`：与标签边缘相邻的真实黑色连通区域，其凸包覆盖标签面积的比例。计算前不会先把白标签涂黑，因此不是自证包围。
- `label_black_fraction`：标签内部低于黑色阈值的原始像素比例，仅作诊断；阴影也可能使此值偏高。

外壳黑色使用 `gray < dark_threshold=115` 且 `max(B,G,R) < black_channel_max=150`；外围背景单独使用 `surround_dark_threshold=105`，并保留同一通道上限。可分别调整外围和外壳对黑色的判定，避免青色灯被当成黑色。

当一条边偏亮时，允许另外三条边满足梯度／对比度要求，并且 `shell_black_fraction >= min_shell_fraction=0.35`、`enclosed_fraction >= min_enclosed_fraction=0.70` 来补偿该弱边。两条边缺少支持仍拒绝。此补偿还必须通过下述四边黑盒门槛；一侧缺少实体黑色区域时仍然拒绝。外壳比外围更黑时给予小幅加分，不把外围明暗作为一票否决条件。

为抑制桌布、灰色反光和相框误检，所有正式外观候选新增不可由 LED 或宽松预览配置绕过的**内层黑盒门槛**：

- `box_scale=1.4`：检查 1.0–1.4 倍轮廓之间的紧邻环带，并分成四边梯形区域。
- `box_dark_threshold=105`、`box_channel_max=125`、`box_color_spread=35`：黑像素必须同时满足灰度 < 105、最大通道 < 125、最大与最小通道差 ≤ 35，避免深蓝物体被当作黑盒。
- `box_min_black=0.75`、`box_min_side_black=0.40`：总体至少 75% 且每边至少 40% 是中性黑色，不允许平均值掩盖一侧缺口。
- `box_min_enclosure=0.95`：真实中性黑色连通分量的凸包至少覆盖 95% 标签面积；这一项只作联合条件，单独的凸包包围不充分。
- 未通过返回 `unsupported_black_box`。报告保存 `box_black_fraction`、`box_side_black`、`box_enclosed_fraction` 和 `box_supported`，排序还使用总体和最弱一边的黑色支持度。

最终特征图还要求 `label_black_fraction <= max_label_black_fraction=0.10`，避免把中性灰黑表面误作白标签。该门槛只用于高质量特征选帧，基础检测和连续跟踪仍保留暗处白标签候选。它可能降低严重阴影片段的特征覆盖率，应优先换帧而不是强行补齐。

最终特征图还要求 `feature_min_corner_angle=25` 度，拒绝近退化的斜长反光四边形；不强制 5:3 图像长宽比。

已有候选池可通过以下命令收紧内部黑像素与最小角点夹角门槛；输入签名和其他参数必须一致，不满足时拒绝复用。默认缺一侧时自动重新扫描；`--no-rescan-missing` 明确保留池内无有效候选的缺失，`--expanded-source` 可复用同源视频的补扫候选。输出记录来源和配置，不能用缓存来放宽召回参数：

```bash
python3 -m vive_correct.cv_features --source outputs/cv_box40_results \
  --expanded-source outputs/cv_box40_delivery \
  --out outputs/cv_box40_report --no-rescan-missing --jobs 8
```

这些参数基于当前视频里的背景误检调试，不是留出集精度保证。较强透视、反光或暗外壳偏色可能导致真实标签被拒绝；优先从其他帧寻找四边完整的标签，不靠降低 15 px 短边要求补齐结果。

特征候选还检查主体紧凑程度：黑色主体（连接白标签用于分量分析）面积与标签面积之比不得超过 `body_max_area_ratio=30`，主体填充率至少 `body_min_fill=0.28`，搜索半径为短边的 `body_search_scale=5` 倍（限于 90–280 px）。周围出现至少 `body_max_white_neighbors=4` 个相似大小的白色连通区域时，排除键盘按键等重复纹理；青色灯也不能豁免重复纹理排除。

未跟踪的特征图候选只在中心纵坐标大于图高 `feature_min_y_fraction=0.40` 的范围选取；该先验可配置，不限制已跟踪的手跨越画面区域。黑色主体触及原始图像边界时拒绝（`body_fully_visible=false`），避免边缘遥控器或被裁切的黑色背景误检。主体不紧凑的候选直接排除；附近青色灯不能豁免主体和黑盒检查。

特征图左侧展示标签及外壳原始裁剪；右侧绿色是外壳环带中的黑像素，蓝色是外围黑像素，黄色是标签轮廓。图上打印四项比例，便于人工检查每次调参影响。

## 身份约定与拒绝行为

第一段可靠的双标签窗口必须双手未交叉，画面左侧对应物理左手，右侧对应物理右手。相机必须没有镜像。检测本身无法证明这个约定。标签需要连续通过默认 11 帧窗口，之后用正反向 LK 光流和唯一轮廓关联传播身份。

交叉后不按半区重新命名。检测短暂缺失时允许最多 3 帧光流桥接，但这些帧不能作为观测；光流失败、多个候选同样接近、两条轨迹争抢同一候选或跳帧时输出未知。完全丢失的身份不会再凭半区重新初始化，因为“首次未交叉”不能证明后来的姿势。某手已有的最佳结果可以保留。

标签四角或外侧采样带出画、低色度覆盖不足、至少两条边缺少支持、几何及稳定性检查失败时拒绝。单条弱边可由黑色包围证据补偿。正式输出最短边至少 15 px。相似白色物体也可能满足这些物理外观规则，因此结果始终保留 `human_verified=false` 和身份假设。

## 参数及调整顺序

默认参数在 `vive_correct/config/cv.yaml`。可创建仅包含修改项的 YAML，通过 `--cv-config /path/to/override.yaml` 覆盖。未知参数、非有限值、非法比例会报错。完整生效参数与 SHA-256 记录在报告和运行计划中。

- **彩色反光误检**：降低 `max_saturation`（默认 0.26）或 `max_chroma`（默认 22）。Lab 使用 OpenCV 8 位编码，色度距离为 `sqrt((a-128)^2+(b-128)^2)`。
- **白色覆盖不够**：提高 `min_white_fraction`（默认 0.75）；它测量收缩后内部的合格像素比例，允许少量黑字。提高后会增加脏污或印字标签的漏检。白色比例还排除相对内部亮度明显偏暗的文字，不再把所有中性黑像素都算作白色。
- **灰白背景误检**：提高 `min_contrast`（默认灰度差 28）、`min_dark_fraction`（0.55）或 `min_side_dark`（每边 0.25）。青色灯不豁免这些门槛。
- **阴影标签漏检**：检查 `brightness_quantiles` 和 `side_contrast`，先调整 `thresholds` 及局部对比度分支；不要直接放宽色度或短边门槛。召回使用内部轮廓，避免亮背景将暗标签包住。
- **四角偏移**：查看 `refinement_shift`、`edge_support`、`side_contrast`。`approx_epsilons` 控制轮廓近似；`refine_radius_scale` 控制法向搜索半径，实际限于 2–6 px；`max_line_residual` 控制拟合残差，`max_refine_shift` 限制移动量。
- **速度**：`proposal_max_width` 默认 960，只缩小候选生成图像；候选四角映射回原始分辨率后，颜色验收、精修、几何和最短边检查均使用原始像素。远距离小标签召回不足时可提高该值，代价是更慢。
- **大小变化**：`border_scale` 按短边缩放外侧采样距离（限制 2–6 px）；`min_area`、`max_area_fraction` 限定候选面积。不要用图像长宽比强制 5:3，斜视会改变该比例。
- **身份不稳定**：查看 `tracking_events`；`association_fraction` 是按短边归一化的最大关联距离，`association_margin` 要求最优和次优候选有足够差距。放宽关联会提高身份交换风险。

一次只调整一类参数，在调参视频上比较结果，固定配置后才评估留出视频。白色分数在原图上测量，不对增强后的图像评分。最终排序以白色、外壳和边缘支持为主，面积和青色灯奖励均有上限；最佳帧排序还按光流正反误差和窗口运动量施加小幅惩罚（`selection_score`），不使用模型置信度。

## 验证工具

```bash
python3 -m vive_correct.cv_evaluation prepare --dataset samples --out outputs/cv_evaluation
python3 -m vive_correct.cv_evaluation score --dataset samples \
  --annotations outputs/cv_evaluation/annotations.json \
  --baseline-module /path/to/baseline_detector.py \
  --results outputs/cv_run/batch/delivery/summary.json \
  --out outputs/cv_evaluation/scored
```

`prepare` 按场景排序，将完整视频按 70%/30% 分到 tuning/holdout，每段列出 12 个均匀采样帧。当前四类场景各 10 段，即 28 段调参、12 段留出。缺失视频需要先补齐输入或从标注清单明确排除；不能把解码失败当作负样本。

人工检查每帧所有完整可见标签，填写 `labels: [{"side":"left", "quad":[[x,y],...]}]`，四角按轮廓循环排列，坐标必须是原始分辨率。经确认无标签的困难负例用空数组。确认后才设 `human_reviewed: true`。补充偏色、模糊、斜视、遮挡和交叉片段，并为最终选中的帧补充标注。不得将旧检测结果直接复制为人工真值。

`score` 只评估已确认帧，拒绝同一视频跨集合。IoU ≥ 0.5 做一对一匹配，报告 TP/FP/FN、精确率、召回率、循环对齐角点误差和耗时；可加载保存的旧检测模块作同帧比较。最终结果审计单独报告错误标签、错侧及未标注项。未标注时指标为 null，验收为 pending。

最佳帧输出无法测量完整视频的身份切换次数；仅有最佳帧时工具对此返回 null，不伪造零。估计报告另存 `track_observations`，可用 `--tracking-report /path/to/sticker_pose_report_new.json` 对已人工标注的帧计算侧别切换和错侧次数；未标注间隔不算已验证。连续交叉、歧义和跳帧行为也由自动测试覆盖，真实侧别仍需人工对照视频。验收需同时满足：留出集已确认最终结果无误检或错侧、有效手侧覆盖不下降、角点中位误差不高于基线，并公开未确认项和样本量。


## 输出与人工审核（当前协议 v2）

建议交付时保留以下文件：

- `features.json`：每侧 `frame_index`（0 起始）、`quad`（四个浮点像素点）、`label_center_px`、图像宽高、输入视频、配置与质量诊断、缺失/错误原因及 `observation_id`。
- `frame_NNNNNN_clean.jpg`：无标注原帧，检查真实标签边缘与场景上下文。
- `frame_NNNNNN.jpg`：带四角 q0～q3、中心十字和侧别提示的全帧叠图。
- `left_feature.jpg` / `right_feature.jpg`：局部标签与黑区域诊断；缺失占位图不是检测结果。
- `human_review.json`：人工审核模板，默认 pending。重复执行不会覆盖已有审核记录。
- `batch/delivery/index.html` 和 `features_summary.json`：浏览入口及批量索引。

`candidate_pool.json`、运行计划与配置哈希建议归档，便于重选帧和复现，不必作为普通审核者的主要入口。

坐标约定：校正视频左上为原点，x 向右、y 向下，单位像素。quad 按轮廓循环排序，q0 不保证是左上角，也不代表实物固定朝向。标签中心是 q0-q2 与 q1-q3 对角线交点，不是四点均值，不是 VIVE 内部原点。标签中心与严格几何输出的三维坐标应按各自坐标系使用。

人工审核流程：

1. 打开 HTML，核对帧序号；查看无标注原帧，确认白色区域位于真实 VIVE 机身，而非桌面、剪刀孔或其他白标签。
2. 对照叠图 q0～q3 与局部特征图，检查四角是否落在标签边缘、是否遮挡或裁切、中心是否合理。
3. 编辑对应 `human_review.json`，先确认 observation_id 与当前 features.json 一致；旧记录不匹配时不得复用。填写 status 为 accepted/rejected/uncertain，分别填写 label_correct、corners_correct；必要时在 corrected_quad_px 记录同顺序的修正四角。
4. 物理侧别必须结合视频上下文确认，填写 physical_side 为 left/right/unknown，不能只根据画面左右确认；填写 reviewer、reviewed_at（带时区的 ISO 时间）与 notes。
5. 审核记录与算法输出分开保存。模板不会自动修改算法四角、中心或 human_verified；修正四角需重新计算中心，不能沿用旧中心。尚未核验物理侧别时不得把候选接受解释为左右手身份已验证。

当前方案的数据流调整为：校正视频 → 检测与选帧 → 帧序号/四角/中心及图像 → 独立人工审核；校正视频不可用 → ERROR 日志与缺失报告。没有原始左目回退分支。历史40条报告与 PDF 不会因代码更新而自动改写，生成新版需使用新的输出目录。
