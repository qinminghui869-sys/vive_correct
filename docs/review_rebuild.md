# 导出、重建与排错

启动和日常审核见 [快速开始](../README.md#快速开始) 与 [人工审核指南](user_guide.md)。以下命令在仓库根目录执行，`outputs/stove_review` 应替换为实际审核输出目录。

## 状态与处理范围

- **待审核**：包括草稿、单侧通过、依据过期。不能导出或重建。
- **已通过**：双手和依据均有效，可生成修正文件或 RRD。
- **不通过／excluded**：整条排除，批量任务记录跳过；跳过本身不算失败。

排除清单为 `<out>/rejected_episodes.json`，记录样本目录、视频、特征文件的绝对路径，以及审核人、原因和 UTC 时间。清单在同一输出目录内按样本路径去重，服务重启、重新发布候选不会解除排除。其他输出目录有独立的审核状态。

排除已导出的条目会使其索引失效，历史文件保留；已完成的源数据覆盖不会自动回滚。后台排队或处理中的条目不能修改决定，须等待任务结束。

## 导出与查看

```bash
python3 -m vive_correct repair --out outputs/stove_review --export-jobs 1
python3 -m vive_correct view outputs/stove_review
```

`repair` 不代替人工审核。`view` 只打开已完成的录制，不触发导出。输入、审核、标定、计算代码和产物均未变化时可复用结果。深度输入不足时使用 RGB 路径；完整录制输入见 [RRD 说明](../ego_data/README_rerun.md#完整-rrd-的输入)。

## 重建与备份

```bash
# 独立修正文件
python3 -m vive_correct rebuild --out outputs/stove_review

# 可选：备份后覆盖源文件
python3 -m vive_correct rebuild --out outputs/stove_review --in-place
```

指定单条时加 `--episode /path/to/episode`。`review`、`repair`、`rebuild` 如使用自定义外参，必须传入同一 `--initial-mount FILE`。

重建范围为每条数据 `data/chunk-000/episode_000000/` 下的六个文件：

- `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet`：腕部位置和方向。
- `action.parquet`、`observation.state.parquet`：其中的左右腕部字段。
- `observation.hand_left_points.parquet`、`observation.hand_right_points.parquet`：手部关节点。

保留行数、时间戳、字段类型、Arrow 元数据及非目标字段。缺失或同步失败的坐标组保持原值，并记录跳过数量。不修改 `observation.hand_skeleton.parquet`。

`--in-place` 先生成并校验全部六文件，再备份、登记和替换。首次 `.parquet.bak` 不覆盖；重复重建始终使用原始基线。失败回滚，中断恢复依据 `<episode>/.vive_rebuild/` 的事务记录完成。不要删除备份或事务记录绕过冲突检查。

### 数据集元数据

当前不更新 `meta/`。用于训练交付时，按原数据集口径重新计算 `episodes_stats.jsonl` 中已修改维度的统计量，否则可能影响归一化；若 `readme.json` 记录大小或数量，也需刷新。字段、行数和时间戳不变时无需改写 `info.json`、`episodes.jsonl` 或对齐记录。相机标定保留，腕部修正由依据包追溯。元数据不在当前备份／回滚范围内。

## 并发与响应

```bash
python3 -m vive_correct review --out outputs/stove_review --port 18775 \
  --preview-jobs 8 --frame-jobs 8 --approve-jobs 4 --export-jobs 1 --cache-mib 2048
```

以上是服务默认并发和缓存值。缓存上限不等于进程内存峰值；资源紧张时降低并发和缓存。审核与导出队列独立，可切换到其他条目继续操作；服务重启后中断任务不会自动重提。

## 产物与排查

- `run.log`、`feature_progress.json`：选帧日志与进度。
- `features/.../`：候选、人工记录、任务状态，以及当前 `approval.json` 指针。
- `rejected_episodes.json`：整条不通过的路径清单；无排除记录时文件可不存在。
- `approved/<版本>/`：审核依据及按需生成的修正 Parquet，结构见 [字段与接口](observations_schema.md#依据包)。
- `repair_manifest.json`、`rebuild_manifest.json`：导出／重建结果。
- `repairs/<运行>/rerun/`：录制、统计与 `render.log`。

常见问题：

- **列表一直等待**：核对选帧与审核的 `--out` 是否一致，查看选帧异常。
- **有标签却无候选**：参考 [漏检排查](cv_detection.md#漏检排查)，可换帧或手工补标。
- **依据过期／版本冲突**：重新加载条目，检查后再次审核；不要手改版本或哈希。
- **重投影或同步失败**：按提示核对帧、四角和姿态时间戳，不通过放宽阈值强行放行。
- **导出失败**：查看对应 `render.log`，修复后重试。排队不等于完成。
- **整条不通过**：查看排除清单中的原因，批处理会跳过该条；“拒绝此侧”不会写入此清单。
