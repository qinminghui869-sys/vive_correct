# 导出、重建与排错

启动和页面操作见 [快速开始](../README.md#快速开始) 与 [审核指南](user_guide.md)。命令在仓库根目录执行。默认过程目录是 `outputs/ks3`。

## 状态与处理范围

- **待审核**：草稿、单侧通过或依据过期。不能导出或重建。
- **已通过**：双手和依据都有效，可以生成修正文件或 RRD。
- **不通过**：整条排除。批处理记为跳过，跳过本身不是失败。

排除清单是 `<out>/rejected_episodes.json`，按样本路径去重，含视频、特征文件、审核人、原因和 UTC 时间。重启服务或重新发布候选不会解除排除。每个输出目录有自己的审核状态。

排除一条已导出的数据会使索引失效，历史文件仍保留。已经覆盖到源数据的结果不会自动回滚。排队或处理中的条目要等任务结束才能改决定。

## 导出与查看

```bash
python3 -m vive_correct repair --out outputs/ks3 --export-jobs 1
python3 -m vive_correct view outputs/ks3
```

`repair` 不代替审核。`view` 只打开已有录制。输入、审核、标定、代码和产物都没变时复用结果。深度不够时走 RGB；完整录制的输入见 [RRD 说明](../ego_data/README_rerun.md#完整-rrd-的输入)。

## 重建与备份

```bash
python3 -m vive_correct rebuild --out outputs/ks3
python3 -m vive_correct rebuild --out outputs/ks3 --in-place
```

第二条会在备份后覆盖源文件。单条加 `--episode /path/to/episode`。`review`、`repair`、`rebuild` 若用了自定义外参，必须传同一个 `--initial-mount FILE`。

每条数据只改该条数据目录下的六个文件。单条旧数据在 `data/chunk-000/episode_000000/`；多条数据集在 `data/chunk-000/episode_NNNNNN/`：

- `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet`
- `action.parquet`、`observation.state.parquet` 里的左右腕部字段
- `observation.hand_left_points.parquet`、`observation.hand_right_points.parquet`

保留行数、时间戳、字段类型、Arrow 元数据和非目标字段。缺测或同步失败的坐标组保持原值，并记录跳过数量。不修改 `observation.hand_skeleton.parquet`。

`--in-place` 先生成并校验全部六个文件，再备份、登记、替换。首次 `.parquet.bak` 不覆盖；再次重建仍以原始文件为基线。失败会回滚。中断后按 `<episode>/.vive_rebuild/` 的事务记录恢复。不要删备份或事务记录来绕过冲突检查。

### 数据集元数据

当前不更新 `meta/`。训练交付前，按原数据集口径重算 `episodes_stats.jsonl` 里已改维度的统计，否则可能影响归一化。`readme.json` 若记录大小或数量，也要刷新。字段、行数和时间戳不变时，不必改 `info.json`、`episodes.jsonl` 或对齐记录。相机标定保留，腕部修正由依据包追溯。`meta/` 不在当前备份和回滚范围内。

## 并发与响应

```bash
python3 -m vive_correct review --out outputs/ks3 --port 18775 \
  --preview-jobs 8 --frame-jobs 8 --approve-jobs 4 --export-jobs 1 --cache-mib 2048
```

以上是服务默认值。缓存上限不是进程内存峰值。资源紧时同时降低并发和缓存。审核队列和导出队列分开，可以切到其他条目继续操作。服务重启后，中断的任务不会自动重提。

## 产物与排查

- `run.log`、`feature_progress.json`：选帧日志和进度。
- `features/.../`：候选、人工记录、任务状态，以及当前 `approval.json` 指针。
- `rejected_episodes.json`：整条不通过。没有排除记录时可以不存在。
- `approved/<版本>/`：审核依据和按需生成的修正 Parquet，见 [依据包](observations_schema.md#依据包)。
- `repair_manifest.json`、`rebuild_manifest.json`：导出和重建结果。
- `repairs/<运行>/rerun/`：录制、统计和 `render.log`。

常见问题：

- **列表一直等待**：选帧和审核的 `--out` 是否相同，并看选帧异常。
- **有标签却无候选**：按 [漏检排查](cv_detection.md#漏检排查) 换帧或手工补标。
- **依据过期或版本冲突**：重新加载后再审。不要手改版本或哈希。
- **重投影或同步失败**：按提示核对帧、四角和姿态时间戳。不要放宽阈值强行通过。
- **导出失败**：看对应 `render.log`。排队不等于完成。
- **整条不通过**：看排除清单。「拒绝此侧」不会写入该清单。
