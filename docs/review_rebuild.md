# 导出、重建与排错

启动和页面操作见 [快速开始](../README.md#quick-start) 与 [审核指南](user_guide.md)。命令在仓库根目录执行。默认过程目录是 `outputs/ks3`。

## 状态与处理范围

- **待审核**：草稿、单侧通过或依据过期。不能导出或重建。
- **已通过**：双手和依据都有效，可以生成修正文件或 RRD。
- **不通过**：整条排除。批处理记为跳过，跳过本身不是失败。

排除清单是 `<out>/rejected_episodes.json`，按样本路径去重，含视频、特征文件、审核人、原因和 UTC 时间。重启服务或重新发布候选不会解除排除。每个输出目录有自己的审核状态。

排除一条已导出的数据会使索引失效，历史文件仍保留。已经覆盖到源数据的结果不会自动回滚。排队或处理中的条目要等任务结束才能改决定。

<a name="export-and-view"></a>

## 导出与查看

```bash
python3 -m vive_correct repair --out outputs/ks3 --export-jobs 1
python3 -m vive_correct view outputs/ks3
```

`repair` 只处理已通过且依据有效的数据，生成 RRD；`view` 打开已有录制。输入、审核、标定、代码和产物都没变时复用结果。点云、无掩码预览及 RGB 手点叠加的输入和时间线规则见 [RRD 说明](../ego_data/README_rerun.md#rrd-inputs)。

<a name="rebuild-and-backup"></a>

## Parquet 重建与源文件备份

```bash
python3 -m vive_correct rebuild --out outputs/ks3
```

默认另存修正 Parquet。确认需要覆盖源文件时，显式执行：

```bash
python3 -m vive_correct rebuild --out outputs/ks3 --in-place
```

单条加 `--episode /path/to/episode`。`review`、`repair`、`rebuild` 若用了自定义外参，必须传同一个 `--initial-mount FILE`。

两个源数据写回入口共用以下字段范围。单条旧数据位于 `data/chunk-000/episode_000000/`；多条数据集使用对应 `episode_NNNNNN/`。

| 文件 | 修改字段 |
| --- | --- |
| `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet` | 世界 XYZ、审核旋转对应的姿态角 |
| `observation.hand_left_points.parquet`、`observation.hand_right_points.parquet` | 21 个关节点的世界 XYZ |
| `action.parquet`、`observation.state.parquet` | 左右腕部 XYZ 和姿态角 |

保留头部位姿、手指屈伸值、时间戳、编号、行序、字段类型及元数据。不修改 `observation.hand_skeleton.parquet`。缺测组处理见 [字段保真规则](hand_pipeline.md#synchronization)。

`--in-place` 校验通过后直接替换这六个源文件，并在同一事务里更新 [数据集元数据](#dataset-metadata)。源目录不写 `.parquet.bak` 或 `.vive_rebuild/`。

原始文件留在首次执行的 `<out>/rebuild_state/<样本>/baseline/`，并由 `outputs/rebuild_index.json` 登记。再次重建仍用这份基线。失败或中断会恢复六个源文件和已改的元数据。不要删除 `rebuild_state/` 或索引来绕过校验。

<a name="depth-origin-writeback"></a>

## 用深度原点重新矫正并写回

此脚本直接更新源数据的六个 Parquet，不提供预览、RRD 导出或另存修正文件模式。输入为源数据集及保存了双手通过审核记录和原始基线的历史过程目录：

```bash
python3 scripts/apply_depth_origin.py \
  --dataset "/path/to/dataset" \
  --process-out "outputs/existing_review"
```

**执行即覆盖源文件。** 默认处理全部 episode；单条增加 `--episode episode_000004`，可重复指定。路径含中文、空格或 `+` 时加引号。

脚本读取 `features/**/approval.json`、对应 `approved/` 快照、`rebuild_manifest.json` 和 `rebuild_state/` 原始基线，兼容历史机器路径迁移。它复用审核选择，按[深度反投影规则](hand_pipeline.md)重新计算，不创建新的审核通过记录。缺少原始基线、有效深度或审核材料时停止。

写回字段、基线和统计更新与 [原地重建](#rebuild-and-backup) 一致。

所有选定条目计算通过后才写回，临时文件自动清理。仅在 `<process-out>/depth_origin_runs/` 保留必要执行记录，原始备份与恢复日志继续使用原有 `rebuild_state/`，常规索引会登记该位置。重复执行从原始基线计算，不叠加矫正。

每条内部失败回滚并停止后续写入；跨 episode 不保证原子性，已完成条目保留在执行记录中。发现源数据被外部修改时拒绝覆盖。进程中断的事务可通过现有 `source_data.recover(episode)` 恢复；不要删除基线或索引绕过校验。无掩码深度仍标记 `unmasked_depth_preview`，不代表物理精度已经验证。

<a name="dataset-metadata"></a>

## 数据集元数据

`rebuild --in-place` 和深度原点写回脚本均在六文件事务内更新数据集 `meta/`。失败或中断会把这两份文件和六个 Parquet 一起恢复。

- `episodes_stats.jsonl`：重算该条的 `observation.left_wrist_pose`、`observation.right_wrist_pose`、`action`、`observation.state`、`observation.hand_left_points`、`observation.hand_right_points`。每项重算 `min`、`max`、`mean`、`std` 和 `count`。计算方式沿用这一项原来的数值，未改动的列保持原统计。其他特征不变。
- `readme.json`：`size_bytes` 加上这六个文件的大小变化。`file_count`、`frame_count` 和其他字段不变。

文件不存在时跳过。不改 `info.json`、`episodes.jsonl`、`tasks.jsonl`、`calibrations.jsonl`、`collect_info.jsonl`、`qc_reports.jsonl`。`hand_pose`、`hand_skeleton` 以及头、眼、肌电、脑电的统计不变。

<a name="concurrency"></a>

## 并发与响应

```bash
python3 -m vive_correct review --out outputs/ks3 --port 18775 \
  --preview-jobs 8 --frame-jobs 8 --approve-jobs 4 --export-jobs 1 --cache-mib 2048
```

以上是服务默认值。缓存上限不是进程内存峰值。资源紧时同时降低并发和缓存。审核队列和导出队列分开，可以切到其他条目继续操作。服务重启后，中断的任务不会自动重提。

<a name="outputs-and-troubleshooting"></a>

## 产物与排查

- `run.log`、`feature_progress.json`：选帧日志和进度。
- `features/.../`：候选、人工记录、任务状态，以及当前 `approval.json` 指针。
- `rejected_episodes.json`：整条不通过。没有排除记录时可以不存在。
- `approved/<版本>/`：审核依据和按需生成的修正 Parquet，见 [依据包](observations_schema.md#approval-bundle)。
- `repair_manifest.json`、`rebuild_manifest.json`：导出和重建结果。
- `depth_origin_runs/*.json`：深度原点直接写回记录，包括逐条状态、来源和文件哈希。
- `repairs/<运行>/rerun/`：录制、统计和 `render.log`。

常见问题：

- **列表一直等待**：选帧和审核的 `--out` 是否相同，并看选帧异常。
- **有标签却无候选**：按 [漏检排查](cv_detection.md#detection-troubleshooting) 换帧或手工补标。
- **依据过期或版本冲突**：重新加载后再审。不要手改版本或哈希。
- **原点深度不可用**：核对参考帧中心像素、毫米深度、内参及 RGB/深度时间差。当前原点不再回退到 PnP；可选择有有效深度的参考帧重新审核。
- **重投影或同步失败**：按提示核对帧、四角和姿态时间戳。不要放宽阈值强行通过。
- **导出失败**：先看 `repair_manifest.json` 的错误，再看对应 `render.log` 中的时间线、帧数及异常。仅成功完成的临时录制才发布为 `recording.rrd`；排队不等于完成。
- **窗口出现但文件加载失败**：查看 `viewer.log`。Linux 的 `inotify` 实例耗尽会导致 `failed to create retryable file reader`；`view` 会自动改用仅监听 `127.0.0.1` 的 HTTP 读取，关闭窗口后自动停止服务，无需重新导出。
- **RRD 没有点云**：按 [深度输入](../ego_data/README_rerun.md#rrd-inputs) 检查录制 JSON 的 `mode` 和 `frame_timeline`。缺掩码可生成预览点云；缺深度或参数才回退 RGB。手点未叠加时查看 `hand_projection` 的缺测、内参或外参原因。
- **整条不通过**：看排除清单。「拒绝此侧」不会写入该清单。
