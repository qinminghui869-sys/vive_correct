# 导出、重建与排错

本页适用于两种坐标模式。界面操作见 [人工审核指南](user_guide.md)，首次运行见 [快速开始](../README.md#快速开始)。以下以 `outputs/headband` 为例；眼动模式替换为自己的输出目录。后续命令自动继承已保存的模式。

## 启动与续用

选帧和审核使用同一 `--out`，可边选边审或先选后审：

```bash
python3 -m vive_correct review --out outputs/headband
```

服务仅监听本机，默认端口 18763；占用时加 `--port 18774`。新结果自动加入列表，不切换当前样本或覆盖草稿；选帧结束不影响审核。

左目模式使用自定义 `--initial-mount FILE` 时，审核、导出和重建应保持同一份外参。眼动模式继承选帧时的眼动标定，不使用该参数。更换模式、输入、标定或计算配置应新建输出目录。

## 导出 RRD

通过审核后，点击页面的 **导出 RRD 文件**，或批量导出有效的已审核数据：

```bash
python3 -m vive_correct repair --out outputs/headband --export-jobs 1
python3 -m vive_correct view outputs/headband
```

未审核、依据过期或存在未保存编辑时不能直接导出。排队、处理中和失败都不代表完成；审核成功也不会自动导出。

RRD 直接读取与重建共用的修正 Parquet，避免重复校准。左目模式支持原世界中的修正前后对照及可用深度；眼动模式展示固定参考世界中的 Vive、手骨架、轨迹、同步眼动视频和参考帧叠图。展示范围及输入要求见 [Rerun 说明](../ego_data/README_rerun.md)。

## 重建 Parquet

重建不依赖 RRD，只处理审核依据有效的数据，单条失败不阻断其余样本：

```bash
# 独立输出，保留源数据
python3 -m vive_correct rebuild --out outputs/headband

# 仅处理指定样本
python3 -m vive_correct rebuild --out outputs/headband --episode /path/to/episode

# 显式覆盖源数据，自动备份并登记事务
python3 -m vive_correct rebuild --out outputs/headband --in-place
```

目标为每条数据 `data/chunk-000/episode_000000/` 下的六个文件：

- `observation.left_wrist_pose.parquet`、`observation.right_wrist_pose.parquet`：左右 Vive 位姿。
- `action.parquet`、`observation.state.parquet`：其中的左右腕部字段。
- `observation.hand_left_points.parquet`、`observation.hand_right_points.parquet`：左右 21 关节手点。

两种模式均保留行数、行序、时间戳、字段类型和非目标字段。左目模式的无效坐标组保留原值；眼动模式输出 NaN 并保留原 null 掩码，避免新旧世界坐标混合。眼动 `action/state` 只变换已验证映射的腕部字段，其他字段仍在原坐标系。公式、同步和保真规则见 [校准模型](hand_pipeline.md)。

保留 `features/`、当前 `approved/<版本>/`、原始数据和标定；覆盖源文件后还应保留初始 `.parquet.bak` 及 `<episode>/.vive_rebuild/`。仅有变换 JSON 不足以重建。

## 备份与恢复

覆盖前先生成并校验六个修正文件，再备份、登记事务和替换；全部校验成功才发布完成状态。失败回滚到本次执行前版本，中断在下次操作时恢复。

初始备份永不覆盖，重复重建始终从原始基线计算，避免累积修正。项目登记的覆盖不使审核依据过期。遇到未登记备份、备份损坏或外部修改时停止该条并报告冲突；不要删除备份或登记信息绕过检查。

## 数据集元数据

重建不修改 `observation.hand_skeleton.parquet`，也不自动更新数据集 `meta/`。眼动模式更新修正文件中的 Arrow 坐标系元数据，与数据集 `meta/` 相互独立。

用于训练或数据集交付时，需按原统计口径重算 `episodes_stats.jsonl` 中修改维度的统计量。字段结构、行数和时间戳不变时，`info.json`、`episodes.jsonl` 与时间对齐记录无需修改；保留原相机标定，并由审核依据包记录腕部修正。若 `readme.json` 记录文件大小或数量，应同步刷新。

旧统计不影响 RRD 显示，但可能影响训练归一化。当前备份和事务不包含这些 JSON；扩展元数据写入须纳入事务。

## 并发与响应

默认预览 8、图像编码 8、审核 4、RRD 导出 1，共享缓存上限 2048 MiB：

```bash
python3 -m vive_correct review --out outputs/headband   --preview-jobs 8 --frame-jobs 8 --approve-jobs 4 --export-jobs 1 --cache-mib 2048
```

首次解码较靠后的帧可能较慢，后续操作复用图像、几何及姿态缓存。资源紧张时降低并发及 `--cache-mib`；缓存上限不等于进程总内存上限。参数修改后重启服务。

审核与导出队列独立。同条数据处理中禁止修改或重复提交，其他数据仍可审核。服务重启后，未完成任务标记为中断，需人工重试；提高并发不改变选帧预算或输出质量。

## 产物与排查

```text
<out>/
├── feature_plan.json / feature_progress.json  # 选帧配置、输入与进度
├── review_catalog.json                       # 稳定样本编号
├── features/<样本路径>/
│   ├── features.json / human_review.json     # 候选与人工记录
│   ├── review_job.json                       # 任务状态
│   └── approval.json                         # 当前依据包指针
├── approved/<版本>/                          # 图像、变换、标定、修正文件与哈希
├── repairs/<运行>/rerun/                      # 录制、统计与 render.log
├── rerun/manifest.json                       # 录制索引
├── repair_manifest.json                      # 导出结果
└── rebuild_manifest.json                     # 重建结果
```

- **一直等待**：检查选帧和审核的 `--out` 是否一致，以及选帧进度和异常列表。
- **没有候选或误检**：按 [漏检排查](cv_detection.md#漏检排查) 检查采样与过滤原因，必要时换帧或手工补标。无有效共同帧的眼动样本保持待处理。
- **换帧后标注消失**：眼动模式会清除另一帧的标注，双手需在新的共同帧重新检查，详见 [调整规则](user_guide.md#2-调整四角与方向)。
- **几何或同步失败**：核查标签四角、清晰度及位姿时间戳；不要提高阈值强行放行。
- **设备映射或安装不一致**：检查 `deviceN_*`、源腕部、重命名记录及 point／骨架关系；不能按图像左右自动交换源文件。
- **依据过期或版本冲突**：重新加载并核查；输入或配置已变化时使用新输出目录。
- **导出失败**：查看 `render.log`，修复原因后重试；完整且未变化的产物会复用。

依据文件和 API 结构见 [字段与接口](observations_schema.md)，事务与缓存约束见 [开发说明](development.md)。
