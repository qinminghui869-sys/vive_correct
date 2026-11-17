# 人工审核、RRD 导出与源数据重建

前提：已按 [纯 CV 指南](cv_detection.md) 生成 `<out>/features/`。以下命令始终使用同一个输出目录。

## 1. 通过审核

```bash
python3 -m vive_correct review --out outputs/cv
```

打开 `http://127.0.0.1:18763`，核对左右手参考图，点击 **通过审核**。只有需要修改时，才展开重选帧、四角或物理手侧；**仅暂存**可以保存单侧进度，不代表审核通过。

通过审核会保存原帧、四角叠图、像素坐标、PnP 与深度校验结果、同步腕部姿态、世界偏移、审核人和标定快照。依据包校验成功后显示“已通过”；此时不会生成 RRD，也不会修改源文件。旧记录缺少依据包时，重新点击“通过审核”补齐。

审核和导出按数据独立运行：提交后立即返回，最多同时处理两条，其余排队。同一条处理中不能重复提交，其他数据仍可编辑、暂存或通过审核。

服务仅监听本机；`--port` 可指定端口，占用时明确报错，不自动切换。

## 2. 按需导出 RRD

页面点击 **导出 RRD 文件**，完成后查看：

```bash
python3 -m vive_correct view outputs/cv
```

也可用 `python3 -m vive_correct repair --out outputs/cv` 批量导出。未审核、依据过期或页面存在未保存修改时，不能直接导出。首次 RRD 需要逐帧渲染；“已提交”不代表“导出完成”。

RRD 与重建共用修正 Parquet，显示原始与修正后的腕部和已有 21 个关节；不重新生成手型、不重复加偏移。无深度时使用 RGB 路径。`repair` 和页面导出均不覆盖源文件；历史腕部审核仍保留非覆盖兼容路径。

## 3. 重建六个文件

```bash
# 生成独立修正文件和校验报告
python3 -m vive_correct rebuild --out outputs/cv

# 备份后覆盖源文件
python3 -m vive_correct rebuild --out outputs/cv --in-place

# 只处理指定数据
python3 -m vive_correct rebuild --out outputs/cv --in-place \
  --episode /绝对路径/数据目录
```

仅有完整有效审核依据的数据能够重建；单条失败不阻断其他数据，结果记录在 `rebuild_manifest.json`。`review`、`repair`、`rebuild` 均支持 `--initial-mount FILE`，应始终使用同一份相机外参。

六个文件位于 `<episode>/data/chunk-000/episode_000000/`：

- `observation.{left,right}_wrist_pose.parquet`：有效 `x/y/z`。
- `action.parquet`、`observation.state.parquet`：左右 `*_wrist_x/y/z`。
- `observation.hand_{left,right}_points.parquet`：21 个关节的有效三维位置。

每侧使用自己的参考帧和真实时间戳计算：

```text
Δ = 标签中心世界坐标 − 同时刻原始腕部位置
修正位置(t) = 原始位置(t) + Δ
```

仅做固定世界平移，保留原旋转。各文件按自身行数处理，保留行序、时间戳、缺失值、非目标字段和 Arrow schema 元数据；某组 xyz 不完整时整组保持原值。标签中心不等于已标定的 Vive 内部原点。

## 备份与恢复

覆盖前先生成、校验全部六个文件，再保存初始 `.parquet.bak` 并逐文件替换。六文件校验成功后才发布完成状态；写入失败回滚到本次执行前版本，进程中断后下次操作先恢复。

登记后的初始备份永不覆盖，重复重建始终从备份计算。项目自身的覆盖不会使审核过期，RRD 仍能读取备份显示原始对照。出现未登记的同名备份、备份损坏或外部修改时停止该条，报告冲突；不要通过删除备份或登记信息绕过检查。

## 产物与排查

```text
<out>/features/.../human_review.json   # 可暂存的人工选择
<out>/features/.../approval.json       # 当前有效依据包指针
<out>/approved/<版本>/                 # 审核快照、观测、偏移、图片、修正文件
<out>/repairs/<运行>/rerun/             # recording.rrd、统计与 render.log
<out>/repair_manifest.json             # RRD 导出结果
<out>/rebuild_manifest.json            # 六文件重建结果
<episode>/.vive_rebuild/               # 初始备份登记和覆盖事务
```

- **未找到特征帧**：检查 `--out` 是否指向检测输出；新终端中的 `$vive_out` 可能为空，可直接使用绝对路径。
- **待审核或依据过期**：重新加载该条并点击“通过审核”。输入、人工选择、标定或计算代码变化都会使旧依据失效。
- **四角、同步或求解失败**：人工修改仍保留；按具体原因调整参考帧或角点后重新提交。
- **导出失败**：查看对应运行的 `render.log`；不要把排队或处理中视为完成。

未变化的完整产物会复用缓存。字段定义见 [报告接口](observations_schema.md)，开发与验证见 [开发说明](development.md)。
