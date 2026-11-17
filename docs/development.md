# 开发说明

## 模块职责

- `run_dataset.py`、`dataset.py`：运行入口、样本发现、计划与恢复。
- `estimate/`：标签检测、几何求解、稳定性与绘图；`batch/`：候选选择、模型复核与交付。
- `human_review.py`：人工选择、输入绑定、几何复核；`review_bundle.py`：版本化审核依据和修正文件缓存。
- `review_server.py`、`review_ui/`：本机 HTTP 接口、独立后台任务与交互。
- `review_actions.py`：已审核数据的 RRD 导出和六文件重建编排；`review_repair.py`：`repair` 命令及历史腕部审核兼容。
- `parquet_correction.py`：三维位置平移与字段保真校验；`source_data.py`：原始备份读取、源文件冲突检查与覆盖事务。
- `review_io.py`：JSON 原子写入、输出锁、签名和批次索引；不依赖上层审核或渲染模块。
- `rerun_export.py`、`ego_data/`：录制导出、时间同步、坐标对齐和渲染。

新审核路径按“人工选择 → 依据包 → 共用修正文件 → RRD 或覆盖”单向调用。公共函数不能放在旧兼容入口中；同一份位置变换只实现一次。

## 数据与并发约束

- 算法参数放 `config/*.yaml`，CLI 显式参数覆盖预设。画面半区只作搜索提示，不能替代物理侧别确认。
- 标签中心不是已标定的 Vive 内部原点；当前重建只平移位置，保留旋转和已有手点。
- 普通审核、导出只写输出目录。仅 `rebuild --in-place` 可修改六个源 Parquet；原始备份永不覆盖。
- 长操作持有单条输出锁；根输出锁只用于更新索引。跨输出目录读同一源数据时使用共享锁，覆盖和恢复使用排他锁。
- 同时需要输出锁和源锁时，先取单条输出锁；源文件事务不得反向取得输出锁。
- 六文件先生成并校验，再备份、登记事务、替换、校验和提交。只有全部成功才能发布完成；失败或中断恢复到本次执行前版本。
- JSON 不允许 NaN；事务登记需同步文件和目录。输入、审核、标定或计算代码变化后，旧依据必须重新确认。

坐标和状态字段见 [报告接口](observations_schema.md)，操作与恢复见 [审核与重建指南](review_rebuild.md)。

## 检查

需要 Python 3.10+、Node.js（仅做前端语法检查）和开发依赖：

```bash
python3 -m pip install -r requirements-dev.txt
bash scripts/check.sh
```

检查入口统一执行 Ruff、前端语法及两组单元测试；GitHub CI 使用同一脚本，不依赖本地数据或模型密钥。HTTP 测试需要允许绑定本机临时端口。

修改 Python 后用 `ruff format` 格式化对应文件。前端采用 Prettier 3.6.2：

```bash
npx --yes prettier@3.6.2 --write vive_correct/review_ui
```

维护测试重点：不同参考帧的同步、有效位置偏移与字段保真、审核和导出分离、并行审核、缓存失效、备份冲突、失败回滚、中断恢复，以及 RRD 不重复加偏移。新增测试应覆盖这些行为，避免只复述实现。修改流程时同步更新唯一对应指南，不在各 README 重复完整操作步骤。
