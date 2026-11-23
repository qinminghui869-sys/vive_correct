# 模型对比实验

比较同一批候选在有无模型复核、不同模型下的结果。日常流程见 [项目首页](../README.md)，模型配置见 [运行指南](../vive_correct/README.md#独立模型候选流程)。

## 固定候选池

先扫描数据集，再冻结候选对照：

```bash
python3 -m vive_correct.ablation --dataset /path/to/dataset \
  --source-work outputs/scanned --out outputs/api_ablation

python3 -m vive_correct.ablation --dataset /path/to/dataset \
  --source-work outputs/scanned --out outputs/astra_ablation \
  --transport codex --codex-model gpt-6-astra
```

默认跑 `off` / `on` 两组：共用候选与像素几何，不继承人工基线、不补扫、不提前停止。`off` 按图像半区假设左右；`on` 增加身份复核与 Tracker 定位。

`--mode off|on|summarize` 可单组或重建统计。API 模式严格读 `--model-config` JSON。

## 输出与合并

- `inventory.json`：样本与缺失输入
- `off/batch/`、`on/batch/`：各组报告与图片
- `comparison.json` / `comparison.html`：覆盖、误差、请求与图像对照

```bash
python3 -m vive_correct.model_comparison \
  --old outputs/previous_ablation --qwen outputs/api_ablation \
  --gpt outputs/astra_ablation --out outputs/model_comparison
```

候选 ID 不一致则拒绝比较。`--old` / `--qwen` / `--gpt` 为保留参数名，实际模型以各组记录为准。Codex 与 API 的提示与解码控制不同，差异随实验保存。
