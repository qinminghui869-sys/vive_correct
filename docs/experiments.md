# 模型对比实验

实验工具用于比较同一批候选在有无模型复核、不同模型下的结果。日常识别使用[项目首页](../README.md)的 `run` 命令。

## 固定候选池

先完成数据集扫描，再冻结候选池运行对照：

```bash
python3 -m vive_correct.ablation --dataset samples \
  --source-work outputs/scanned --out outputs/api_ablation

python3 -m vive_correct.ablation --dataset samples \
  --source-work outputs/scanned --out outputs/astra_ablation \
  --transport codex --codex-model gpt-6-astra
```

默认运行 `off` 和 `on` 两组。两组共用候选和像素几何计算，不继承人工基线、不补扫、不提前停止。`off` 按图像半区假设左右手；`on` 增加标签身份复核和独立 Tracker 定位。

`--mode off|on` 单独运行一组，`--mode summarize` 重建统计。API 模式严格读取 `--model-config` 指定的 JSON，模型名称须与服务端一致。候选文件哈希和输入签名用于检查实验是否可比。

## 输出与比较

- `inventory.json`：全部样本及缺失输入。
- `off/batch/`、`on/batch/`：各组完整报告和图片。
- `comparison.json`：覆盖、稳定性、重投影误差、模型请求和候选记录。
- `comparison.html`：图像对照页面。

合并已有实验：

```bash
python3 -m vive_correct.model_comparison \
  --old outputs/previous_ablation --qwen outputs/api_ablation \
  --gpt outputs/astra_ablation --out outputs/model_comparison
```

合并工具检查各组评审的候选 ID；候选不同则拒绝比较。`--old`、`--qwen`、`--gpt` 是保留的命令参数名，实际模型以各组记录为准。

Codex 与 API 的提示环境、解码控制不同；CLI 不复现 API 的 `temperature`、`max_tokens`、`enable_thinking` 和 `response_format`。这些差异随实验保存。

服务错误单独计数，不当作标签否决。覆盖、模型一致性和重投影误差只能用于诊断；定位精度需用人工标注或实测三维真值评估。异帧相机坐标不能直接相减。
