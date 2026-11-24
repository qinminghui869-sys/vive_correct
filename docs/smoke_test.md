# Smoke 测试

用于验证真实输入能完成选帧、AI 预审和审核服务读取。它不代替人工几何验收，也不自动通过审核或覆盖源文件。

## 执行

在仓库根目录安装运行与开发依赖后执行：

```bash
bash scripts/check.sh
DATASET="/mnt/nas/synnas/脑电/双手肌电 + MANUS/KS3/l3_beverage_shop_mix_beverage_left_1920x1184_30fps_11views_eeg128_emg8"
OUT="outputs/smoke_beverage"
bash scripts/run_cv.sh "$DATASET" "$OUT" --jobs 2 \
  --ai-final-review --model-jobs 1 --model-timeout 30
python3 -m vive_correct review --out "$OUT" --port 18775
```

AI 需要有效的本地模型配置和网络；配置见 [AI 预审](ai_review.md)。只测离线路径时去掉 `--ai-final-review`。输入、代码或参数变化后使用新的输出目录。

## 验收

- 检查日志退出码，以及每条 `features.json` 的错误、候选和停止原因；退出码 0 不等于检出正确。
- 检查每条 `ai_review.json` 的完成状态与错误；模型失败时，CV 和人工审核仍可用，但 AI 链路不能记为通过。
- 打开审核页，确认条目可读取、特征图可显示。人工检查左右手、四角与 XYZ 后，才能验证通过审核、RRD 导出和 Parquet 重建。
- 留存命令、代码版本、日志和输出目录。仅有候选而没有人工真值时，不报告准确率。

逐条标定位于 `calibrations/chunk-000/episode_NNNNNN/rectify_params.yaml`；解析规则见 [输入要求](../vive_correct/README.md#输入要求)。
