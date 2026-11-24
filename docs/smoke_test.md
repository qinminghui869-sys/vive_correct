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

## 最近验证：2026-11-24

代码版本 `0a7f5ef`，使用上面的饮料店数据（8 条）：

- `scripts/check.sh` 通过：276 项主测试、28 项数据工具测试、Ruff 和前端语法检查。
- 离线快速选帧完成 8/8，无输入或解码错误；7 条有双侧候选，`episode_000001` 缺右手候选。候选未经过人工验收。
- 本机审核页面、状态接口、8 条详情及首帧 PNG 读取通过。
- AI 联网预审使用 `--model-jobs 2 --model-timeout 60`，8/8 请求读取超时，未通过。选帧进程退出码为 0，不能据此判断 AI 成功。
- 未执行人工通过、RRD 导出或 Parquet 重建。

本机证据位于 `outputs/smoke_beverage_20261124_cv/` 与 `outputs/smoke_beverage_20261124_ai/`，包含 `run.log`、逐条产物和 `smoke_result.json`；CV 目录另含 `checks.log`、`http_smoke.json`。这些生成文件不提交 Git。
