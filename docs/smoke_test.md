# Smoke 测试

用于验证真实输入能完成选帧、可选 AI 预审和审核服务读取。RRD 验证须单独记录实际渲染模式与帧覆盖；功能通过不等于几何精度通过，不自动创建审核通过记录或覆盖源文件。

## 执行

在仓库根目录安装运行与开发依赖后执行：

```bash
bash scripts/check.sh
DATASET="/path/to/dataset"
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

逐条标定位于 `calibrations/chunk-000/episode_NNNNNN/rectify_params.yaml`；解析规则见 [输入要求](../vive_correct/README.md#inputs)。

## 历史验证：2026-11-24

代码版本 `0a7f5ef`，使用 `l3_beverage_shop_mix_beverage_left_1920x1184_30fps_11views_eeg128_emg8`（8 条）：

- `scripts/check.sh` 通过：276 项主测试、28 项数据工具测试、Ruff 和前端语法检查。
- 离线快速选帧完成 8/8，无输入或解码错误；7 条有双侧候选，`episode_000001` 缺右手候选。候选未经过人工验收。
- 本机审核页面、状态接口、8 条详情及首帧 PNG 读取通过。
- AI 联网预审使用 `--model-jobs 2 --model-timeout 60`，8/8 请求读取超时，未通过。选帧进程退出码为 0，不能据此判断 AI 成功。
- 未执行人工通过、RRD 导出或 Parquet 重建。

本机证据位于 `outputs/smoke_beverage_20261124_cv/` 与 `outputs/smoke_beverage_20261124_ai/`，包含 `run.log`、逐条产物和 `smoke_result.json`；CV 目录另含 `checks.log`、`http_smoke.json`。这些生成文件不提交 Git。

## RRD 兼容验证：2026-11-27

基于 `4e8991f` 的本次工作区修改，使用 `双手肌电 + MANUS/KS3/l3_beverage_shop_pour_beverage_left_1920x1184_30fps_11views_eeg128_emg8`：

- 8/8 条输入检查通过，逐条内参匹配 1920×1184；当时的导出实现只读取旧版独立元数据，8 条均选择 RGB 回退；此为历史行为，新格式兼容规则见 [RRD 输入](../ego_data/README_rerun.md#rrd-inputs)。
- 每条渲染 RGB 第 30–32 帧，共 24 帧；Rerun 数据接口读回确认包含 RGB、矫正双腕、手点和骨架。
- 使用明确标识的非生产校准夹具，对真实位姿施加 10 mm 本地平移及 15° 旋转；未创建人工审核通过记录，不代表实际校准精度。
- 源文件大小及修改时间在验证前后一致；输出仅写入本机 `outputs/rrd_compat_20261127/`，含 `smoke.py`、`result.json`、逐条 RRD 和日志，不随仓库提供。
- 固定 Ruff 0.14.3 下 `scripts/check.sh` 通过：281 项主测试、28 项数据工具测试、92 个文档链接、格式及前端语法检查。HTTP 回归需要允许绑定本机临时端口。

## 点云预览与 RGB 叠加验证：2026-11-27

同一饮料店数据的 8 条 episode 均识别为 `depth_unmasked_preview`，内参分辨率为 1920×1184。`episode_000004` 第 30–32 帧分别导出点云和强制 RGB 两份 RRD，并用 Rerun 数据接口读取验证：

- 点云每帧 142,080 点（默认步长采样），包含颜色与世界坐标手点。缺掩码的结果只作预览。
- RGB 录制包含 2D 关节与骨节实体；这三帧的左手可见关节数为 0、1、3，右手均在可显示范围外。原始手点和默认安装外参不构成准确贴合的保证。
- 回归覆盖新格式参数、旧格式优先级、无掩码模式、无深度时间线、投影裁剪、缺测清除及审核刚性修正仅应用一次。
- `scripts/check.sh`：286 项主测试、28 项数据工具测试，以及格式、静态、前端语法和文档链接检查通过。

本机证据在 `outputs/rrd_modes_20261127/`；验证未写入源数据或生产审核记录。这里记录的是短片段验证，完整 episode 需重新运行导出。

## 中心像素深度原点验证：2026-11-27

基于 `3cd8446` 后的工作区修改，使用同一 `pour_beverage` 数据的 8 条原审核选择和已核验的原始六文件基线，另存待审核预览：

- 16 个参考原点全部取得有效中心像素深度，范围 193–403 mm；RGB 与对应深度时间差均为 0。按实际解码帧序复核值一致。
- 8 条完整 RRD 共 16,235 帧，经读取与帧覆盖检查；世界原点按相机外参和头部位姿转换，参考腕部与目标原点在数值精度范围内一致。该检查不表示物理精度通过。
- 316 项测试通过，覆盖原点不依赖 PnP 深度、相机到世界转换、无效深度拒绝、投影、刚性修正和源文件保真。
- 本机产物位于 `outputs/pour_depth_pixel_review_20261127/`，包括 `rerun/manifest.json`、`origin_comparison.json`、`origin_verification.json` 和 `validation.json`。源文件大小及修改时间前后一致；未覆盖原审核结果或创建新的生产通过记录。

## 统一深度入口与直接写回验证：2026-11-27

基于 `3cd8446` 后的工作区实现，使用同一 `pour_beverage` 数据：

- 8 条数据的 16 个参考原点与上述完整 RRD 的校准结果一致。
- 在 `episode_000004` 的临时副本上连续写回两次，六文件哈希一致；验证更新数据集统计，生产源数据保持不变。
- `scripts/check.sh` 通过 297 项主测试、28 项数据工具测试。新增覆盖直接写回、原始基线复用、跨文件系统发布、失败回滚后重试、非目标字段保真及旧原点校准拒绝。

本机证据位于 `outputs/depth_origin_script_validation/`，包含 `copy_direct_write.log`、`copy_validation.json` 和 `check_direct_write.log`，不随仓库分发。该记录验证功能与数值一致性，不代表独立物理精度验收。
