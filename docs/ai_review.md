# AI 预审：最终标签框与 XYZ

AI 预审默认关闭，只检查 OpenCV 最终选中的结果，不遍历候选池或补搜其他帧。调整结果是待审核建议，不能自动通过或排除数据。

## 启用

在 [CV 选帧命令](cv_detection.md) 后加 `--ai-final-review`。Codex 通道使用已登录的本机 CLI；模型默认值来自 [run.yaml](../vive_correct/config/run.yaml)，可用 `--codex-model` 覆盖。

```bash
bash scripts/run_cv.sh /path/to/dataset outputs/ai_review \
  --ai-final-review --transport codex --model-jobs 2

python3 -m vive_correct review --out outputs/ai_review
```

使用 API 通道时，将 `--transport codex` 替换为 `--transport api --model-config /path/to/model.local.json`。配置格式见 [示例](../vive_correct/openai.example.json)。

`--jobs` 控制 OpenCV 数据并发，`--model-jobs` 控制 AI 并发，`--model-timeout` 控制单次请求超时。单条选帧发布后即可人工审核；选帧命令会等待已排队的 AI 任务结束。`--dry-run` 只预览计划，不调用模型。

## 输入与处理边界

每次只发送一张拼接 JPEG，每手一行，从左至右为：**原图全景 → 未标注局部 → 当前框与轴 → PnP 基准轴**。同一行的局部裁剪范围一致，包含可见轴端点；双手可能来自不同帧。双手图为 1600×1026，单手图为 1600×556。实际 token 与耗时需由模型通道实测。

- **框有问题**：AI 返回原始完整帧内的粗定位区域（0–1000 归一化坐标）。OpenCV 局部重检测、精修四角并校验 PnP；失败则保留原结果交人工。区域不能直接作为最终四角。
- **轴有问题**：保持、按 PnP 校正、校正后局部旋转或无法判断。旋转仅允许绕当前 X/Y/Z 执行 ±90° 或 180°，首轮最多三个有序操作，按矩阵右乘组合，不累加欧拉角。
- **调用上限**：正常一次，调整后最多追加一次复核；复核轮不再改框，最多一次轴旋转。不循环修改，也不使用 `--model-retries` 自动重试。
- **几何不足**：缺少姿态、标定或存在平面歧义时不强行改轴；模型失败不阻塞人工审核。

框变化后先重算 PnP，再复核新框与轴。旋转不移动中心或修改四角。模型检查约定的标签几何方向，不宣称恢复设备真实固定轴。

## L3_test_data/new 批次

本批使用 [new_codex_review_mount.json](../configs/new_codex_review_mount.json)，轴定义为 `wrist_y_normal`。该定义已在 PnP 标签基准上执行局部 X −90°；AI 和人工“按四角校正”均已包含转换，**无需再补转一次**。数学约定见 [PnP 建议轴](../calibrations/README.md#pnp-建议轴)。

```bash
bash scripts/run_cv.sh /path/to/L3_test_data/new outputs/new_codex_review \
  --ai-final-review --transport codex --model-jobs 2 \
  --initial-mount configs/new_codex_review_mount.json

python3 -m vive_correct review --out outputs/new_codex_review \
  --initial-mount configs/new_codex_review_mount.json
```

后续 `repair`、`rebuild` 同样传入该配置。更换代码、输入或配置后使用新输出目录，并重启服务加载修改。

## 结果与人工接管

`features/.../ai_review.json` 独立保存原结果、建议、原因、操作和输入签名；字段见 [AI 建议记录](observations_schema.md#ai-建议记录)。相同输入和模型的完成结果可复用，失败后可重新运行。输入或计算代码变化使旧建议失效。

页面如何应用、恢复及处理失效建议，统一见 [人工审核：AI 建议](user_guide.md#ai-建议可选)。AI 不改写已有人工记录或源文件。
