# AI 预审：最终标签框与 XYZ

AI 预审默认关闭。它只检查 OpenCV 最终选中的结果，不遍历候选池，也不补搜其他帧。输出是待审核建议，不能自动通过或排除。

## 启用

在 [CV 选帧](cv_detection.md) 命令后加 `--ai-final-review`。默认 `transport=api`。模型、地址和密钥读 `vive_correct/openai.local.json`，该文件优先于环境变量。格式见 [示例](../vive_correct/openai.example.json)，当前示例模型为 `qwen3.8-max`。`--model`、`--api-key` 或 `--model-config` 可覆盖它。

Codex 通道显式加 `--transport codex`。模型名来自 [run.yaml](../vive_correct/config/run.yaml)，可用 `--codex-model` 覆盖。

```bash
bash scripts/run_cv.sh /mnt/nas/synnas/脑电/KS3 outputs/ks3 \
  --ai-final-review --model-jobs 2

python3 -m vive_correct review --out outputs/ks3
```

`--jobs` 是 OpenCV 并发，`--model-jobs` 是 AI 并发，`--model-timeout` 是单次超时。一条选帧发布后就可以审核；选帧进程会等已排队的 AI 结束。`--dry-run` 不调用模型。

## 输入与处理边界

每次只发一张拼接 JPEG。每手一行，从左到右是：**原图全景、未标注局部、当前框与轴、PnP 基准轴**。同一行的局部裁剪范围一致。双手可以来自不同帧。双手图为 1600×1026，单手图为 1600×556。

- **框**：先看全景。标签必须在手腕或手套上的黑色 Tracker，其他白矩形都不算。Tracker 仍在当前帧时，给原图 0–1000 的粗区域；当前帧没有时，改到元数据里的一个候选帧。区域和候选帧只选一个。程序重检四角，两者都不是最终四角。
- **轴**：框不是 ok 时不改轴。框通过后，源轴和 PnP 方向不同仍保持。只有轴穿入黑色壳体、左右镜像或缺轴才采用 PnP。旋转只允许绕当前 X／Y／Z 做 ±90° 或 180°，首轮最多三个，按矩阵右乘，不累加欧拉角。
- **调用**：正常一次，调整后最多再复核一次。复核不再改框，最多一次旋转。不循环，也不用 `--model-retries` 重试这次预审。
- **几何不足**：缺姿态、缺标定或平面有歧义时不强行改轴。模型失败不挡住人工审核。

框变化后先重算 PnP，再复核新框和轴。旋转不移动中心，也不改四角。

## L3_test_data/new 批次

本批使用 [new_codex_review_mount.json](../configs/new_codex_review_mount.json)，轴定义为 `wrist_y_normal`。该定义已包含局部 X −90°。AI 和人工「按四角校正」都用这一次转换，不要再补转。约定见 [PnP 建议轴](../calibrations/README.md#pnp-建议轴)。

```bash
bash scripts/run_cv.sh <dataset> outputs/new_review \
  --ai-final-review --model-jobs 2 \
  --initial-mount configs/new_codex_review_mount.json

python3 -m vive_correct review --out outputs/new_review \
  --initial-mount configs/new_codex_review_mount.json --port 18775
```

后续 `repair` 和 `rebuild` 传入同一配置。代码、输入或配置变化后换新输出目录，并重启审核服务。

## 结果与人工接管

`features/.../ai_review.json` 保存原结果、建议、原因和输入签名。字段见 [AI 建议记录](observations_schema.md#ai-建议记录)。相同输入和模型的完成结果可复用。输入或计算代码变化后，旧建议失效。

页面上如何应用、恢复和处理失效建议，见 [人工审核](user_guide.md#ai-建议可选)。AI 不改写人工记录或源文件。
