# 运行指南

基本安装和命令见[项目首页](../README.md)。统一入口为 `python3 -m vive_correct run`；旧入口 `python3 vive_correct/run_dataset.py` 仍可使用。

## 输入

标签识别需要每条样本的矫正左目视频、时间戳和 `calibrations/rectify_params.yaml`。视频分辨率必须与标定一致。

安装检查使用头部和双腕姿态。骨架叠图使用左右手 21 点 parquet，缺失时仍可生成标签图片。Rerun 导出还需要深度、有效深度掩码、深度内参、眼动视频和 gaze 数据。

## 纯 CV 全流程

```bash
bash scripts/run_cv.sh samples outputs/cv_full --jobs 4
```

等效入口是 `python3 -m vive_correct run --profile cv --stage all --model-review off --dataset samples --out outputs/cv_full --jobs 4`。脚本另将 stdout/stderr 保存为 `run.log` 并保留退出码。

只接受校正左目视频，缺失或解码失败报错。顺序运行连续跟踪/几何观测与特征选帧，产物包括帧序号、四角、标签中心、原帧、特征图和人工审核模板。`batch/delivery/index.html` 为特征图入口，`tracked.html` 为独立跟踪报告。模型密钥和 Codex 登录均不需要；不会自动进行物理身份验证、轨迹修复或 Rerun 导出。

特征首轮扫描前10秒每5帧、后段每15帧；仍缺任意一侧时，后段每5帧补扫，再按需全段逐帧补扫。纯 CV 身份初始化需要连续5帧可靠双侧关联；之后直接按当前帧几何选取观测，不再检查稳定窗口。跟踪过程不跳帧。

完整输入结构、40条清单命令、错误处理和审核步骤见[纯 CV 指南](../docs/cv_detection.md)。后面的模型识别流程与模型交付目录只适用于 Astra/legacy 路径。

## 模型与参数

默认 `--profile astra` 使用 Codex 登录账户调用 `gpt-6-astra`，不读取 API 密钥。预设在 [config/run.yaml](config/run.yaml)，显式命令参数优先：

- 粗扫步长 15 帧，自动补扫步长 3 帧，Tracker 局部搜索 6 帧。
- 首轮评审 4 个候选，每批 4 个，每条样本最多 24 个几何合格候选。
- 身份置信阈值 0.8，超时 300 秒，重试参数 2。
- CPU 并发 10，模型并发 4，可用 `--jobs` 和 `--model-jobs` 调整。

API 模式使用 `--profile legacy`。复制 `openai.example.json` 为 `openai.local.json`，填写 `api_key`、`base_url` 和 `model`；也可使用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。配置优先级为命令参数、环境变量、`.env`、JSON。当前 API 入口要求 Qwen 模型。

`legacy` 使用旧搜索参数：补扫步长 5、首轮 8、每批 1，不做 Tracker 局部搜索。若只想切换模型通道而保留 Astra 搜索参数，使用 `--transport api`。

复核会发送全帧和局部图。Codex 请求的图片、提示和返回内容保存在 `codex_requests/`；该目录属于本地运行产物。

## 识别流程

1. 检查输入和模型连接，从原始视频生成候选。
2. 用像素边缘拟合四角，通过 PnP 计算标签中心，检查 11 帧窗口的稳定性。
3. 模型判断标签身份、可见性和物理左右手。
4. 未获得稳定双手时补扫一次：加密扫描，并在模型定位的 Tracker 区域内搜索标签。
5. 根据质量选择候选，再用无候选轮廓的原图独立复核 Tracker 归属。
6. 输出有效观测、缺失手侧、停止原因和错误记录。

模型框只限定搜索区域，不作为标签角点或中心。允许 Tracker 外壳部分出画，但标签四角必须完整。候选预算不包含 Tracker 定位和独立复核请求；每次独立复核最多检查 8 轮。

不稳定候选可能保留为 `conditional_motion`。双手可以来自不同帧，不能据此计算同时刻相对位姿。身份复核仍是模型判断，结果需按用途人工检查。

## 运行控制

- `--episodes-file FILE`：JSON 数组中的相对样本路径，保持顺序；拒绝重复或未知路径。
- `--dry-run`：仅用于默认 `all` 阶段。预览参数、样本和输入签名，不联网、不写结果；不检查视频解码或服务可用性。API 配置中的模型在正式执行时解析。
- `--profile cv` 或 `--model-review off`：纯 CV 连续跟踪，每手选择最佳观测；首次双手窗口假设未交叉，结果仍未人工核验。参数见[纯 CV 指南](../docs/cv_detection.md)。兼容旧名称 `--qwen-review`；显式固定候选池消融仍保留旧半区流程。
- `--fixed-candidates`：关闭补扫和提前停止，用于固定候选池比较。
- `--tracker-guided-frames N`：局部搜索帧数，0 表示关闭。模型关闭、固定候选或基线回放模式不支持此项。
- `--axis-baseline FILE`：加载与样本及报告哈希绑定的观测，保留已有结果并搜索缺失手。
- `--baseline-only`：要求基线已有双手，跳过模型调用。
- `--initial-mount FILE`：指定头相机外参；默认值说明见[相机参数](../calibrations/README.md)。

完整参数见 `python3 -m vive_correct run --help`。几何和时序阈值在 [config/estimate.yaml](config/estimate.yaml)，独立扫描与安装检查参数在 [config/batch.yaml](config/batch.yaml)。

## 阶段与产物

模型路径默认 `--stage all` 执行估计、扫描、复核、安装检查和交付；纯 CV 的 `all` 范围见上文。`estimate`、`scan` 用于分阶段诊断；`finish` 接受已有人工复核文件，具体参数见帮助。

```text
<out>/
├── run_plan.json                 # 参数、样本、输入签名和代码哈希
├── estimates/                    # 初始估计
├── search/、search_refined/      # 粗扫与补扫候选
├── qwen_reviews/                 # 模型评审；实际模型以报告字段为准
├── codex_requests/               # Codex 请求记录
├── batch/
│   ├── results/                  # 完整工作报告
│   ├── run_metadata.json         # 运行参数与计划哈希
│   ├── delivery/index.html       # 图片与精简报告入口
│   └── bilateral_results.zip     # 交付包
└── rerun/                        # 录制、转换报告和失败日志
```

恢复运行会检查模型、参数、样本、代码哈希及输入大小和修改时间。旧复核目录没有 `run_plan.json` 时，请换新输出目录；已有结果仍可查看或重新导出：

```bash
python3 -m vive_correct run --out outputs/demo --stage gallery
python3 -m vive_correct run --out outputs/demo --stage rerun --open-rerun
```

重新导出需要完整源数据与工作报告；自定义安装外参也须保持一致。缺少任意一侧有效原点时不导出录制。接口或解码错误保留诊断并返回非零退出码，不当作标签否决。

每条工作报告的 `model_review.stop_reason` 记录结束原因，`missing_hands` 记录缺失手侧。交付字段与坐标约定见[报告接口](../docs/observations_schema.md)，实验工具见[模型对比实验](../docs/experiments.md)。
