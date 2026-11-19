# 眼动模式验证记录

本文记录 `optimize/eye-camera-world` 的验证范围。使用入口见 [快速开始](../README.md#快速开始) 和 [人工审核指南](user_guide.md)；共同外参公式及误差门槛见 [校准模型](hand_pipeline.md#眼动固定参考系)。

## 仅平移原点版本

当前输出采用 `eye_origin_standing_axes`：相机原点、Standing 轴向，源 Vive 旋转字段不改写。相机外参仍用于参考帧投影；两手和已有 point 使用同一平移。旧审核依据需重新生成。

`scripts/check.sh` 通过 **295 个测试**（主套件 268、ego_data 27）；最终手点投影调整另通过 18 个眼动集成测试，Ruff、前端语法及空白检查通过。覆盖输出旋转及 null 保真、非规范欧拉角不重编码、相机／输出坐标变换一致、手点仅平移、旧旋转模型读取兼容和 RRD 输出。

真实样本 `2026-09-03-15-28-18` 在独立目录 `outputs/eye_origin_only_smoke/` 重新选帧：采样 57 帧，保留帧 540 的单标签候选。浏览器已验证参考帧预览及保留原轴向的界面说明，截图为 `origin_only_review.png`。未通过人工审核，不覆盖源数据。

## 当前限制

- 仅平移原点保留源 Vive 旋转及既有手点形状，不能修复手点与真实人手不贴合的问题。当前分支不代表物理精度验收通过。
- 图像三轴依赖四角和人工安装方向估计的相机外参。默认方向已来自四角，因此零角度时“按四角校正方向”通常没有可见变化；不能据此独立验证设备物理轴向。
- 标签中心近似 Vive 原点，标签到 Vive 的完整安装变换尚未测量；MANUS 的 `[y,z,x]` 换轴与 `[0,-0.085,-0.058] m` 偏移仍为既有假设。
- 源欧拉角按 `xyz` 外旋、角度制解析，尚未对照采集端转换核实。后续应先核验旋转编码，再分别标定标签到 Vive、MANUS 到 Vive 的安装关系，用独立帧验收。
- 本分支未实现上述安装标定方案；不估计后续相机运动，不提供移动相机下的逐帧手点贴合。

## 上一版真实数据检查（选帧规则未变）

数据目录：`/home/baai/qmh/dataset/L3/L3_test_data/old/早期脑电`。全部 10 条已检查，结果写入独立目录 `outputs/eye_shared_standing/`，未覆盖源数据。

- `2026-08-06-10-53-23`、`2026-08-20-09-26-23` 缺少完整手点和骨架，整条跳过。
- 其余 8 条完成固定采样，逐条核对实际帧号符合规定；其中 `2026-08-27-09-16-32` 采满 100 帧，其余在片尾结束。
- 6 条产生单标签候选：8/25 帧 495；8/27 帧 770、775；8/31 13:52 帧 590、580、585；8/31 19:09 帧 580、585、590；9/1 帧 615；9/3 帧 540。
- 9/7、9/8 没有通过高质量门槛的标签。未检出双标签候选；单标签不保证另一手或其标签可见，仍可能包含外观误检。

8 条均保持待处理，未提交真实数据批准或导出。9/3 帧 540 用于浏览器功能验证：两侧帧号一致，右侧提供临时轴向预览，左侧仍需补标。安装方向、标签原点偏差以及采集端欧拉角转换尚未独立验收。

运行目录包含 `checks.log`、`scan.log`、`validation.json` 和 `shared_standing_review.png`。候选仅覆盖规定采样窗口，无候选不代表整段视频没有可用帧。

## 当前版本全流程

在已安装项目依赖的 Python 环境中执行。此机器可使用 `conda activate egophi`；新建环境见仓库首页。

```bash
cd /home/baai/qmh/vive_repair
conda activate egophi

# 1. 选帧：10 秒起、每隔 5 帧、最多 100 帧
bash scripts/run_cv.sh \
  '/home/baai/qmh/dataset/L3/L3_test_data/old/早期脑电' \
  outputs/eye_origin_only --coordinate-mode eye-reference --jobs 4

# 2. 人工审核（本终端持续运行）
python3 -m vive_correct review --out outputs/eye_origin_only --port 18774
```

打开 <http://127.0.0.1:18774>，同帧依次确认双手四角、物理手侧及标签安装方向，填写审核人，点击“通过审核”。单标签需补齐另一侧；无有效共同帧时保留待处理。

审核完成后，在另一终端激活同一环境并执行：

```bash
cd /home/baai/qmh/vive_repair
conda activate egophi

# 3. 独立重建 Parquet，不覆盖源数据
python3 -m vive_correct rebuild --out outputs/eye_origin_only

# 4. 导出已审核样本的 RRD
python3 -m vive_correct repair --out outputs/eye_origin_only --export-jobs 1

# 5. 打开 Rerun
python3 -m vive_correct view outputs/eye_origin_only
```

`rebuild` 和 `repair` 只处理有效的已审核样本。新输出目录避免复用上一版旋转校准结果；历史实测候选不代表已完成当前版本的人工精度验收。
