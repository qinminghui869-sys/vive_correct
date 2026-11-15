"""Local, portable visual comparison of completed ablation arms."""

import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np


def render(output, comparison):
    output = Path(output)
    images = output / "comparison_images"
    images.mkdir(exist_ok=True)
    modes = [mode for mode in ("off", "on") if mode in comparison["summary"]]
    total = len(comparison["entries"])
    missing = sum(bool(row["missing_inputs"]) for row in comparison["entries"])
    titles = {
        "off": "无模型：图像半区身份假设",
        "on": str(comparison["summary"].get("on", {}).get("model") or "Qwen")
        + "：身份复核 + Tracker 定位",
    }
    cards, csv_rows = [], []
    for row in comparison["entries"]:
        panels = []
        for mode in modes:
            result = row.get(mode)
            if result is None:
                continue
            report = json.loads(Path(result["report"]).read_text())
            content = []
            for obs in report["observations"]:
                image = cv2.imread(obs["image"])
                if image is None:
                    raise ValueError(f"Missing source image: {obs['image']}")
                for side, pose in obs["stickers"].items():
                    corners = np.asarray(pose["image_corners_px"])
                    color = (230, 210, 20) if side == "left" else (200, 60, 230)
                    cv2.polylines(image, [np.round(corners).astype(np.int32)], True, color, 4)
                    x, y = np.round(corners.min(axis=0)).astype(int)
                    cv2.putText(
                        image,
                        side
                        + (
                            " ASSUMED"
                            if mode == "off"
                            else " " + str(comparison["summary"][mode].get("model") or "Qwen")
                        ),
                        (max(0, x), max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
                    csv_rows.append(
                        dict(
                            index=row["index"],
                            episode=row["episode"],
                            mode=mode,
                            side=side,
                            frame=obs["frame_index"],
                            timestamp_s=obs["timestamp_s"],
                            x_m=pose["origin_camera_m"][0],
                            y_m=pose["origin_camera_m"][1],
                            z_m=pose["origin_camera_m"][2],
                            rms_px=pose["rms_px"],
                            stability_passed=pose.get("stability", {}).get("passed"),
                            identity_source=pose.get("identity_review", {}).get("source"),
                        )
                    )
                relative = f"comparison_images/{mode}_{row['index']:02d}_{obs['frame_index']}.jpg"
                cv2.imwrite(str(output / relative), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                content.append(
                    f"<p>帧 {obs['frame_index']} · {html.escape(' / '.join(obs['stickers']))}</p>"
                    f'<a href="{relative}"><img loading="lazy" src="{relative}"></a>'
                )
            if not content:
                content.append("<p>没有通过本组筛选的观测；不表示视频中没有标签。</p>")
            if result["errors"]:
                content.append("<p>存在模型请求错误，结果不完整；详情见工作报告。</p>")
            if result.get("service_rejections"):
                count = sum(len(r.get("candidate_ids", [])) for r in result["service_rejections"])
                content.append(
                    f"<p><b>服务端图像内容检查拒绝了 {count} 个候选，未得到模型判断；不能计作标签身份否决。</b></p>"
                )
            report_link = str(Path(result["report"]).relative_to(output))
            if result.get("world_offsets_m"):
                content.append(
                    "<p>整段固定世界平移 / 米："
                    + html.escape(
                        json.dumps(
                            {
                                s: [round(v, 4) for v in xyz]
                                for s, xyz in result["world_offsets_m"].items()
                            },
                            ensure_ascii=False,
                        )
                    )
                    + "</p>"
                )
            if result.get("alignment_error"):
                content.append(
                    "<p>无法生成平移：" + html.escape(result["alignment_error"]) + "</p>"
                )
            panels.append(
                f"<section><h3>{titles[mode]}</h3>{''.join(content)}"
                f'<a href="{html.escape(report_link)}">完整观测 JSON</a></section>'
            )
        if row["missing_inputs"]:
            panels.append(
                "<p>两组均缺失输入：" + html.escape(", ".join(row["missing_inputs"])) + "</p>"
            )
        cards.append(
            f"<article><h2>{row['index']:02d} · {html.escape(Path(row['episode']).parent.name)} / "
            f'{html.escape(Path(row["episode"]).name)}</h2><div class="pair">{"".join(panels)}</div></article>'
        )
    metrics = [
        ("可处理样本", "processed"),
        ("输出双手观测", "bilateral"),
        ("仅单手观测", "single_hand"),
        ("无观测", "no_observation"),
        ("请求错误样本", "errors"),
        ("服务端内容检查拒绝的候选", "service_rejected_candidates"),
        ("判断未完整完成的样本", "incomplete_review_samples"),
        ("选中手侧观测", "selected_hands"),
        ("其中稳定性通过", "stable_hands"),
        ("评估候选", "evaluated_candidates"),
        ("满足双手位置修复输入", "alignment_ready"),
        ("重投影 RMS 中位数 / px", "rms_median_px"),
    ]
    table = "".join(
        "<tr><th>"
        + label
        + "</th>"
        + "".join(
            "<td>"
            + str(
                round(comparison["summary"][m][key], 4)
                if isinstance(comparison["summary"][m][key], float)
                else comparison["summary"][m][key]
            )
            + "</td>"
            for m in modes
        )
        + "</tr>"
        for label, key in metrics
    )
    subset = comparison.get("complete_review_subset")
    subset_html = ""
    if subset and subset["sample_count"]:
        values = subset["summary"]
        subset_html = f"<p><b>排除未完成判断的配对子集：{subset['sample_count']} 个样本。</b> "
        subset_html += (
            "；".join(
                f"{titles[m]}：双手 {values[m]['bilateral']}、单手 {values[m]['single_hand']}、无观测 {values[m]['no_observation']}"
                for m in modes
            )
            + "。</p>"
        )
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>标签复核模型消融</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#19232c;background:#fafafa}}
h1{{font-size:28px}}h2{{font-size:19px}}h3{{font-size:16px}}article{{border-top:1px solid #ccd2d8;padding:20px 0}}
.pair{{display:grid;grid-template-columns:repeat({len(modes)},minmax(0,1fr));gap:24px}}img{{width:100%}}
table{{border-collapse:collapse}}th,td{{padding:9px 24px 9px 0;text-align:left;border-bottom:1px solid #ddd}}
a{{color:#165e95}}p{{line-height:1.6}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>标签复核模型消融 · {total} 样本</h1>
<p>{total - missing} 个输入齐全的样本，{missing} 个缺少输入，详见清单。两组冻结同一历史 OpenCV 候选池；当前代码重新拟合几何。
每样本最多 24 个候选，11 帧稳定窗口，阈值 0.8；不继承人工基线，不自适应补扫或提前停止。</p>
<p><b>这些是输出覆盖率和几何诊断，不是识别准确率或三维精度。</b>无模型组按图像半区分配左右手，可能选中背景矩形。
不同帧的相机坐标不能直接比较。青色框为左手、粉色框为右手；关闭组标注 ASSUMED。</p>
<table><tr><th>指标</th>{"".join("<th>" + titles[m] + "</th>" for m in modes)}</tr>{table}</table>
<p>上表包含服务拒绝造成的实际输出缺口；这是服务开启后的整体结果，不能把所有减少都归因于标签筛选。完整完成模型判断的配对子集另见 comparison.json 的 complete_review_subset。</p>
{subset_html}
<p><a href="comparison.json">统计与逐样本 JSON</a> · <a href="selected_observations.csv">选中观测 CSV</a> · <a href="inventory.json">完整样本清单</a></p>
{"".join(cards)}</html>"""
    (output / "comparison.html").write_text(page)
    if csv_rows:
        with (output / "selected_observations.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
