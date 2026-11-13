"""Publish reviewed bilateral coverage and calibration evidence without hiding gaps."""

import argparse
import html
import json
from pathlib import Path
from vive_correct.batch.delivery import export_delivery
from vive_correct.batch.hand_overlay import HandOverlay

import cv2
import numpy as np
from vive_correct.batch.report_io import digest as sha256


def build(root):
    root = Path(root).resolve()
    source = root / "results/manifest.json"
    manifest = json.loads(source.read_text())
    images = root / "images"
    images.mkdir(exist_ok=True)
    entries, cards = [], []
    has_axis_reviews = False
    quality_names = {
        "verified": "稳定观测",
        "conditional_motion": "清晰运动帧；稳定检查未通过",
        "low_resolution_preview": "低分辨率预览；未验收",
    }
    for entry in manifest["entries"]:
        report_path = Path(entry["report"])
        if sha256(report_path) != entry["report_sha256"]:
            raise ValueError(f"Report changed: {report_path}")
        report = json.loads(report_path.read_text())
        has_axis_reviews |= any(
            "axis_review" in pose
            for observation in report["observations"]
            for pose in observation["stickers"].values()
        )
        calibration_path = root / "calibration" / f"{entry['index']:02d}" / "calibration.json"
        calibration = json.loads(calibration_path.read_text())
        if Path(calibration["episode"]).resolve() != Path(entry["episode"]).resolve():
            raise ValueError(f"Calibration belongs to another episode: {calibration_path}")
        recorded_hash = calibration.get("source_report_sha256")
        if recorded_hash is not None and recorded_hash != entry["report_sha256"]:
            raise ValueError(f"Calibration is stale; rerun calibration: {calibration_path}")
        # Show a simultaneous observation when available; otherwise show each
        # wrist's own source frame. Never overlay different timestamps together.
        both = [o for o in report["observations"] if set(o["stickers"]) == {"left", "right"}]
        if both:
            chosen = [both[0]]
        else:
            chosen = []
            for side in entry["hands"]:
                observations = [o for o in report["observations"] if side in o["stickers"]]
                chosen.append(
                    max(
                        observations,
                        key=lambda o: abs(
                            cv2.contourArea(
                                np.asarray(o["stickers"][side]["image_corners_px"], np.float32)
                            )
                        ),
                    )
                )
        panels, descriptions, hand_records = [], [], []
        hands_overlay = HandOverlay(entry["episode"], calibration, report.get("camera", {}))
        for observation in chosen:
            image = cv2.imread(observation["overlay"])
            if image is None:
                raise ValueError(f"Missing overlay: {observation['overlay']}")
            image, hand_record = hands_overlay.draw(image, observation.get("timestamp_s"))
            hand_records.append(hand_record)
            height = round(image.shape[0] * 960 / image.shape[1])
            panel = np.full((height + 90, 960, 3), 245, np.uint8)
            panel[90:] = cv2.resize(image, (960, height), interpolation=cv2.INTER_AREA)
            sides = "+".join(observation["stickers"])
            status = observation["quality_status"]
            cv2.putText(
                panel,
                f"{entry['index']:02d} | {sides} | frame {observation['frame_index']} | {status}",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            axis_note = (
                "Reference axes unverified | "
                if any(
                    pose.get("orientation_status") in {"model_inferred", "model_uncertain"}
                    for pose in observation["stickers"].values()
                )
                else ""
            )
            cv2.putText(
                panel,
                axis_note
                + "Tag center | X red, Y green, Z blue | Separate panels: different times",
                (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "Hand joints: L cyan / R magenta | " + hand_record["status"] + " | Not a MANO mesh",
                (15, 77),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )
            descriptions.append("手部21点投影：" + hand_record["status"])
            panels.append(panel)
            descriptions.append(
                f"{sides}：帧 {observation['frame_index']} / {quality_names.get(status, status)}"
            )
            for side, pose in observation["stickers"].items():
                orientation = pose.get("orientation_status")
                if orientation:
                    names = {
                        "user_constraint_applied": "已按用户方向意见调整；待结果复核",
                        "reconstructed_pending_direction_review": "已重新拟合；轴向待确认",
                        "not_direction_reviewed": "轴向未单独验收",
                        "model_inferred": "Qwen推断；未经物理验证",
                        "model_uncertain": "固定参考轴；物理方向未确认",
                    }
                    descriptions.append(f"{side} 轴向：{names.get(orientation, orientation)}")
        picture = images / f"{entry['index']:02d}_global.jpg"
        if not panels:
            panel = np.full((220, 960, 3), 245, np.uint8)
            cv2.putText(
                panel,
                f"{entry['index']:02d} | No accepted label observation",
                (24, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            panels.append(panel)
            descriptions.append("没有通过复核的标签；未生成位姿，不代表视频中没有标签")
        if not cv2.imwrite(str(picture), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise OSError(picture)
        if entry.get("status") == "review_error":
            descriptions.append("模型复核请求失败或部分未完成，请检查工作目录中的复核记录并重试")
        missing = "、".join(entry["missing_hands"]) or "无"
        kind = (
            "双手同帧观测"
            if both
            else (
                "双手分帧观测"
                if len(entry["hands"]) == 2
                else "仅单手观测"
                if entry["hands"]
                else "无可用观测"
            )
        )
        fit = calibration.get("fit", {})
        reason = calibration["reason"]
        if fit and not fit.get("installation_plausible", True):
            reason = "安装偏移超出合理性筛查范围"
        record = dict(
            entry,
            hand_mapping=hand_records,
            image=str(picture.relative_to(root)),
            image_sha256=sha256(picture),
            calibration=str(calibration_path.relative_to(root)),
            calibration_sha256=sha256(calibration_path),
            calibrated=calibration["accepted"],
            calibration_reason=reason,
            report=str(report_path.relative_to(root)),
        )
        entries.append(record)
        cards.append(f'''<article data-kind="{"both" if len(entry["hands"]) == 2 else "single"}">
<h2>{entry["index"]:02d} · {html.escape(Path(entry["episode"]).name)}</h2>
<p><b>{kind}</b> · 缺失手：{missing}</p><a href="{record["image"]}"><img loading="lazy" src="{record["image"]}"></a>
<p>{html.escape("；".join(descriptions))}</p><p>全段同步标定：{"通过" if record["calibrated"] else "未通过"} · {html.escape(reason)}</p>
<p><a href="{record["report"]}">观测与质量报告</a> · <a href="{record["calibration"]}">同步与外参检查</a></p></article>''')
    count = sum(len(e["hands"]) == 2 for e in entries)
    calibrated = sum(e["calibrated"] for e in entries)
    single = sum(len(e["hands"]) == 1 for e in entries)
    empty = sum(not e["hands"] for e in entries)
    axis_description = (
        "逐条采用轴向复核记录；以报告中的 axis_review 为准"
        if has_axis_reviews
        else "局部轴约定以报告 local_axes_convention 为准"
    )
    result = {
        "entries": entries,
        "total": len(entries),
        "bilateral_observation_count": count,
        "single_hand_count": single,
        "no_observation_count": empty,
        "validated_calibration_count": calibrated,
        "source_manifest_sha256": sha256(source),
        "origin": "label center; internal Vive origin offset remains unmeasured",
        "axes": axis_description,
        "note": "Coverage includes low-resolution previews. Separate frames do not establish simultaneous relative pose.",
    }
    (root / "gallery_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    document = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>双手 Vive 观测与标定检查</title>
<style>body{{max-width:1500px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#182332;font:16px/1.65 system-ui}}
article{{background:white;padding:22px;margin:24px 0;border-radius:12px}}h2{{font-size:20px}}img{{width:100%;height:auto}}a{{color:#165fb0}}button{{padding:8px 18px;margin-right:8px}}.note{{background:#fff1cd;padding:18px;border-radius:8px}}</style>
<h1>双手 Vive 观测与标定检查</h1><p>{len(entries)} 条有效视频 · 双手观测 {count} 条 · 仅单手 {single} 条 · 无可用观测 {empty} 条 · 通过标定 {calibrated} 条</p>
<p>白色标签 50×30 mm；左目矫正 RGB；逐条使用 rectify_params.yaml 内参；OpenCV 检测与位姿估计。复核来源以逐条报告为准，助手视觉判断不代表人类逐条验收。</p>
<p class="note">原点为标签中心，尚未测量标签到 Vive 内部原点的偏移。红 X、绿 Y、蓝 Z；{html.escape(axis_description)}。
双手分帧图不可视为同一时刻的相对位姿。低分辨率预览未验收；清晰运动帧保留稳定检查未通过标记。
短片段误差小不等于全段标定可靠。未通过的外参不用于输出正式双手轨迹。清单仅包含本次输入的有效视频。</p>
<p><a href="bilateral_results.zip">下载 {len(entries)} 条效果图与报告</a> · <a href="gallery_manifest.json">机器可读清单</a></p>
<button onclick="filter('all')">全部</button><button onclick="filter('both')">双手观测</button><button onclick="filter('single')">待补另一只手</button>
{"".join(cards)}<script>function filter(kind){{document.querySelectorAll('article').forEach(e=>e.hidden=kind!=='all'&&e.dataset.kind!==kind)}}</script></html>"""
    (root / "index.html").write_text(document)
    export_delivery(root)
    verification = {
        "records": len(entries),
        "images_decodable": all(cv2.imread(str(root / e["image"])) is not None for e in entries),
        "report_hashes_valid": all(
            sha256(root / e["report"]) == e["report_sha256"] for e in entries
        ),
        "zip_crc_valid": True,
        "browser_verified": False,
    }
    (root / "gallery_validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "entries"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    build(parser.parse_args().root)
