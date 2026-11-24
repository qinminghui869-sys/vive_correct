"""Combine completed frozen-pool experiments without treating coverage as accuracy."""

import argparse
import html
import json
from pathlib import Path

from vive_correct.batch.report_io import write_json


def combine(arms, output):
    results, rows, votes = {}, {}, {}
    for name, root, mode in arms:
        data = json.loads((root / "comparison.json").read_text())
        stats = dict(data["summary"][mode])
        decisions = {}
        for row in data["entries"]:
            target = rows.setdefault(
                row["index"],
                dict(
                    index=row["index"],
                    episode=row["episode"],
                    missing_inputs=row["missing_inputs"],
                    arms={},
                ),
            )
            if (
                target["episode"] != row["episode"]
                or target["missing_inputs"] != row["missing_inputs"]
            ):
                raise ValueError(f"Sample mismatch at index {row['index']}")
            if mode not in row:
                continue
            target["arms"][name] = row[mode]
            if mode == "on":
                detail = json.loads(
                    (
                        root / mode / "qwen_reviews" / f"{row['index']:02d}" / "review_summary.json"
                    ).read_text()
                )
                for decision in detail["decisions"]:
                    if "is_label" in decision:
                        key = (row["index"], decision["candidate_id"])
                        if key in decisions:
                            raise ValueError("Duplicate candidate decision")
                        decisions[key] = decision
        if mode == "on":
            stats["returned_decisions"] = len(decisions)
            stats["label_yes"] = sum(d["is_label"] for d in decisions.values())
            stats["clear_label_yes"] = sum(d["is_label"] and d["clear"] for d in decisions.values())
            stats["identity_pass"] = sum(
                d["is_label"]
                and d["clear"]
                and d["physical_side"] in ("left", "right")
                and d["confidence"] >= 0.8
                for d in decisions.values()
            )
            votes[name] = decisions
        results[name] = dict(stats=stats, root=str(root), mode=mode)
    comparable = []
    for row in rows.values():
        values = list(row["arms"].values())
        ids = [v.get("evaluated_candidate_ids") for v in values]
        row["same_candidate_ids"] = (
            len(values) == len(arms)
            and bool(ids)
            and all(i is not None and i == ids[0] for i in ids)
        )
        if not row["missing_inputs"] and not row["same_candidate_ids"]:
            raise ValueError(f"Candidate mismatch in sample {row['index']}")
        if row["same_candidate_ids"] and all(v["review_complete"] for v in values):
            comparable.append(row["index"])
    agreements = []
    names = list(votes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            keys = votes[a].keys() & votes[b].keys()
            agreements.append(
                dict(
                    a=a,
                    b=b,
                    paired_returned_candidates=len(keys),
                    label_agreement=sum(
                        votes[a][k]["is_label"] == votes[b][k]["is_label"] for k in keys
                    ),
                    disagreement_candidates=[
                        dict(index=k[0], candidate_id=k[1], a=votes[a][k], b=votes[b][k])
                        for k in sorted(keys)
                        if votes[a][k]["is_label"] != votes[b][k]["is_label"]
                    ],
                    label_disagreement=sum(
                        votes[a][k]["is_label"] != votes[b][k]["is_label"] for k in keys
                    ),
                )
            )
    result = dict(
        arms=results,
        entries=list(rows.values()),
        decision_agreement=agreements,
        complete_sample_indices=comparable,
        complete_subset={
            name: {
                "bilateral": sum(len(rows[i]["arms"][name]["hands"]) == 2 for i in comparable),
                "single_hand": sum(len(rows[i]["arms"][name]["hands"]) == 1 for i in comparable),
                "no_observation": sum(not rows[i]["arms"][name]["hands"] for i in comparable),
            }
            for name in results
        },
        caveats=[
            "No independent human ground truth: coverage, agreement and reprojection RMS are not detection accuracy.",
            "Model and transport are recorded per arm; API and Codex system instructions and decoding controls may differ.",
            "Every usable sample uses the same evaluated candidate IDs; no adaptive scan or inherited human baseline.",
            "Final observations also depend on independent tracker localization, geometry and selection. Candidate identity_pass is before these downstream stages.",
        ],
    )
    write_json(output / "comparison.json", result)
    metrics = [
        ("样本数", "processed"),
        ("双手观测", "bilateral"),
        ("单手观测", "single_hand"),
        ("无观测", "no_observation"),
        ("选中手侧", "selected_hands"),
        ("稳定手侧", "stable_hands"),
        ("实际评估候选", "evaluated_candidates"),
        ("返回有效判断", "returned_decisions"),
        ("判断为标签", "label_yes"),
        ("标签且清晰", "clear_label_yes"),
        ("标签/清晰/手侧/置信度通过", "identity_pass"),
        ("服务拒绝候选", "service_rejected_candidates"),
        ("判断未完整样本", "incomplete_review_samples"),
        ("可生成双手修复参数", "alignment_ready"),
        ("RMS 中位数 / px", "rms_median_px"),
    ]
    esc = html.escape
    table = "<tr><th>指标</th>" + "".join(f"<th>{esc(n)}</th>" for n in results) + "</tr>"
    for title, key in metrics:
        table += (
            "<tr><th>"
            + title
            + "</th>"
            + "".join(
                "<td>"
                + esc(
                    str(
                        round(r["stats"][key], 4)
                        if isinstance(r["stats"].get(key), float)
                        else r["stats"].get(key, "不适用")
                    )
                )
                + "</td>"
                for r in results.values()
            )
            + "</tr>"
        )
    links = "".join(
        f'<li><a href="{Path(r["root"]).as_uri()}/comparison.html">{esc(n)}：逐样本图片与坐标</a></li>'
        for n, r in results.items()
    )
    records = ""
    for row in rows.values():
        cells = ""
        for name in results:
            value = row["arms"].get(name)
            text = (
                "输入缺失"
                if value is None
                else " / ".join(f"{s}: f{h['frame']}" for s, h in value["hands"].items())
                or "无观测"
            )
            cells += "<td>" + esc(text) + "</td>"
        records += (
            f"<tr><td>{row['index']:02d} · {esc(Path(row['episode']).parent.name)}</td>{cells}</tr>"
        )
    visual_cards = []
    reviewed_names = [name for name, arm in results.items() if arm["mode"] == "on"]
    for row in rows.values():
        panels = []
        for name in reviewed_names:
            value = row["arms"].get(name)
            frames = sorted({h["frame"] for h in value["hands"].values()}) if value else []
            content = ""
            for frame in frames:
                picture = (
                    Path(results[name]["root"])
                    / "comparison_images"
                    / f"on_{row['index']:02d}_{frame}.jpg"
                )
                if not picture.exists():
                    raise ValueError(f"Missing annotated comparison image: {picture}")
                content += f'<p>帧 {frame}</p><a href="{picture.as_uri()}"><img loading="lazy" src="{picture.as_uri()}" style="width:100%"></a>'
            if not content:
                content = "<p>输入缺失</p>" if value is None else "<p>本组没有保留观测。</p>"
            panels.append(f"<section><h3>{esc(name)}</h3>{content}</section>")
        if panels:
            visual_cards.append(
                f'<article><h3>{row["index"]:02d} · {esc(Path(row["episode"]).name)}</h3><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px">{"".join(panels)}</div></article>'
            )
    agreement_text = "".join(
        f"<p>{esc(a['a'])} / {esc(a['b'])}：共同返回判断 {a['paired_returned_candidates']} 个；标签判断一致 {a['label_agreement']} 个，不一致 {a['label_disagreement']} 个。</p>"
        for a in agreements
    )
    missing = sum(bool(row["missing_inputs"]) for row in rows.values())
    page = f"""<!doctype html><meta charset="utf-8"><title>标签检测模型对比</title><style>body{{font:16px system-ui;max-width:1300px;margin:36px auto;padding:0 20px;color:#19232c}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:10px}}p{{line-height:1.7}}a{{color:#165e95}}</style>
<h1>标签检测模型对比 · {len(rows)} 个样本</h1><p>{len(rows) - missing} 个输入完整，{missing} 个输入缺失。各组评估候选数见下表。</p>
<p>比较覆盖、候选判断和模型一致性。没有独立真值，不能据此计算准确率。模型与调用通道见各组记录。</p>
<table>{table}</table><h2>候选级标签判断一致性</h2>{agreement_text}<p>只比较两模型均返回有效判断的候选；服务拒绝或请求失败不记为标签否决。一致不代表正确。</p>
<p>所有组均完整完成判断的配对子集：{len(comparable)} 个样本。详细记录见 <a href="comparison.json">JSON</a>。</p><h2>逐样本图片报告</h2><ul>{links}</ul><h2>各组最终选择帧</h2><table><tr><th>样本</th>{"".join("<th>" + esc(n) + "</th>" for n in results)}</tr>{records}</table>"""
    page += (
        "<h2>模型复核图片对照</h2><p>各列可能选择不同帧；不可直接比较跨帧像素或相机系坐标差。</p>"
        + "".join(visual_cards)
    )
    (output / "comparison.html").write_text(page)
    return result


def model_label(root, label):
    stats = json.loads((root / "comparison.json").read_text())["summary"]["on"]
    return f"{label} · {stats.get('model') or '未记录模型'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine(
        [
            ("无模型", args.old.resolve(), "off"),
            (model_label(args.old, "基线复核"), args.old.resolve(), "on"),
            (model_label(args.qwen, "对照组"), args.qwen.resolve(), "on"),
            (model_label(args.gpt, "实验组"), args.gpt.resolve(), "on"),
        ],
        args.out.resolve(),
    )


if __name__ == "__main__":
    main()
