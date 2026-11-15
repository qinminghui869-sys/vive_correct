"""Export a portable result bundle without scan caches or debug artifacts."""

import json
import re
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from vive_correct.batch.report_io import digest


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _select(data, keys):
    return {key: data[key] for key in keys if key in data}


def compact_report(report, calibration):
    """Schema v6: pose, synchronization and essential uncertainty; no review history."""
    result = _select(
        report,
        ("episode", "video_key", "origin", "axis_length_m", "mode", "identity", "reviewer_type"),
    )
    result.update(
        schema_version=6,
        camera=_select(report.get("camera", {}), ("K", "resolution", "distortion")),
        camera_coordinates="OpenCV: X right, Y down, Z forward; metres",
        pose_transform="T_camera_local maps local coordinates to camera coordinates",
        mount_calibration=_select(calibration, ("accepted", "reason")),
        observations=[],
    )
    for observation in report["observations"]:
        item = _select(observation, ("frame_index", "timestamp_s"))
        item["stickers"] = {}
        for side, pose in observation["stickers"].items():
            transform = pose.get("T_camera_local")
            axes_generated = pose.get("axes_generated", transform is not None)
            origin = pose.get("origin_camera_m")
            if origin is None and transform is not None:
                origin = [row[3] for row in transform[:3]]
            value = dict(
                origin_camera_m=origin,
                axes_generated=axes_generated,
                axes_failure_reason=pose.get("axes_failure_reason"),
                T_camera_local=transform if axes_generated else None,
                local_axes_convention=pose.get(
                    "local_axes_convention", report.get("local_axes_convention", "unknown")
                ),
                quality_status=pose.get(
                    "quality_status", observation.get("quality_status", "unknown")
                ),
                orientation_status=pose.get("orientation_status", "unknown"),
                rms_px=pose.get("rms_px"),
                stability_passed=pose.get("stability", {}).get("passed"),
                temporal_consistency_passed=pose.get("temporal_consistency", {}).get("passed"),
                baseline_retained=bool(pose.get("baseline_retention")),
                identity_source=pose.get("identity_review", {}).get("source", "unknown"),
            )
            # Unknown is null, never silently promoted to verified or unambiguous.
            for field in (
                "plane_ambiguous",
                "orientation_180_unresolved",
                "physical_axes_verified",
                "true_vive_origin_calibrated",
            ):
                value[field] = pose.get(field)
            item["stickers"][side] = value
        result["observations"].append(item)
    return result


def export_delivery(root):
    """Export global axis views, compact reports, a page and one index."""
    root = Path(root).resolve()
    gallery = json.loads((root / "gallery_manifest.json").read_text())
    with TemporaryDirectory(prefix=".delivery-", dir=root) as temporary:
        bundle = Path(temporary)

        def copy(source, target):
            target = bundle / target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return str(target.relative_to(bundle))

        entries = []
        for entry in gallery["entries"]:
            report_path = root / entry["report"]
            if digest(report_path) != entry["report_sha256"]:
                raise ValueError(f"Report changed: {report_path}")
            report = json.loads(report_path.read_text())
            calibration = json.loads((root / entry["calibration"]).read_text())
            if calibration.get("source_report_sha256") not in (None, entry["report_sha256"]):
                raise ValueError("Calibration does not match source report")
            _write(bundle / entry["report"], compact_report(report, calibration))
            copy(root / entry["image"], entry["image"])
            record = _select(
                entry,
                (
                    "index",
                    "episode",
                    "hands",
                    "missing_hands",
                    "mode",
                    "image",
                    "report",
                    "status",
                    "hand_mapping",
                ),
            )
            record["report_sha256"] = digest(bundle / entry["report"])
            record["image_sha256"] = digest(bundle / entry["image"])
            entries.append(record)
        index = _select(
            gallery,
            (
                "total",
                "bilateral_observation_count",
                "single_hand_count",
                "validated_calibration_count",
                "no_observation_count",
                "origin",
                "axes",
                "note",
            ),
        )
        index.update(schema_version=6, path_base="delivery root", entries=entries)
        _write(bundle / "gallery_manifest.json", index)
        page = (root / "index.html").read_text()
        page = re.sub(r'<a href="bilateral_results.zip">.*?</a> · ', "", page)
        for entry in gallery["entries"]:
            page = page.replace(f' · <a href="{entry["calibration"]}">同步与外参检查</a>', "")
        # Old batch pages may link to optional diagnostics excluded from delivery.
        page = re.sub(
            r'<a href="([^"]+)">(.*?)</a>',
            lambda m: m.group(0) if (bundle / m.group(1)).is_file() else m.group(2),
            page,
            flags=re.DOTALL,
        )
        (bundle / "index.html").write_text(page)
        archive_path = root / ".bilateral_results.zip"
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(bundle)))
        with ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC validation failed")
        destination = root / "delivery"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(bundle), destination)
        archive_path.replace(root / "bilateral_results.zip")
    return destination
