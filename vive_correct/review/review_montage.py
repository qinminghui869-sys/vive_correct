"""One bounded image for final review; crops retain visible axes and original coordinates."""

import cv2
import numpy as np

WIDTH = 1600
HEADER = 86
ROW_HEIGHT = 470
LAYOUT_VERSION = 1


def crop_bounds(shape, quad, projections):
    """Include the tag, context and every visible current/PnP axis segment."""
    h, w = shape[:2]
    q = np.asarray(quad, float)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("Montage requires four finite corners")
    points = list(q)
    for projection in projections:
        if not projection or projection.get("origin") is None:
            continue
        origin = projection["origin"]
        for endpoint in projection.get("endpoints", []):
            if endpoint is None:
                continue
            segment = np.asarray([origin, endpoint], float)
            if segment.shape != (2, 2) or not np.isfinite(segment).all():
                continue
            a, b = [tuple(np.clip(np.round(p), -1000000, 1000000).astype(int)) for p in segment]
            visible, a, b = cv2.clipLine((0, 0, w, h), a, b)
            if visible:
                points.extend((a, b))
    points = np.asarray(points)
    margin = np.maximum(np.ptp(q, axis=0) * 0.5, 40)
    low = np.maximum(0, np.floor(points.min(0) - margin)).astype(int)
    high = np.minimum([w, h], np.ceil(points.max(0) + margin + 1)).astype(int)
    if np.any(high <= low):
        raise ValueError("Montage crop is outside the original frame")
    return [int(v) for v in (*low, *high)]


def _text(image, x, y, value, scale=0.58):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (65, 48, 32), 1, cv2.LINE_AA)


def _fit(canvas, image, rect):
    x, y, w, h = rect
    scale = min(w / image.shape[1], h / image.shape[0], 2.0)
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )
    x += (w - size[0]) // 2
    y += (h - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized


def compose_review_image(rows):
    """Rows contain BGR originals/overlays. Output coordinates never replace source coordinates."""
    if not rows or len(rows) > 2 or len({r["side"] for r in rows}) != len(rows):
        raise ValueError("Montage requires one or two distinct hand sides")
    canvas = np.full((HEADER + ROW_HEIGHT * len(rows), WIDTH, 3), 255, np.uint8)
    _text(
        canvas,
        20,
        27,
        "FINAL BOX / AXIS REVIEW    X=red  Y=green  Z=blue  yellow=tag corners",
        0.75,
    )
    _text(
        canvas,
        20,
        59,
        "Return region in ORIGINAL frame coordinates (0..1000), never montage or crop coordinates.",
        0.65,
    )
    metadata = []
    for row_index, row in enumerate(rows):
        side, raw = row["side"], row["original"]
        if side not in ("left", "right"):
            raise ValueError("Unknown montage side")
        for name in ("current", "pnp"):
            if row.get(name) is not None and row[name].shape != raw.shape:
                raise ValueError("Montage panels must use the same original frame dimensions")
        bounds = crop_bounds(raw.shape, row["quad"], row.get("projections", []))
        x0, y0, x1, y1 = bounds
        y = HEADER + row_index * ROW_HEIGHT
        _text(
            canvas,
            20,
            y + 24,
            f"{side} (side UNVERIFIED) | frame={row['frame_index']} | original={raw.shape[1]}x{raw.shape[0]}",
            0.72,
        )
        columns = [
            (20, 632, "ORIGINAL / CONTEXT"),
            (674, 286, "RAW CROP"),
            (982, 286, "CURRENT BOX + XYZ"),
            (1290, 286, "PNP HINT"),
        ]
        for x, width, title in columns:
            _text(canvas, x, y + 56, title, 0.6)
        _fit(canvas, raw, (20, y + 72, 632, 390))
        for (x, width, _), picture in zip(columns[1:], [raw, row["current"], row.get("pnp")]):
            if picture is not None:
                _fit(canvas, picture[y0:y1, x0:x1], (x, y + 83, width, 330))
            else:
                _text(canvas, x + 10, y + 230, "PnP unavailable", 0.7)
                _text(canvas, x + 10, y + 260, "axes: keep or unknown", 0.6)
        if not row.get("current_axes_available", True):
            _text(canvas, 982, y + 430, "Current axes unavailable", 0.51)
        _text(
            canvas,
            674,
            y + 455,
            f"Shared original crop: x=[{x0},{x1}), y=[{y0},{y1}). Differing from PnP is not an axis error.",
            0.42,
        )
        metadata.append(
            dict(
                side=side,
                frame_index=row["frame_index"],
                original_size_px=[raw.shape[1], raw.shape[0]],
                crop_xyxy_px=bounds,
                current_axes_available=row.get("current_axes_available", True),
                pnp_available=row.get("pnp") is not None,
                pnp_axis_convention=row.get("pnp_axis_convention", "label_z_normal"),
            )
        )
    return canvas, dict(
        layout_version=LAYOUT_VERSION, image_size_px=[WIDTH, canvas.shape[0]], rows=metadata
    )
