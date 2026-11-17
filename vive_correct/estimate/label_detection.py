"""White-label candidates; geometry never substitutes the Tracker shell."""

from collections import Counter

import cv2
import numpy as np

from vive_correct.estimate.cv_config import CVConfig
from vive_correct.estimate.geometry import LABEL_SIZE_M, MIN_EDGE_PX, order_quad

# Outer 1–3 px ring around a white 50×30 mm label must sit on the black Tracker shell.
BORDER_BAND_PX = (1, 3)
BORDER_DARK_THRESHOLD = 100
BORDER_MIN_DARK = 0.62
LABEL_ASPECT = float(LABEL_SIZE_M[0] / LABEL_SIZE_M[1])


def default_rois(shape):
    height, width = shape[:2]
    return {
        "left": [0, height // 2, width // 2, height],
        "right": [width // 2, height // 2, width, height],
    }


def validate_rois(rois, shape):
    height, width = shape[:2]
    if not rois or set(rois) - {"left", "right"}:
        raise ValueError("ROIs must contain left and/or right")
    for box in rois.values():
        if len(box) != 4 or not all(isinstance(v, int) for v in box):
            raise ValueError("ROI must be four integer pixels [x0,y0,x1,y1]")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROI is outside the video frame")


def black_border_score(
    gray,
    quad,
    *,
    band_px=BORDER_BAND_PX,
    dark_threshold=BORDER_DARK_THRESHOLD,
):
    """Fraction of pixels in the outer band_px ring that are darker than threshold.

    band_px=(inner, outer) is inclusive distance in pixels outside the filled quad.
    Empty rings (clipped off-frame) score 0.
    """
    inner, outer = band_px
    if not (1 <= inner <= outer):
        raise ValueError("band_px must satisfy 1 <= inner <= outer")
    height, width = gray.shape
    q = np.asarray(quad, float)
    x, y, w, h = cv2.boundingRect(q.astype(np.float32))
    pad = outer + 1
    xa, ya = max(0, x - pad), max(0, y - pad)
    xb, yb = min(width, x + w + pad), min(height, y + h + pad)
    if xb <= xa or yb <= ya:
        return 0.0
    local = np.zeros((yb - ya, xb - xa), np.uint8)
    cv2.fillConvexPoly(local, np.round(q - [xa, ya]).astype(int), 255)
    kernel_outer = np.ones((2 * outer + 1, 2 * outer + 1), np.uint8)
    outer_mask = cv2.dilate(local, kernel_outer) > 0
    if inner <= 1:
        ring = outer_mask & (local == 0)
    else:
        kernel_inner = np.ones((2 * (inner - 1) + 1, 2 * (inner - 1) + 1), np.uint8)
        ring = outer_mask & (cv2.dilate(local, kernel_inner) == 0)
    if not ring.any():
        return 0.0
    return float((gray[ya:yb, xa:xb][ring] < dark_threshold).mean())


def has_tracker_body(gray, quad):
    """Require a compact dark body around the label when the wrist light is hidden."""
    height, width = gray.shape
    cx, cy = quad.mean(0).astype(int)
    x0, y0 = max(0, cx - 160), max(0, cy - 160)
    patch = gray[y0 : min(height, cy + 160), x0 : min(width, cx + 160)]
    body = (patch < 85).astype(np.uint8)
    cv2.fillConvexPoly(body, np.round(quad - [x0, y0]).astype(int), 1)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    label = labels[cy - y0, cx - x0]
    if label == 0:
        return False
    _, _, w, h, area = stats[label]
    label_area = cv2.contourArea(quad.astype(np.float32))
    return bool(
        1500 <= area <= 45000
        and 2 <= area / label_area <= 25
        and min(w, h) >= 40
        and max(w, h) / min(w, h) < 2.8
        and area / (w * h) > 0.55
    )


def quad_aspect(quad):
    edges = np.linalg.norm(np.asarray(quad, float) - np.roll(quad, -1, axis=0), axis=1)
    return float(edges.max() / max(float(edges.min()), 1e-6))


def wrist_cyan_count(image, quad):
    """Count cyan wrist-light pixels near a candidate (Tracker LED cue)."""
    height, width = image.shape[:2]
    cx, cy = np.asarray(quad, float).mean(0).astype(int)
    context = image[
        max(0, cy - 40) : min(height, cy + 240), max(0, cx - 170) : min(width, cx + 170)
    ]
    if context.size == 0:
        return 0
    b, g, r = cv2.split(context.astype(float))
    return int(((b > 100) & (g > 100) & (r < g * 0.65) & (b > g * 0.8)).sum())


def _sample(gray, points):
    return cv2.remap(
        gray.astype(np.float32),
        points[..., 0].astype(np.float32),
        points[..., 1].astype(np.float32),
        cv2.INTER_LINEAR,
    )


def _quad_crop(image, quad, pad=12):
    q = np.asarray(quad, float)
    x, y, w, h = cv2.boundingRect(q.astype(np.float32))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    return image[y0:y1, x0:x1], q - [x0, y0], np.array([x0, y0])


def black_context_metrics(image, quad, config=None, *, masks=False):
    """Measure actual black pixels outside the label, never fill the label as black.

    Enclosure is label area covered by the convex hull of a connected black
    component touching its border. This is support, not proof of Tracker identity.
    """
    config = config or CVConfig.load()
    q = np.asarray(quad, float)
    center = q.mean(0)
    outer_q = center + (q - center) * config.context_scale
    x, y, w, h = cv2.boundingRect(outer_q.astype(np.float32))
    x0, y0 = max(0, x - 2), max(0, y - 2)
    x1, y1 = min(image.shape[1], x + w + 3), min(image.shape[0], y + h + 3)
    patch = image[y0:y1, x0:x1]
    offset = np.array([x0, y0])

    def polygon(scale):
        mask = np.zeros(patch.shape[:2], np.uint8)
        cv2.fillConvexPoly(
            mask, np.round(center + (q - center) * scale - offset).astype(np.int32), 1
        )
        return mask > 0

    label, shell, outer = polygon(1), polygon(config.shell_scale), polygon(config.context_scale)
    near, context = shell & ~label, outer & ~shell
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    black = (gray < config.dark_threshold) & (patch.max(2) < config.black_channel_max)
    surrounding_black = (gray < config.surround_dark_threshold) & (
        patch.max(2) < config.black_channel_max
    )
    shell_black = black & near
    adjacent = cv2.dilate(label.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    n, components, stats, _ = cv2.connectedComponentsWithStats(shell_black.astype(np.uint8))
    # Select only a real contiguous component next to this label.
    touching = set(np.unique(components[adjacent & ~label])) - {0}
    chosen = max(touching, key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
    enclosure = np.zeros_like(gray, np.uint8)
    if chosen is not None:
        points = cv2.findNonZero((components == chosen).astype(np.uint8))
        if points is not None and len(points) >= 3:
            cv2.fillConvexPoly(enclosure, cv2.convexHull(points), 1)
    result = {
        "shell_black_fraction": float(black[near].mean()) if near.any() else 0.0,
        "surround_black_fraction": float(surrounding_black[context].mean())
        if context.any()
        else 0.0,
        "label_black_fraction": float(black[label].mean()) if label.any() else 0.0,
        "enclosed_fraction": float(enclosure[label].mean()) if label.any() else 0.0,
        "shell_visible_fraction": min(
            1.0,
            float(shell.sum())
            / max(
                cv2.contourArea((center + (q - center) * config.shell_scale).astype(np.float32)), 1
            ),
        ),
    }
    if masks:
        result.update(
            patch=patch,
            black_mask=black,
            context_black_mask=surrounding_black,
            shell_mask=near,
            context_mask=context,
            enclosure_mask=enclosure > 0,
            quad=q - offset,
            offset=offset,
        )
    return result


def box_metrics(image, quad, config=None):
    """Require neutral dark material around all four sides, not a convex hull alone."""
    from dataclasses import replace

    config = config or CVConfig.load()
    local = replace(
        config,
        shell_scale=config.box_scale,
        dark_threshold=config.box_dark_threshold,
        black_channel_max=config.box_channel_max,
    )
    evidence = black_context_metrics(image, quad, local, masks=True)
    patch, q = evidence["patch"], evidence["quad"]
    neutral = patch.max(2).astype(float) - patch.min(2) <= config.box_color_spread
    black = evidence["black_mask"] & neutral
    ring = evidence["shell_mask"]
    center = q.mean(0)
    outer = center + (q - center) * config.box_scale
    sides = []
    for i in range(4):
        j = (i + 1) % 4
        mask = np.zeros(patch.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round([q[i], q[j], outer[j], outer[i]]).astype(np.int32), 1)
        region = (mask > 0) & ring
        sides.append(float(black[region].mean()) if region.any() else 0.0)
    # Enclosure must come from the neutral-black component, not colored dark pixels.
    count, components, stats, _ = cv2.connectedComponentsWithStats((black & ring).astype(np.uint8))
    hull = np.zeros(patch.shape[:2], np.uint8)
    if count > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        points = cv2.findNonZero((components == k).astype(np.uint8))
        if points is not None and len(points) >= 3:
            cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    label = np.zeros_like(hull)
    cv2.fillConvexPoly(label, np.round(q).astype(np.int32), 1)
    enclosure = float(hull[label > 0].mean()) if label.any() else 0.0
    fraction = float(black[ring].mean()) if ring.any() else 0.0
    supported = (
        fraction >= config.box_min_black
        and min(sides) >= config.box_min_side_black
        and enclosure >= config.box_min_enclosure
    )
    return dict(
        box_supported=bool(supported),
        box_black_fraction=fraction,
        box_side_black=sides,
        box_enclosed_fraction=enclosure,
    )


def tracker_body_metrics(image, quad, config=None):
    """Measure a compact dark component; the filled label only joins its shell."""
    config = config or CVConfig.load()
    q = np.asarray(quad, float)
    center = q.mean(0)
    short = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).min()
    radius = int(np.clip(short * config.body_search_scale, 90, 280))
    cx, cy = center.astype(int)
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    patch = image[
        y0 : min(image.shape[0], cy + radius + 1), x0 : min(image.shape[1], cx + radius + 1)
    ]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    black = ((gray < config.dark_threshold) & (patch.max(2) < config.black_channel_max)).astype(
        np.uint8
    )
    cv2.fillConvexPoly(black, np.round(q - [x0, y0]).astype(np.int32), 1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(black)
    component = labels[cy - y0, cx - x0]
    x, y, w, h, area = stats[component]
    # Count repeated keys on this body, not unrelated white objects in the ROI.
    body_points = cv2.findNonZero((labels == component).astype(np.uint8))
    body_hull = cv2.convexHull(body_points)
    ratio = float(area / max(cv2.contourArea(q.astype(np.float32)), 1))
    fill = float(area / (w * h))
    touches = (
        int(x == 0) + int(y == 0) + int(x + w == patch.shape[1]) + int(y + h == patch.shape[0])
    )
    saturation = (patch.max(2).astype(float) - patch.min(2)) / np.maximum(patch.max(2), 1)
    white = ((gray > 130) & (saturation < 0.30)).astype(np.uint8)
    count, _, white_stats, centers = cv2.connectedComponentsWithStats(white)
    label_area = max(cv2.contourArea(q.astype(np.float32)), 1)
    neighbors = 0
    for k in range(1, count):
        _, _, ww, hh, aa = white_stats[k]
        if (
            0.15 * label_area <= aa <= 3 * label_area
            and aa / (ww * hh) > 0.4
            and min(ww, hh) > 4
            and cv2.pointPolygonTest(body_hull, tuple(centers[k]), False) >= 0
            and cv2.pointPolygonTest(
                q.astype(np.float32), tuple((centers[k] + [x0, y0]).tolist()), False
            )
            < 0
        ):
            neighbors += 1
    fully_visible = (
        x0 + x > 2
        and y0 + y > 2
        and x0 + x + w < image.shape[1] - 2
        and y0 + y + h < image.shape[0] - 2
    )
    compact = (
        fully_visible
        and component != 0
        and 1.3 <= ratio <= config.body_max_area_ratio
        and fill >= config.body_min_fill
        and max(w, h) / max(min(w, h), 1) < 3.5
        and touches < 3
        and neighbors < config.body_max_white_neighbors
    )
    return dict(
        compact_body=bool(compact),
        nearby_white_components=neighbors,
        body_area_ratio=ratio,
        body_fill=fill,
        body_context_edges=touches,
        body_fully_visible=bool(fully_visible),
    )


def label_metrics(image, quad, config=None):
    config = config or CVConfig.load()
    patch, q, _ = _quad_crop(image, quad)
    metrics = _label_metrics(patch, q, config)
    if "white_fraction" not in metrics or metrics["white_fraction"] < config.min_white_fraction:
        return metrics
    context = black_context_metrics(image, quad, config)
    metrics.update(context)
    enclosed = (
        context["shell_black_fraction"] >= config.min_shell_fraction
        and context["enclosed_fraction"] >= config.min_enclosed_fraction
    )
    # Three strong edges plus independent enclosure evidence can rescue one weak edge.
    supported_edges = sum(
        d >= config.min_side_dark and e >= config.min_edge_support and c >= config.min_contrast
        for d, e, c in zip(metrics["side_dark"], metrics["edge_support"], metrics["side_contrast"])
    )
    if (
        metrics["reason"] in {"weak_shell", "low_contrast", "weak_edge"}
        and enclosed
        and supported_edges >= 3
    ):
        metrics.update(accepted=True, reason=None)
    if metrics["accepted"]:
        metrics.update(box_metrics(image, quad, config))
    if metrics["accepted"] and not metrics["box_supported"]:
        metrics.update(accepted=False, reason="unsupported_black_box", score=0.0)
    if metrics["accepted"]:
        cyan = wrist_cyan_count(image, np.asarray(quad, float))
        metrics["cyan"] = cyan
        metrics.update(tracker_body_metrics(image, quad, config))
        area = cv2.contourArea(np.asarray(quad, np.float32))
        edge = float(np.mean(metrics["edge_support"]))
        shell_score = (
            0.5 * metrics["dark"]
            + 0.3 * context["shell_black_fraction"]
            + 0.2 * context["enclosed_fraction"]
        )
        compactness_bonus = 1 + 0.08 * max(
            0.0, context["shell_black_fraction"] - context["surround_black_fraction"]
        )
        metrics["score"] = (
            compactness_bonus
            * metrics["box_black_fraction"]
            * (0.5 + 0.5 * min(metrics["box_side_black"]))
            * metrics["white_fraction"]
            * edge
            * shell_score
            * (1 + 0.2 * min(np.sqrt(area) / 80, 1))
            * (1 + 0.03 * min(cyan / 80, 1))
        )
    return metrics


def _label_metrics(image, quad, config=None):
    """Original-pixel evidence; never measure whiteness on an enhanced image."""
    config = config or CVConfig.load()
    q = np.asarray(quad, float)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    short = float(np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).min())
    band = float(np.clip(short * config.border_scale, 2, 6))
    if np.any(q < band + 1) or np.any(q > np.array(image.shape[1::-1]) - band - 2):
        return {"accepted": False, "reason": "incomplete_border", "score": 0.0}
    x, y, w, h = cv2.boundingRect(q.astype(np.float32))
    local = image[y : y + h, x : x + w]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.round(q - [x, y]).astype(np.int32), 1)
    inset = max(1, int(round(short * 0.08)))
    inside = (
        cv2.erode(
            mask,
            np.ones((2 * inset + 1, 2 * inset + 1), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    if not inside.any():
        return {"accepted": False, "reason": "empty_interior", "score": 0.0}
    pixels = local[inside].astype(float)
    saturation = (pixels.max(1) - pixels.min(1)) / np.maximum(pixels.max(1), 1)
    lab = cv2.cvtColor(local, cv2.COLOR_BGR2LAB)[inside].astype(float)
    chroma = np.linalg.norm(lab[:, 1:] - 128, axis=1)
    lum = gray[y : y + h, x : x + w][inside]
    light = lum >= max(25.0, float(np.percentile(lum, 75)) * 0.55)
    white = float(
        ((saturation <= config.max_saturation) & (chroma <= config.max_chroma) & light).mean()
    )
    side_dark, contrasts, supports = [], [], []
    center = q.mean(0)
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        if np.dot(normal, center - (a + b) / 2) < 0:
            normal = -normal
        base = a + (b - a) * np.linspace(0.12, 0.88, 32)[:, None]
        distances = np.linspace(1.5, band, 4)
        outer = _sample(gray, base[:, None] - distances[None, :, None] * normal)
        inner = _sample(gray, base[:, None] + distances[None, :, None] * normal)
        side_dark.append(float((outer < config.dark_threshold).mean()))
        contrast = np.median(inner, axis=1) - np.median(outer, axis=1)
        contrasts.append(float(np.median(contrast)))
        supports.append(float((contrast >= config.min_contrast).mean()))
    dark = float(np.mean(side_dark))
    reason = next(
        (
            name
            for failed, name in (
                (white < config.min_white_fraction, "not_white"),
                (
                    dark < config.min_dark_fraction or min(side_dark) < config.min_side_dark,
                    "weak_shell",
                ),
                (min(contrasts) < config.min_contrast, "low_contrast"),
                (min(supports) < config.min_edge_support, "weak_edge"),
            )
            if failed
        ),
        None,
    )
    # Bounded area reward cannot outweigh weak color or edge support.
    area = float(cv2.contourArea(q.astype(np.float32)))
    score = white * min(supports) * dark * (1 + 0.2 * min(np.sqrt(area) / 80, 1))
    return {
        "accepted": reason is None,
        "reason": reason,
        "white_fraction": white,
        "chroma_median": float(np.median(chroma)),
        "brightness_quantiles": np.percentile(lum, [10, 50, 90]).tolist(),
        "side_dark": side_dark,
        "side_contrast": contrasts,
        "edge_support": supports,
        "dark": dark,
        "score": float(score if reason is None else 0),
    }


def label_likeness_score(gray, quad, image=None, config=None):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image is None else image
    return label_metrics(image, quad, config)["score"]


def quad_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def detect_label_candidates(
    image, rois=None, *, config=None, min_edge_px=MIN_EDGE_PX, preserve_variants=False
):
    """Detailed candidates and rejection counts; ROI names are only spatial hints."""
    config = config or CVConfig.load()
    rois = default_rois(image.shape) if rois is None else rois
    validate_rois(rois, image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rejected, found = Counter(), {}
    for side, (x0, y0, x1, y1) in rois.items():
        lum = gray[y0:y1, x0:x1]
        patch = image[y0:y1, x0:x1]
        scale = min(1.0, config.proposal_max_width / patch.shape[1])
        if scale < 1:
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # Use actual independent scales after rounding the dimensions.
            factors = np.array([patch.shape[1] / (x1 - x0), patch.shape[0] / (y1 - y0)])
            lum = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            factors = np.ones(2)
        saturation = (patch.max(2).astype(float) - patch.min(2)) / np.maximum(patch.max(2), 1)
        neutral = saturation < max(0.30, config.max_saturation)
        masks = [((lum > t) & neutral).astype(np.uint8) * 255 for t in config.thresholds]
        local_mean = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 15)
        masks.append(
            ((lum - local_mean > config.min_contrast * 0.4) & neutral).astype(np.uint8) * 255
        )
        candidates, seen = [], set()
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x < 2 or y < 2 or x + w > patch.shape[1] - 2 or y + h > patch.shape[0] - 2:
                    rejected["crop_boundary"] += 1
                    continue
                area = cv2.contourArea(contour) / float(factors.prod())
                if (
                    not config.min_area
                    <= area
                    <= image.shape[0] * image.shape[1] * config.max_area_fraction
                ):
                    continue
                hull = cv2.convexHull(contour)
                for epsilon in config.approx_epsilons:
                    quad = cv2.approxPolyDP(hull, epsilon * cv2.arcLength(hull, True), True)
                    if len(quad) != 4:
                        continue
                    try:
                        q = order_quad(quad.reshape(4, 2) / factors + [x0, y0], min_edge_px)
                    except ValueError:
                        rejected["geometry"] += 1
                        continue
                    key = tuple(q.ravel())
                    if key in seen:
                        continue
                    seen.add(key)
                    fill = area / max(cv2.contourArea(q.astype(np.float32)), 1)
                    if not config.min_fill <= fill <= 1 / config.min_fill:
                        rejected["poor_fill"] += 1
                        continue
                    original_metrics = label_metrics(image, q, config)
                    if original_metrics.get("reason") in {
                        "not_white",
                        "empty_interior",
                        "incomplete_border",
                    }:
                        rejected[original_metrics["reason"]] += 1
                        continue
                    refined = refine_corners(image, q, config=config)
                    metrics = label_metrics(image, refined, config)
                    if not metrics["accepted"]:
                        refined, metrics = q, original_metrics
                    if not metrics["accepted"]:
                        rejected[metrics["reason"]] += 1
                        continue
                    metrics.update(
                        quad=refined,
                        fill=float(fill),
                        refinement_shift=float(np.linalg.norm(refined - q, axis=1).max()),
                    )
                    candidates.append(metrics)
        ranked = sorted(candidates, key=lambda c: -c["score"])
        kept = []
        for candidate in ranked:
            if any(quad_iou(candidate["quad"], c["quad"]) >= config.nms_iou for c in kept):
                continue
            kept.append(candidate)
        found[side] = kept if preserve_variants else kept[:8]
    return {"candidates": found, "rejected": dict(rejected)}


def detect_labels(
    image, rois=None, *, preserve_variants=False, min_edge_px=MIN_EDGE_PX, config=None
):
    result = detect_label_candidates(
        image, rois, config=config, min_edge_px=min_edge_px, preserve_variants=preserve_variants
    )
    return {
        side: [c["quad"] for c in candidates] for side, candidates in result["candidates"].items()
    }


def select_best_label_frames(frames, rois=None):
    """Keep at most one best left and one best right label observation across frames.

    ``frames`` is an iterable of BGR images or ``(frame_index, image)`` pairs.
    Returns ``{side: {frame_index, quad, score, has_tracker_body, dark, aspect}}``.
    Legacy spatial-ROI helper, not physical-hand identification. For sequential
    identity tracking and geometry/stability gates use cv_tracking.select_episode.
    """
    best = {}
    for seq, item in enumerate(frames):
        if isinstance(item, tuple) and len(item) == 2 and not isinstance(item[0], np.ndarray):
            index, image = item
        else:
            index, image = seq, item
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for side, quads in detect_labels(image, rois).items():
            for quad in quads:
                body = has_tracker_body(gray, quad)
                cyan = wrist_cyan_count(image, quad)
                score = label_likeness_score(gray, quad, image)
                if score <= 0:
                    continue
                candidate = {
                    "frame_index": int(index),
                    "quad": np.asarray(quad, float),
                    "score": float(score),
                    "has_tracker_body": bool(body),
                    "cyan": int(cyan),
                    "dark": black_border_score(gray, quad),
                    "aspect": quad_aspect(quad),
                }
                current = best.get(side)
                if current is None:
                    best[side] = candidate
                    continue
                if score > current["score"]:
                    best[side] = candidate
    return best


def refine_corners(image, quad, *, config=None):
    patch, local, offset = _quad_crop(image, quad)
    return _refine_corners(patch, local, config=config) + offset


def _refine_corners(image, quad, *, config=None):
    """Fit four gradient-supported edge lines; keep the input on weak support."""
    config = config or CVConfig.load()
    q = np.asarray(quad, float)
    radius = float(
        np.clip(
            np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1).min() * config.refine_radius_scale,
            2,
            6,
        )
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lines = []
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        v = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-v[1], v[0]])
        if np.dot(normal, q.mean(0) - (a + b) / 2) < 0:
            normal = -normal
        base = a + (b - a) * np.linspace(0.15, 0.85, 30)[:, None]
        points = base[:, None] + np.linspace(-radius, radius, 49)[None, :, None] * normal
        x, y = points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32)
        response = (
            cv2.remap(gx, x, y, cv2.INTER_LINEAR) * normal[0]
            + cv2.remap(gy, x, y, cv2.INTER_LINEAR) * normal[1]
        )
        indices = response.argmax(axis=1)
        good = response[np.arange(30), indices] > 100
        if good.sum() < 18:
            return q
        peaks = points[np.arange(30), indices][good].astype(np.float32)
        vx, vy, cx, cy = cv2.fitLine(peaks, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        n = np.array([-vy, vx])
        if np.percentile(abs((peaks - [cx, cy]) @ n), 90) > config.max_line_residual:
            return q
        lines.append((n, float(n @ [cx, cy])))
    try:
        refined = np.array(
            [
                np.linalg.solve(
                    np.array([lines[i - 1][0], lines[i][0]]),
                    [lines[i - 1][1], lines[i][1]],
                )
                for i in range(4)
            ]
        )
    except np.linalg.LinAlgError:
        return q
    if np.max(
        np.linalg.norm(refined - q, axis=1)
    ) > config.max_refine_shift or not cv2.isContourConvex(refined.astype(np.float32)):
        return q
    return refined
