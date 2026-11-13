"""White-label candidates; geometry never substitutes the Tracker shell."""

import cv2
import numpy as np

from vive_correct.estimate.geometry import MIN_EDGE_PX, order_quad


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


def detect_labels(image, rois=None, *, preserve_variants=False, min_edge_px=MIN_EDGE_PX):
    """Conservative CV candidates in BGR images, ranked within each wrist ROI.

    Without supplied ROIs, require a nearby wrist light or compact dark body.
    Image-side assignment assumes uncrossed hands and still needs review.
    """
    supplied_rois = rois is not None
    rois = default_rois(image.shape) if rois is None else rois
    validate_rois(rois, image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = {}
    height, width = gray.shape
    for side, (x0, y0, x1, y1) in rois.items():
        patch = image[y0:y1, x0:x1]
        lum = gray[y0:y1, x0:x1]
        saturation = (patch.max(2).astype(float) - patch.min(2)) / np.maximum(patch.max(2), 1)
        candidates = []
        thresholds = range(55, 236, 20) if preserve_variants else (95, 120, 145, 175, 205)
        for threshold in thresholds:
            mask = ((lum > threshold) & (saturation < 0.30)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x < 2 or y < 2 or x + w > patch.shape[1] - 2 or y + h > patch.shape[0] - 2:
                    # A crop boundary is not an observed label edge.
                    continue
                area = cv2.contourArea(contour)
                if not 180 <= area <= (30000 if preserve_variants else 9000):
                    continue
                hull = cv2.convexHull(contour)
                quad = cv2.approxPolyDP(hull, 0.025 * cv2.arcLength(hull, True), True)
                if len(quad) != 4:
                    continue
                try:
                    q = order_quad(quad.reshape(4, 2) + [x0, y0], min_edge_px)
                except ValueError:
                    continue
                if any(
                    (
                        min(
                            np.linalg.norm(q - np.roll(item["quad"], shift, axis=0), axis=1).max()
                            for shift in range(4)
                        )
                        < 1.5
                        if preserve_variants
                        else np.linalg.norm(q.mean(0) - item["quad"].mean(0)) < 8
                    )
                    for item in candidates
                ):
                    continue
                x, y, w, h = cv2.boundingRect(q.astype(np.float32))
                if max(w, h) > (360 if preserve_variants else 200):
                    continue
                xa, ya, xb, yb = (
                    max(0, x - 8),
                    max(0, y - 8),
                    min(width, x + w + 8),
                    min(height, y + h + 8),
                )
                local = np.zeros((yb - ya, xb - xa), np.uint8)
                cv2.fillConvexPoly(local, np.round(q - [xa, ya]).astype(int), 255)
                ring = cv2.dilate(local, np.ones((11, 11), np.uint8)) > 0
                ring &= cv2.dilate(local, np.ones((3, 3), np.uint8)) == 0
                dark = float((gray[ya:yb, xa:xb][ring] < 100).mean()) if ring.any() else 0
                if dark < 0.62:
                    continue
                cx, cy = q.mean(0).astype(int)
                context = image[
                    max(0, cy - 40) : min(height, cy + 240), max(0, cx - 170) : min(width, cx + 170)
                ]
                b, g, r = cv2.split(context.astype(float))
                cyan = int(((b > 100) & (g > 100) & (r < g * 0.65) & (b > g * 0.8)).sum())
                if not supplied_rois and cyan < 8 and not has_tracker_body(gray, q):
                    continue
                candidates.append({"quad": q, "score": float(np.sqrt(area) * dark)})
        ranked = sorted(candidates, key=lambda c: -c["score"])
        found[side] = [item["quad"] for item in (ranked if preserve_variants else ranked[:8])]
    return found


def refine_corners(image, quad):
    """Fit four gradient-supported edge lines; keep the input on weak support."""
    q = np.asarray(quad, float)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    lines = []
    for a, b in zip(q, np.roll(q, -1, axis=0)):
        v = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-v[1], v[0]])
        base = a + (b - a) * np.linspace(0.15, 0.85, 30)[:, None]
        points = base[:, None] + np.linspace(-4, 4, 33)[None, :, None] * normal
        x, y = points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32)
        response = abs(
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
    if np.max(np.linalg.norm(refined - q, axis=1)) > 6 or not cv2.isContourConvex(
        refined.astype(np.float32)
    ):
        return q
    return refined
