"""Draw three orthographic views from the exported local axes and label corners."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render(report_path, output, observation_index=0):
    report = json.loads(Path(report_path).read_text())
    observations = report["observations"]
    if not 0 <= observation_index < len(observations):
        raise ValueError("Choose an observation containing a usable label")
    observation = observations[observation_index]
    records = []
    for side, pose in observation["stickers"].items():
        if pose.get("axes_generated") is False:
            t = np.array(pose["origin_camera_m"])
            records.append((side, t[None, :], t, np.tile(t, (3, 1))))
            continue
        R, t = np.array(pose["R_object_to_camera"]), np.array(pose["t_camera_m"])
        corners = np.array(pose["object_corners_local_m"]) @ R.T + t
        axes = t + R.T * report["axis_length_m"]
        records.append((side, corners, t, axes))
    canvas = Image.new("RGB", (1560, 680), "#f5f6f7")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 15),
        f"Label reference frames | frame {observation['frame_index']}",
        font=font(26),
        fill="#17232b",
    )
    draw.text(
        (24, 55),
        "Origin: label centre. X red / Y green / Z blue. Tracker mounting is not calibrated.",
        font=font(18),
        fill="#46545e",
    )
    views = [
        ("Front", 0, 1, -1, "camera X (m)", "-camera Y (m)"),
        ("Top", 0, 2, 1, "camera X (m)", "camera Z (m)"),
        ("Side", 2, 1, -1, "camera Z (m)", "-camera Y (m)"),
    ]
    colors = ("#ef4444", "#16a34a", "#2563eb")
    for column, (name, a, b, sign, xlabel, ylabel) in enumerate(views):

        def project(points):
            points = np.asarray(points)
            return np.stack([points[..., a], points[..., b] * sign], axis=-1)

        points = np.vstack([np.zeros(3), *[np.vstack([q, t, axes]) for _, q, t, axes in records]])
        projected = project(points)
        low, high = projected.min(0) - 0.05, projected.max(0) + 0.05
        span = max(high - low)
        center = (low + high) / 2
        low, high = center - span / 2, center + span / 2
        x0, y0, size = column * 520 + 70, 160, 410

        def pixel(point):
            p = (project(point) - low) / span
            return tuple([float(x0 + p[0] * size), float(y0 + (1 - p[1]) * size)])

        draw.text((column * 520 + 70, 105), name, font=font(24), fill="#17232b")
        for tick in np.linspace(0, 1, 5):
            x, y = x0 + tick * size, y0 + (1 - tick) * size
            draw.line([(x, y0), (x, y0 + size)], fill="#dfe3e6")
            draw.line([(x0, y), (x0 + size, y)], fill="#dfe3e6")
            draw.text(
                (x - 18, y0 + size + 8),
                f"{low[0] + tick * span:.2f}",
                font=font(14),
                fill="#46545e",
            )
            draw.text(
                (x0 - 50, y - 7), f"{low[1] + tick * span:.2f}", font=font(14), fill="#46545e"
            )
        draw.rectangle((x0, y0, x0 + size, y0 + size), outline="#88949d")
        draw.text((x0 + 100, y0 + size + 40), xlabel, font=font(18), fill="#17232b")
        draw.text((x0, y0 - 26), ylabel, font=font(16), fill="#17232b")
        origin = pixel(np.zeros(3))
        draw.ellipse((origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4), fill="#17232b")
        draw.text((origin[0] + 6, origin[1]), "camera", font=font(13), fill="#17232b")
        for side, corners, t, axes in records:
            polygon = [pixel(p) for p in corners]
            draw.polygon(polygon, outline="#52616a", fill="#dce2e6")
            c = np.array(pixel(t))
            draw.text(tuple(c + [8, 6]), side, font=font(15), fill="#17232b")
            for key, end3, color in zip("XYZ", axes, colors):
                end = np.array(pixel(end3))
                direction = end - c
                length = np.linalg.norm(direction)
                if length < 2:
                    draw.ellipse((c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3), outline=color)
                    continue
                unit = direction / length
                normal = np.array([-unit[1], unit[0]])
                draw.line([tuple(c), tuple(end)], fill=color, width=3)
                head = min(9, length * 0.3)
                triangle = [
                    end,
                    end - unit * head + normal * head * 0.4,
                    end - unit * head - normal * head * 0.4,
                ]
                draw.polygon([tuple(point) for point in triangle], fill=color)
                draw.text(tuple(end + [3, -14]), key, font=font(16), fill=color)
    caption = (
        f"Source: {Path(report_path).name}; axes {report['axis_length_m'] * 1000:g} mm; "
        "one observation per plot."
    )
    draw.text((24, 650), caption, font=font(16), fill="#46545e")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--observation", type=int, default=0)
    args = parser.parse_args()
    try:
        path = render(
            args.report,
            args.out or args.report.with_name("sticker_three_views_new.png"),
            args.observation,
        )
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(path)


if __name__ == "__main__":
    main()
