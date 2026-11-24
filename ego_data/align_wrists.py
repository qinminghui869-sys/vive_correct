"""Desktop tool: pick tracker targets on calibrated RGB + depth, fit each wrist."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageTk
from render_rerun import frames, table, nearest
from wrist_alignment import pose_matrix, backproject, project_world, fit_samples
from depth_metadata import load_metadata

ROOT = Path(__file__).resolve().parent
COLORS = {"left": "#00ddff", "right": "#ff5599"}


def render_style_arguments(path):
    if path is None:
        return []
    options = json.loads(Path(path).read_text())["render_options"]
    allowed = {
        "stride",
        "pose-markers",
        "pose-axis-length",
        "head-axis-length",
        "frustum-depth",
        "pose-axis-radius",
        "frustum-radius",
        "hand-point-radius",
        "cloud-point-radius",
        "bone-radius",
        "hand-cloud-fade-radius",
    }
    if not set(options) <= allowed:
        raise ValueError("Unknown fixed render-style option")
    return [part for name, value in options.items() for part in ("--" + name, str(value))]


class App:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.mount = json.loads(args.calibration.read_text())
        from vive_correct.data.layout import (
            annotation_dir,
            data_directory,
            dataset_root,
            episode_stem,
        )

        self.data = data_directory(self.dataset)
        self.videos = dataset_root(self.dataset) / "videos/chunk-000"
        self.stem = episode_stem(self.dataset)
        self.depth_key = "observation.images.depth_headband_left_rectified"
        self.ann = annotation_dir(self.dataset, self.depth_key)
        self.k = load_metadata(self.ann)["intrinsics"]["left"]
        self.times = pq.read_table(self.data / (self.depth_key + ".parquet"))[
            "timestamp"
        ].to_numpy()
        self.streams = {
            key: table(self.data / f"observation.{key}.parquet")
            for key in ["head_pose", "left_wrist_pose", "right_wrist_pose"]
        }
        self.samples = []
        self.models = {
            side: ("world" if args.world_only else args.default_model) for side in ("left", "right")
        }
        self.manual_offsets = {}
        if args.output.exists():
            saved = json.loads(args.output.read_text())
            if saved.get("camera_calibration") != self.mount:
                raise ValueError(
                    "保存的样本使用了不同相机标定。请使用新的 --output 路径，避免混用。"
                )
            if saved.get("pose_euler_sequence") != args.pose_euler_sequence or saved.get(
                "pose_degrees"
            ) != (not args.pose_radians):
                raise ValueError(
                    "保存的校准使用了不同姿态角约定，请保持参数一致或使用新的 --output 路径。"
                )
            if saved.get("world_axes", args.world_axes) != args.world_axes:
                raise ValueError("保存的校准与当前世界轴向不一致")
            self.samples = saved.get("samples", [])
            self.models.update({s: e["model"] for s, e in saved.get("wrists", {}).items()})
            for side, entry in saved.get("wrists", {}).items():
                if entry.get("method") == "manual":
                    offset = np.asarray(entry["offset_m"], dtype=float)
                    if offset.shape != (3,) or not np.isfinite(offset).all():
                        raise ValueError("保存的手动偏移无效")
                    self.manual_offsets[side] = offset
        if args.world_only and any(model != "world" for model in self.models.values()):
            raise ValueError(
                "此入口只使用固定世界偏移；旧局部偏移请另存后重新设置，不能直接解释为世界偏移。"
            )
        self.render_style_args = render_style_arguments(args.render_settings)
        if args.end is None:
            args.end = len(self.times) - 1
        if not 0 <= args.start <= args.end < len(self.times):
            raise ValueError("帧范围超出深度视频")
        initial = args.start if args.initial_frame is None else args.initial_frame
        if not args.start <= initial <= args.end:
            raise ValueError("初始帧超出范围")
        self.root = tk.Tk()
        self.root.title("左右腕 VIVE 对齐 — RGB + 深度对应点")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.pending = None
        self.loaded = None
        self.side = tk.StringVar(value="left")
        self.model = tk.StringVar(value=self.models["left"])
        self.frame = tk.IntVar(value=initial)
        self.offset_vars = [tk.StringVar(value="0") for _ in range(3)]
        self.depth_text = tk.StringVar()
        self.status = tk.StringVar(value="正在加载；点击图像中腕部 VIVE 应在的位置。")
        self.build()
        self.refresh_list()
        self.sync_offset_entries()
        self.load()

    def build(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Label(bar, text="深度帧").pack(side="left")
        ttk.Spinbox(
            bar, from_=self.args.start, to=self.args.end, textvariable=self.frame, width=6
        ).pack(side="left")
        ttk.Button(bar, text="加载帧", command=self.load).pack(side="left")
        ttk.Button(bar, text="上一帧", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(bar, text="下一帧", command=lambda: self.step(1)).pack(side="left")
        for label, value in [("左腕", "left"), ("右腕", "right")]:
            ttk.Radiobutton(
                bar, text=label, variable=self.side, value=value, command=self.change_side
            ).pack(side="left", padx=8)
        if self.args.world_only:
            ttk.Label(bar, text="左右腕独立 · 固定世界坐标偏移").pack(side="left", padx=8)
        else:
            ttk.Label(bar, text="拟合模型").pack(side="left")
            combo = ttk.Combobox(
                bar,
                textvariable=self.model,
                values=["wrist_local", "world"],
                state="readonly",
                width=12,
            )
            combo.pack(side="left")
            combo.bind("<<ComboboxSelected>>", self.change_model)
        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(content, width=960, height=592, bg="#111111", highlightthickness=0)
        self.canvas.pack(side="left", padx=6, pady=6)
        self.canvas.bind("<Button-1>", self.pick)
        panel = ttk.Frame(content)
        panel.pack(side="left", fill="both", expand=True, padx=6)
        ttk.Label(
            panel,
            text="空心：原腕点；实心：修正腕点\n蓝左 / 粉右；十字：所选目标\n可输入偏移，或点击图像添加对应点。",
            justify="left",
        ).pack(anchor="w")
        ttk.Label(panel, text="直接输入当前侧偏移（毫米）").pack(anchor="w", pady=(10, 0))
        row = ttk.Frame(panel)
        row.pack(fill="x")
        for axis, var in zip("XYZ", self.offset_vars):
            ttk.Label(row, text=axis).pack(side="left")
            ttk.Spinbox(row, from_=-1000, to=1000, increment=1, textvariable=var, width=7).pack(
                side="left", padx=2
            )
        ttk.Button(panel, text="应用手动偏移并预览", command=self.apply_manual).pack(
            fill="x", pady=3
        )
        ttk.Button(panel, text="恢复按选点拟合", command=self.use_fit).pack(fill="x", pady=3)
        ttk.Label(
            panel,
            text="世界轴：X左、Y上、Z前。彩色轴遵循所选模型。\n左右手独立；输入后请点击应用。",
            justify="left",
        ).pack(anchor="w")
        ttk.Label(panel, text="目标深度 Z（毫米，可手动修正）").pack(anchor="w", pady=(12, 0))
        ttk.Entry(panel, textvariable=self.depth_text, width=14).pack(anchor="w")
        ttk.Button(panel, text="添加当前对应点", command=self.add_sample).pack(fill="x", pady=4)
        self.listbox = tk.Listbox(panel, width=42, height=6, exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        ttk.Button(panel, text="删除选中样本", command=self.delete_sample).pack(fill="x", pady=3)
        self.stats = ttk.Label(panel, text="", justify="left")
        self.stats.pack(anchor="w", pady=6)
        ttk.Button(panel, text="保存校准", command=self.save).pack(fill="x", pady=3)
        self.export_button = ttk.Button(panel, text="保存并生成 Rerun", command=self.export)
        self.export_button.pack(fill="x", pady=3)
        ttk.Label(self.root, textvariable=self.status, wraplength=1250).pack(
            fill="x", padx=8, pady=4
        )
        ttk.Label(
            self.root,
            text=(
                "世界轴：X左、Y上、Z前。左右侧各自的固定偏移应用于全部帧，手部内部几何不变。\n注意：可见表面深度不一定是 Tracker 原点深度。"
                if self.args.world_only
                else "wrist_local：腕部局部固定偏移；world：世界固定偏移。可见表面深度不一定是 Tracker 原点深度。"
            ),
            justify="left",
        ).pack(anchor="w", padx=8, pady=4)

    def step(self, delta):
        if not self.busy:
            self.frame.set(max(self.args.start, min(self.args.end, self.frame.get() + delta)))
            self.load()

    def change_side(self):
        self.model.set(self.models[self.side.get()])
        self.pending = None
        self.sync_offset_entries()
        self.redraw()

    def change_model(self, event=None):
        self.models[self.side.get()] = self.model.get()
        self.sync_offset_entries()
        self.redraw()

    def sync_offset_entries(self):
        offset, _ = self.fit(self.side.get())
        for var, value in zip(self.offset_vars, offset * 1000):
            var.set(f"{value:.2f}")

    def apply_manual(self):
        if self.busy:
            return
        try:
            offset = np.array([float(var.get()) for var in self.offset_vars]) / 1000
            if not np.isfinite(offset).all():
                raise ValueError("XYZ 必须为有限数值")
            side = self.side.get()
            self.manual_offsets[side] = offset
            self.redraw()
            self.status.set(f"{side} 手动偏移已预览；保存后应用到整个导出帧范围。另一侧独立设置。")
        except ValueError as e:
            messagebox.showerror("偏移错误", str(e))

    def use_fit(self):
        self.manual_offsets.pop(self.side.get(), None)
        self.sync_offset_entries()
        self.redraw()

    def read_frame(self, index):
        depth = next(frames(self.videos / self.depth_key / f"{self.stem}.mp4", index, index))
        mask = next(frames(self.ann / "valid_mask_ffv1.mkv", index, index))
        rgb = next(
            frames(
                self.videos / f"observation.images.headband_left_rgb_rectified/{self.stem}.mp4",
                index,
                index,
                True,
            )
        )
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        t = float(self.times[index])
        poses = {k: nearest(v, t, 0.005) for k, v in self.streams.items()}
        camera = pose_matrix(
            poses["head_pose"], self.mount["head_euler_sequence"], self.mount["head_euler_degrees"]
        ) @ np.array(self.mount["T_head_rectified_camera"])
        return index, depth, mask, rgb, poses, camera

    def load(self):
        if self.busy:
            return
        try:
            index = self.frame.get()
            if not self.args.start <= index <= self.args.end:
                raise ValueError("帧超出范围")
        except Exception as e:
            messagebox.showerror("帧错误", str(e))
            return
        self.busy = True
        self.pending = None
        self.status.set(f"正在精确解码第 {index} 帧…")
        future = self.pool.submit(self.read_frame, index)
        self.poll(future, self.loaded_frame)

    def poll(self, future, callback):
        if not future.done():
            self.root.after(100, lambda: self.poll(future, callback))
            return
        try:
            callback(future.result())
        except Exception as e:
            self.busy = False
            self.export_button.state(["!disabled"])
            self.status.set(str(e))
            messagebox.showerror("操作失败", str(e))

    def loaded_frame(self, value):
        self.loaded = value
        self.busy = False
        self.frame.set(value[0])
        self.redraw()
        self.status.set(f"第 {value[0]} 帧已加载。选择左腕或右腕，然后点击对应的 VIVE 位置。")

    def fit(self, side, model=None):
        samples = [s for s in self.samples if s["side"] == side]
        if model is None and side in self.manual_offsets:
            offset = self.manual_offsets[side]
            residuals = []
            for sample in samples:
                predicted = (
                    np.array(sample["wrist_rotation"]) @ offset
                    if self.models[side] == "wrist_local"
                    else offset
                )
                residuals.append(
                    float(np.linalg.norm(predicted - np.array(sample["delta_world_m"])))
                )
            return offset, residuals
        return fit_samples(samples, model or self.models[side])

    def marker(self, u, v, color, filled=False, label=""):
        x, y = u * self.scale, v * self.scale
        self.canvas.create_oval(
            x - 6, y - 6, x + 6, y + 6, outline=color, fill=color if filled else "", width=2
        )
        if label:
            self.canvas.create_text(x + 9, y - 10, text=label, fill=color, anchor="w")

    def redraw(self):
        if self.loaded is None:
            return
        index, depth, mask, rgb, poses, camera = self.loaded
        self.scale = 960 / rgb.shape[1]
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(rgb).resize((960, round(rgb.shape[0] * self.scale)))
        )
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        for side in ("left", "right"):
            p = poses[f"{side}_wrist_pose"]
            if not np.isfinite(p).all():
                continue
            offset, _ = self.fit(side)
            delta = (
                pose_matrix(p, self.args.pose_euler_sequence, not self.args.pose_radians)[:3, :3]
                @ offset
                if self.models[side] == "wrist_local"
                else offset
            )
            for point, filled, label in [
                (p[:3], False, side + " raw"),
                (p[:3] + delta, True, side + " corrected"),
            ]:
                uv = project_world(point, camera, self.k)
                if uv:
                    self.marker(*uv, COLORS[side], filled, label)
            if side == self.side.get():
                rotation = (
                    pose_matrix(p, self.args.pose_euler_sequence, not self.args.pose_radians)[
                        :3, :3
                    ]
                    if self.models[side] == "wrist_local"
                    else np.eye(3)
                )
                base = p[:3] + delta
                start = project_world(base, camera, self.k)
                if start:
                    for axis, direction, color in zip(
                        "XYZ", rotation.T, ["#ff5050", "#40ee70", "#5599ff"]
                    ):
                        tip = project_world(base + direction * 0.05, camera, self.k)
                        if tip:
                            self.canvas.create_line(
                                start[0] * self.scale,
                                start[1] * self.scale,
                                tip[0] * self.scale,
                                tip[1] * self.scale,
                                fill=color,
                                width=2,
                                arrow="last",
                            )
                            self.canvas.create_text(
                                tip[0] * self.scale,
                                tip[1] * self.scale,
                                text=axis,
                                fill=color,
                                anchor="sw",
                            )
        for sample in self.samples:
            if sample["frame"] == index:
                self.cross(*sample["pixel"], COLORS[sample["side"]])
        if self.pending:
            self.cross(*self.pending["pixel"], "#ffff00")
        text = []
        for side in ("left", "right"):
            offset, errors = self.fit(side)
            text.append(
                f"{side}: {'手动' if side in self.manual_offsets else '选点拟合'} / {len(errors)} 个样本 / {self.models[side]}\n偏移 mm: "
                + ", ".join(f"{v:.1f}" for v in offset * 1000)
            )
            if errors:
                text.append(f"拟合残差 RMS: {np.sqrt(np.mean(np.square(errors))) * 1000:.1f} mm")
                if not self.args.world_only:
                    other = "world" if self.models[side] == "wrist_local" else "wrist_local"
                    _, other_errors = self.fit(side, other)
                    text.append(
                        f"另一模型 {other} RMS: {np.sqrt(np.mean(np.square(other_errors))) * 1000:.1f} mm"
                    )
        self.stats.config(text="\n".join(text))

    def cross(self, u, v, color):
        x, y = u * self.scale, v * self.scale
        self.canvas.create_line(x - 8, y, x + 8, y, fill=color, width=2)
        self.canvas.create_line(x, y - 8, x, y + 8, fill=color, width=2)

    def pick(self, event):
        if self.busy or self.loaded is None:
            return
        index, depth, mask, *_ = self.loaded
        u, v = int(event.x / self.scale), int(event.y / self.scale)
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return
        self.pending = {"pixel": [u, v], "side": self.side.get(), "frame": index}
        valid = mask[v, u] > 0 and 0 < depth[v, u] <= 4000
        self.depth_text.set(str(int(depth[v, u])) if valid else "")
        self.redraw()
        self.status.set(
            f"{self.side.get()} 目标像素 ({u},{v})，"
            + (
                f"深度 {depth[v, u]} mm。确认后添加样本。"
                if valid
                else "此处深度无效。请换点或手动输入可靠的目标深度。"
            )
        )

    def add_sample(self):
        if not self.pending or self.busy:
            return
        try:
            z = float(self.depth_text.get()) / 1000
            if not np.isfinite(z) or not 0 < z <= 4:
                raise ValueError("请输入 0–4000mm 之间的有效深度")
            index, depth, mask, rgb, poses, camera = self.loaded
            side = self.pending["side"]
            p = poses[f"{side}_wrist_pose"]
            rotation = pose_matrix(p, self.args.pose_euler_sequence, not self.args.pose_radians)[
                :3, :3
            ]
            u, v = self.pending["pixel"]
            target = camera[:3, :3] @ backproject(u, v, z, self.k) + camera[:3, 3]
            self.samples.append(
                dict(
                    side=side,
                    frame=index,
                    timestamp=float(self.times[index]),
                    pixel=[u, v],
                    depth_m=z,
                    target_world_m=target.tolist(),
                    source_world_m=p[:3].tolist(),
                    delta_world_m=(target - p[:3]).tolist(),
                    wrist_rotation=rotation.tolist(),
                )
            )
            self.manual_offsets.pop(side, None)
            self.pending = None
            self.refresh_list()
            self.sync_offset_entries()
            self.redraw()
            self.status.set(
                "样本已添加；建议换到其他帧，特别是手腕转动后的帧，继续选点。尚未保存。"
            )
        except Exception as e:
            messagebox.showerror("无法添加", str(e))

    def refresh_list(self):
        self.listbox.delete(0, "end")
        for s in self.samples:
            self.listbox.insert(
                "end", f"{s['side']} | 帧 {s['frame']} | {s['pixel']} | {s['depth_m'] * 1000:.0f}mm"
            )

    def delete_sample(self):
        selected = self.listbox.curselection()
        if selected:
            del self.samples[selected[0]]
            self.refresh_list()
            self.sync_offset_entries()
            self.redraw()

    def save(self):
        result = {
            "version": 2,
            "camera_calibration": self.mount,
            "world_axes": self.args.world_axes,
            "pose_euler_sequence": self.args.pose_euler_sequence,
            "pose_degrees": not self.args.pose_radians,
            "samples": self.samples,
            "wrists": {},
        }
        for side in ("left", "right"):
            offset, errors = self.fit(side)
            if errors or side in self.manual_offsets:
                result["wrists"][side] = {
                    "model": self.models[side],
                    "offset_m": offset.tolist(),
                    "method": "manual" if side in self.manual_offsets else "point_fit",
                    "sample_count": len(errors),
                    "residuals_m": errors,
                }
        self.args.output.parent.mkdir(exist_ok=True, parents=True)
        temporary = self.args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        temporary.replace(self.args.output)
        self.status.set(f"已保存 {self.args.output}；未校准的一侧保持原始位置。")

    def export(self):
        if self.busy:
            return
        if not self.samples and not self.manual_offsets:
            messagebox.showinfo("需要偏移", "请添加一个对应点，或输入 XYZ 并点击应用手动偏移。")
            return
        self.save()
        self.busy = True
        self.export_button.state(["disabled"])
        output = self.args.rerun_output
        cmd = [
            sys.executable,
            str(ROOT / "render_rerun.py"),
            "--dataset",
            str(self.dataset),
            "--start",
            str(self.args.start),
            "--end",
            str(self.args.end),
            "--calibration",
            str(self.args.calibration),
            "--wrist-calibration",
            str(self.args.output),
            "--hide-axes",
            "--world-axes",
            self.args.world_axes,
            "--pose-euler-sequence",
            self.args.pose_euler_sequence,
            "--output",
            str(output),
        ]
        cmd.extend(self.render_style_args)
        if self.args.pose_radians:
            cmd.append("--pose-radians")

        def render():
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-2500:])
            return output

        self.status.set("正在生成校准后的 Rerun 录制，通常需要几十秒…")
        self.poll(self.pool.submit(render), self.exported)

    def exported(self, path):
        self.busy = False
        self.export_button.state(["!disabled"])
        self.status.set(f"生成完成：{path}")
        messagebox.showinfo("已生成", f"用 Rerun 打开：\n{path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "L3-001-2026-10-27-13-47-37")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT.parent / "calibrations/left_camera_to_head_vive.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wrist_alignment_A.json")
    parser.add_argument(
        "--world-only", action="store_true", help="Lock both wrists to fixed world-space offsets"
    )
    parser.add_argument("--default-model", choices=["world", "wrist_local"], default="wrist_local")
    parser.add_argument("--render-settings", type=Path, help="Frozen render appearance JSON")
    parser.add_argument(
        "--initial-frame",
        type=int,
        help="Initially displayed frame; does not restrict exported frames",
    )
    parser.add_argument(
        "--rerun-output", type=Path, default=ROOT / "outputs/l3_A_wrists_aligned_all.rrd"
    )
    parser.add_argument("--world-axes", choices=["LUF", "RUB"], default="LUF")
    parser.add_argument("--start", type=int, default=360)
    parser.add_argument("--end", type=int, default=540)
    parser.add_argument("--pose-euler-sequence", default="xyz")
    parser.add_argument("--pose-radians", action="store_true")
    args = parser.parse_args(argv)
    if args.end == -1:
        args.end = None
    if args.start < 0 or (args.end is not None and args.start > args.end):
        parser.error("Invalid frame range")
    app = App(args)
    app.root.mainloop()


if __name__ == "__main__":
    main()
