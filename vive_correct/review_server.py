"""Single-user browser review on a loopback-only HTTP server."""

import argparse
import getpass
import hashlib
import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, monotonic, time
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from vive_correct.human_review import ReviewStore, decode_frame, output_directory
from vive_correct.dataset import VIDEO_SUFFIX
from vive_correct.rerun_export import DEFAULT_MOUNT
from vive_correct.review_actions import export_approved
from vive_correct.review_bundle import approve, review_state
from vive_correct.review_io import read_json, atomic_json, output_lock
from vive_correct.axis_calibration import preview, axis_angles, local_rotation, enrich_pose
from vive_correct.runtime_cache import CACHE, cached_file, file_stamp, source_stamp
from vive_correct.feature_progress import scan_progress
from vive_correct.task_locks import reserve, adopt, operation_lock
from vive_correct.source_data import baseline_read
from vive_correct.review_exclusions import FILENAME

DEFAULT_PORT = 18763
UI = Path(__file__).with_name("review_ui")


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        output,
        port=DEFAULT_PORT,
        calibration=DEFAULT_MOUNT,
        *,
        approve_jobs=4,
        export_jobs=1,
        preview_jobs=8,
        frame_jobs=8,
        cache_mib=2048,
    ):
        if min(approve_jobs, export_jobs, preview_jobs, frame_jobs) < 1:
            raise ValueError("并发数必须为正整数")
        if cache_mib <= 0:
            raise ValueError("--cache-mib 必须为正整数")
        CACHE.resize(cache_mib * 1024 * 1024)
        self.store = ReviewStore(output, allow_empty=True)
        self.calibration = Path(calibration)
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.job_lock = threading.Lock()
        self.catalog_lock = threading.Lock()
        self.last_refresh = 0.0
        self.preview_slots = threading.BoundedSemaphore(preview_jobs)
        self.frame_slots = threading.BoundedSemaphore(frame_jobs)
        self.known_jobs = set()
        self.validated = {}
        self.validation_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.service_lock = output_lock(self.store.output, name=".review_server.lock")
        try:
            self.service_lock.__enter__()
        except Exception:
            super().server_close()
            raise
        self.executor = ThreadPoolExecutor(
            max_workers=approve_jobs, thread_name_prefix="review-approve"
        )
        self.export_executor = ThreadPoolExecutor(
            max_workers=export_jobs, thread_name_prefix="review-export"
        )
        self.refresh()

    def server_close(self):
        super().server_close()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            self.export_executor.shutdown(wait=True)
            self.service_lock.__exit__(None, None, None)

    def refresh(self):
        with self.catalog_lock:
            if monotonic() - self.last_refresh < 0.5:
                return
            self.store.refresh()
            for index, path in enumerate(self.store.paths):
                if index in self.known_jobs:
                    continue
                job_path = path.with_name("review_job.json")
                if job_path.exists():
                    job = read_json(job_path)
                    if job.get("running"):
                        job.update(
                            status="interrupted", running=False, error="审核服务已重启，请重试"
                        )
                        atomic_json(job_path, job)
                    with self.job_lock:
                        self.jobs[index] = job
                self.known_jobs.add(index)
            self.last_refresh = monotonic()

    def entry_version(self, index):
        p = self.store.path(index)
        return (
            file_stamp(p),
            file_stamp(p.with_name("human_review.json")),
            file_stamp(p.with_name("approval.json")),
        )

    def list_version(self):
        versions = [self.entry_version(i) for i in range(len(self.store.paths))]
        versions.append(file_stamp(self.store.output / FILENAME))
        return hashlib.sha256(repr(versions).encode()).hexdigest()

    def summary(self, index):
        try:
            state = self.store.load(index)
            if state["exclusion"]:
                return dict(state, approved=False, validation="summary")
            pointer = self.store.path(index).with_name("approval.json")
            bundle = read_json(pointer) if pointer.exists() else {}
            state["approved"] = (
                state["status"] == "ready" and bundle.get("review_revision") == state["revision"]
            )
            if not state["approved"] and state["status"] == "ready":
                state.update(status="pending", pending_reason="请重新通过审核保存依据")
            state["validation"] = "summary"
            return state
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(
                id=index,
                episode=str(self.store.path(index).parent),
                status="failed",
                approved=False,
                unavailable=True,
                error=str(error),
            )

    def state(self, index, *, full=False):
        # Polling checks identities, not large-file hashes. Public mutations always
        # perform the original full validation independently of this display cache.
        state = self.store.load(index)
        if state["exclusion"]:
            return dict(state, approved=False)
        pointer = self.store.path(index).with_name("approval.json")
        bundle = read_json(pointer) if pointer.exists() else {}
        key = (
            self.entry_version(index),
            file_stamp(self.calibration),
            source_stamp(Path(state["episode"])),
            tuple(file_stamp(p) for p in bundle.get("artifacts", {})),
        )
        with self.validation_lock:
            previous = self.validated.get(index)
        if not full:
            if previous and previous[0] == key:
                return deepcopy(previous[1])
            state = self.summary(index)
            state.update(
                approved=False,
                status="pending",
                pending_reason="依据尚未校验或输入已变化，请重新加载本条数据",
            )
            return state
        result = review_state(self.store, index, self.calibration)
        with self.validation_lock:
            self.validated[index] = (key, deepcopy(result))
        return result

    def start_job(self, index, kind, payload=None):
        path = self.store.path(index)
        if kind not in ("approve", "export"):
            raise ValueError("未知任务")
        reservation = reserve(path.parent)
        job = dict(
            index=index, kind=kind, status="queued", running=True, error=None, queued_at=time()
        )
        try:
            rejected = self.store.load(index)
            if rejected["exclusion"]:
                raise ValueError(rejected["pending_reason"])
            with self.job_lock:
                self.jobs[index] = job
                atomic_json(path.with_name("review_job.json"), job)
            executor = self.executor if kind == "approve" else self.export_executor
            executor.submit(self.process_one, index, kind, deepcopy(payload), reservation)
        except Exception:
            reservation.close()
            if self.jobs.get(index) is job:
                self.update_job(index, status="failed", running=False)
            raise
        return dict(running=True, submitted=[index], kind=kind)

    def update_job(self, index, **values):
        with self.job_lock:
            self.jobs[index].update(values)
            atomic_json(self.store.path(index).with_name("review_job.json"), self.jobs[index])

    def process_one(self, index, kind, payload, reservation):
        with adopt(reservation):
            started = perf_counter()
            try:
                self.update_job(
                    index,
                    status="validating",
                    queue_s=round(time() - self.jobs[index]["queued_at"], 3),
                )
                if kind == "approve":
                    approve(self.store, index, payload, self.calibration)
                    self.update_job(index, status="approved")
                else:
                    row = export_approved(
                        self.store,
                        index,
                        self.calibration,
                        progress=lambda status: self.update_job(index, status=status),
                    )
                    self.update_job(index, **{k: v for k, v in row.items() if k != "index"})
            except Exception as error:
                self.update_job(index, status="failed", error=str(error))
            finally:
                try:
                    state = self.state(index, full=True)
                    self.update_job(index, review=state, episode=state["episode"])
                except (ValueError, OSError, KeyError):
                    pass
                self.update_job(index, running=False, elapsed_s=round(perf_counter() - started, 3))

    def save_review(self, index, payload):
        with operation_lock(self.store.path(index).parent):
            self.store.save(index, payload)
            with self.job_lock:
                self.jobs.pop(index, None)
                self.store.path(index).with_name("review_job.json").unlink(missing_ok=True)
            return self.state(index, full=True)

    def reject_episode(self, index, payload):
        with operation_lock(self.store.path(index).parent):
            self.store.reject_episode(index, payload)
            with self.job_lock:
                self.jobs.pop(index, None)
                self.store.path(index).with_name("review_job.json").unlink(missing_ok=True)
            return self.state(index, full=True)

    def axis_preview(self, index, payload):
        angles = axis_angles(payload)
        geometry = {k: payload.get(k) for k in ("side", "frame_index", "quad")}
        state = self.store.load(index)
        episode = Path(state["episode"])
        with baseline_read(episode):
            stamp = source_stamp(episode)
            key = (
                "axis_geometry",
                json.dumps(
                    [state["binding"], stamp, read_json(self.calibration), geometry], sort_keys=True
                ),
            )

            def calculate():
                with self.preview_slots:
                    return preview(episode, payload.get("side"), geometry, self.calibration)

            cached = CACHE.get(key, calculate)
            if source_stamp(episode) != stamp:
                raise ValueError("预览期间源数据发生变化，请重试")
            result = deepcopy(cached)
            result["axis_angles_deg"] = angles.tolist()
            sample = result["calibration_sample"]
            correction = local_rotation(payload)
            sample["rotation_local"] = correction.tolist()
            sample["rotation_camera_corrected"] = (
                np.asarray(sample["rotation_camera_source"]) @ correction
            ).tolist()
            return enrich_pose(result, sample, result["camera_K"])

    def frame_png(self, index, frame):
        episode = Path(self.store.load(index)["episode"])

        def encode():
            with self.frame_slots:
                image, _ = decode_frame(episode, frame)
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("无法编码原帧")
                return encoded.tobytes()

        return cached_file(("frame_png", frame), episode / VIDEO_SUFFIX, encode)


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A newer browser edit superseded this response.

    def authorized(self, write=False):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.send(403, {"error": "仅支持本机访问"})
            return False
        if write and (
            self.headers.get("X-Review-Token") != self.server.token
            or self.headers.get("Content-Type") != "application/json"
        ):
            self.send(403, {"error": "无效的本地审核会话，请重新加载页面"})
            return False
        return True

    def arguments(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        return url.path, query

    def do_GET(self):
        if not self.authorized():
            return
        try:
            path, query = self.arguments()
            self.server.refresh()
            store = self.server.store
            if path in ("/", "/review.js", "/axis_rotation.js", "/review.css"):
                name = "index.html" if path == "/" else path[1:]
                body = (UI / name).read_text().replace("__REVIEW_TOKEN__", self.server.token)
                kind = {
                    "index.html": "text/html",
                    "review.js": "text/javascript",
                    "axis_rotation.js": "text/javascript",
                    "review.css": "text/css",
                }[name]
                return self.send(200, body, kind + "; charset=utf-8")
            if path == "/api/episodes":
                return self.send(
                    200,
                    [self.server.summary(i) for i in range(len(store.paths))],
                )
            if path == "/api/status":
                with self.server.job_lock:
                    jobs = deepcopy(list(self.server.jobs.values()))
                result = dict(
                    jobs=jobs,
                    running=any(j["running"] for j in jobs),
                    list_version=self.server.list_version(),
                    scan=scan_progress(store.output),
                    discovery_errors=store.catalog_errors,
                )
                summary = store.output / "repair_manifest.json"
                result["summary"] = read_json(summary) if summary.exists() else None
                return self.send(200, result)
            index = int(query.get("id", ["-1"])[0])
            if path == "/api/review-state":
                return self.send(200, self.server.state(index))
            if path == "/api/episode":
                state = self.server.state(index, full=True)
                state["default_reviewer"] = getpass.getuser()
                state["candidates"] = store.candidates(index)
                try:
                    image, count = decode_frame(Path(state["episode"]), 0)
                    state.update(frame_count=count, image_size=[image.shape[1], image.shape[0]])
                except ValueError as error:
                    state.update(frame_count=0, video_error=str(error))
                return self.send(200, state)
            if path == "/api/frame":
                return self.send(
                    200, self.server.frame_png(index, int(query["frame"][0])), "image/png"
                )
            return self.send(404, {"error": "未知接口"})
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
            self.send(400, {"error": str(error)})

    def do_POST(self):
        if not self.authorized(write=True):
            return
        try:
            path, query = self.arguments()
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("无效的请求大小")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("请求必须为 JSON 对象")
            if path == "/api/axis-preview":
                return self.send(200, self.server.axis_preview(int(query["id"][0]), payload))
            if path == "/api/confirm-export":
                return self.send(
                    410,
                    {"error": "审核与导出已拆分，请刷新页面，使用「通过审核」或「导出 RRD 文件」"},
                )
            if path == "/api/approve":
                return self.send(
                    202, self.server.start_job(int(query["id"][0]), "approve", payload)
                )
            if path == "/api/reject":
                return self.send(200, self.server.reject_episode(int(query["id"][0]), payload))
            if path == "/api/export":
                return self.send(202, self.server.start_job(int(query["id"][0]), "export"))
            if path == "/api/review":
                result = self.server.save_review(int(query["id"][0]), payload)
                return self.send(200, result)
            return self.send(404, {"error": "未知接口"})
        except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
            self.send(400, {"error": str(error)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=output_directory, required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--initial-mount", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--approve-jobs", type=int, default=4)
    parser.add_argument("--export-jobs", type=int, default=1)
    parser.add_argument("--preview-jobs", type=int, default=8)
    parser.add_argument("--frame-jobs", type=int, default=8)
    parser.add_argument("--cache-mib", type=int, default=2048, help="共享缓存上限（MiB）")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1–65535 之间")
    try:
        server = ReviewServer(
            args.out,
            args.port,
            args.initial_mount,
            approve_jobs=args.approve_jobs,
            export_jobs=args.export_jobs,
            preview_jobs=args.preview_jobs,
            frame_jobs=args.frame_jobs,
            cache_mib=args.cache_mib,
        )
    except OSError as error:
        parser.exit(
            1,
            f"无法监听 127.0.0.1:{args.port}：{error}；端口被占用时请使用 --port 指定其他端口。\n",
        )
    except ValueError as error:
        parser.exit(1, f"review: {error}\n")
    print(f"审核工具：http://127.0.0.1:{server.server_port} （Ctrl+C 退出）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
