"""Single-user browser review on a loopback-only HTTP server."""

import argparse
import getpass
import json
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import cv2

from vive_correct.human_review import ReviewStore, decode_frame, output_directory
from vive_correct.rerun_export import DEFAULT_MOUNT
from vive_correct.review_actions import export_approved
from vive_correct.review_bundle import approve, review_state
from vive_correct.review_io import read_json

DEFAULT_PORT = 18763
UI = Path(__file__).with_name("review_ui")


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, output, port=DEFAULT_PORT, calibration=DEFAULT_MOUNT):
        self.store = ReviewStore(output)
        self.calibration = Path(calibration)
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.job_lock = threading.Lock()
        self.episode_locks = [threading.Lock() for _ in self.store.paths]
        super().__init__(("127.0.0.1", port), ReviewHandler)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="review-export")

    def server_close(self):
        super().server_close()
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def start_job(self, index, kind, payload=None):
        self.store.path(index)
        if kind not in ("approve", "export"):
            raise ValueError("未知任务")
        if not self.episode_locks[index].acquire(blocking=False):
            raise ValueError("这条数据正在处理；可继续审核其他数据")
        job = dict(index=index, kind=kind, status="queued", running=True, error=None)
        with self.job_lock:
            self.jobs[index] = job
        try:
            self.executor.submit(self.process_one, index, kind, deepcopy(payload))
        except Exception:
            self.episode_locks[index].release()
            self.update_job(index, status="failed", running=False)
            raise
        return dict(running=True, submitted=[index], kind=kind)

    def update_job(self, index, **values):
        with self.job_lock:
            self.jobs[index].update(values)

    def process_one(self, index, kind, payload):
        started = perf_counter()
        try:
            self.update_job(index, status="validating")
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
                state = review_state(self.store, index, self.calibration)
                self.update_job(index, review=state, episode=state["episode"])
            except (ValueError, OSError, KeyError):
                pass
            self.update_job(index, running=False, elapsed_s=round(perf_counter() - started, 3))
            self.episode_locks[index].release()

    def save_review(self, index, payload):
        self.store.path(index)
        lock = self.episode_locks[index]
        if not lock.acquire(blocking=False):
            raise ValueError("这条数据正在处理；可继续审核其他数据")
        try:
            self.store.save(index, payload)
            with self.job_lock:
                self.jobs.pop(index, None)
            return review_state(self.store, index, self.calibration)
        finally:
            lock.release()


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
        self.wfile.write(body)

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
            store = self.server.store
            if path in ("/", "/review.js", "/review.css"):
                name = "index.html" if path == "/" else path[1:]
                body = (UI / name).read_text().replace("__REVIEW_TOKEN__", self.server.token)
                kind = {
                    "index.html": "text/html",
                    "review.js": "text/javascript",
                    "review.css": "text/css",
                }[name]
                return self.send(200, body, kind + "; charset=utf-8")
            if path == "/api/episodes":
                return self.send(
                    200,
                    [
                        review_state(store, i, self.server.calibration)
                        for i in range(len(store.paths))
                    ],
                )
            if path == "/api/status":
                with self.server.job_lock:
                    jobs = deepcopy(list(self.server.jobs.values()))
                result = dict(jobs=jobs, running=any(j["running"] for j in jobs))
                summary = store.output / "repair_manifest.json"
                result["summary"] = read_json(summary) if summary.exists() else None
                return self.send(200, result)
            index = int(query.get("id", ["-1"])[0])
            if path == "/api/review-state":
                return self.send(200, review_state(store, index, self.server.calibration))
            if path == "/api/episode":
                state = review_state(store, index, self.server.calibration)
                state["default_reviewer"] = getpass.getuser()
                state["candidates"] = store.candidates(index)
                try:
                    image, count = decode_frame(Path(state["episode"]), 0)
                    state.update(frame_count=count, image_size=[image.shape[1], image.shape[0]])
                except ValueError as error:
                    state.update(frame_count=0, video_error=str(error))
                return self.send(200, state)
            if path == "/api/frame":
                state = store.load(index)
                image, _ = decode_frame(Path(state["episode"]), int(query["frame"][0]))
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise ValueError("无法编码原帧")
                return self.send(200, encoded.tobytes(), "image/png")
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
            if path == "/api/confirm-export":
                return self.send(
                    410,
                    {"error": "审核与导出已拆分，请刷新页面，使用「通过审核」或「导出 RRD 文件」"},
                )
            if path == "/api/approve":
                return self.send(
                    202, self.server.start_job(int(query["id"][0]), "approve", payload)
                )
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
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1–65535 之间")
    try:
        server = ReviewServer(args.out, args.port, args.initial_mount)
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
