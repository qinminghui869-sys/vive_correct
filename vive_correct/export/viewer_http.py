"""Private local-file transport for viewers that cannot allocate inotify watches."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
from threading import Thread
from urllib.parse import quote, urlsplit


class RecordingHandler(SimpleHTTPRequestHandler):
    """Expose exactly the selected recordings; never serve directory listings."""

    def __init__(self, *args, recordings, **kwargs):
        self.recordings = recordings
        super().__init__(*args, **kwargs)

    def send_head(self):
        if urlsplit(self.path).path not in self.recordings:
            self.send_error(404)
            return None
        return super().send_head()

    def translate_path(self, path):
        return str(self.recordings[urlsplit(path).path])


def serve_and_launch(command, recordings):
    token = secrets.token_urlsafe(24)
    files = {
        f"/{token}/{index}/{quote(Path(path).name)}": Path(path).resolve()
        for index, path in enumerate(recordings)
    }
    if not files or any(not path.is_file() for path in files.values()):
        raise ValueError("Selected recordings are missing")
    handler = partial(RecordingHandler, recordings=files)
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        urls = [f"http://127.0.0.1:{server.server_port}{path}" for path in files]
        try:
            return subprocess.run([*command, *urls], stdin=subprocess.DEVNULL).returncode
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    raise SystemExit(serve_and_launch(json.loads(sys.argv[1]), sys.argv[2:]))
