"""Auditable image review through the user's logged-in Codex account.

CLI sessions are not API-equivalent: temperature, token cap and enable_thinking
cannot be reproduced. Image bytes and task prompts are preserved.
"""

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from vive_correct.estimate.openai_vision import OpenAIError


class CodexTransport:
    def __init__(self, root, *, model="gpt-6.1-sol", timeout=300):
        self.root = Path(root).resolve()
        self.model = model
        self.timeout = timeout
        self.binary = shutil.which("codex")
        if not self.binary:
            raise ValueError("Codex CLI is unavailable")

    def __call__(self, config, body):
        if config.model != self.model or body["model"] != self.model:
            raise ValueError("Exact Codex model mismatch")
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        folder = self.root / key
        folder.mkdir(parents=True, exist_ok=True)
        texts, images = [], []
        for message in body["messages"]:
            for part in message["content"]:
                if part["type"] == "text":
                    texts.append(part["text"])
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/jpeg;base64,"):
                        raise ValueError("Only in-memory JPEG attachments are supported")
                    path = folder / f"image_{len(images) + 1:02d}.jpg"
                    path.write_bytes(base64.b64decode(url.split(",", 1)[1], validate=True))
                    images.append(path)
                    texts.append(f"[附件图片 {len(images)}：{path.name}]")
                else:
                    raise ValueError("Unsupported content type")
        prompt = (
            "仅根据依次附带的图片完成下列视觉判断。不要使用工具、读取其他文件或联网；"
            "只返回所要求的JSON。附件编号与以下标记一一对应。\n\n" + "\n".join(texts)
        )
        (folder / "prompt.txt").write_text(prompt)
        result = folder / "response.txt"
        result.unlink(missing_ok=True)
        command = [
            self.binary,
            "--no-daemon",
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            self.model,
            "--sandbox",
            "read-only",
            "-C",
            str(folder),
            "--json",
            "-o",
            str(result),
        ]
        for path in images:
            command.extend(["-i", str(path)])
        command.append("-")
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise OpenAIError("Codex model request timed out") from error
        (folder / "events.jsonl").write_text(completed.stdout)
        (folder / "stderr.log").write_text(completed.stderr)
        metadata = dict(
            model=self.model,
            transport="codex_cli_chatgpt_account",
            command=command,
            returncode=completed.returncode,
            elapsed_seconds=time.monotonic() - start,
            images=[
                dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in images
            ],
            task_body_sha256=key,
            unsupported_api_controls=[
                "temperature",
                "max_tokens",
                "enable_thinking",
                "response_format",
            ],
        )
        (folder / "invocation.json").write_text(json.dumps(metadata, indent=2))
        if completed.returncode or not result.exists():
            raise OpenAIError("Codex request failed; see invocation and stderr: " + str(folder))
        for line in completed.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            item_type = event.get("item", {}).get("type", "")
            if item_type in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
                raise OpenAIError("Codex used a tool during isolated image review: " + str(folder))
        return result.read_text()
