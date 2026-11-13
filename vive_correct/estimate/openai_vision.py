"""OpenAI-compatible vision call for the two Vive stickers.

The classical detector does not use a neural model. This module is used only
when --detector ai is explicitly selected and an API key is configured.
The default model name is gpt-4o and the
default base URL is the official OpenAI v1 endpoint. Any server that accepts
`POST /chat/completions` with a Bearer key and an image_url content part can
be selected with base_url and model.

Key lookup order: --api-key, OPENAI_API_KEY, vive_correct/.env or the repo
.env, then vive_correct/openai.local.json. The key is never written to disk
by this module.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parent
DEFAULT_MODEL = "gpt-4o"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_CONFIG = PACKAGE / "openai.local.json"


class OpenAIError(RuntimeError):
    pass


@dataclass
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str
    key_source: str

    def __repr__(self) -> str:
        return (
            f"OpenAIConfig(model={self.model!r}, base_url={self.base_url!r}, "
            f"key_source={self.key_source!r})"
        )


def chat_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _dotenv(paths: list[Path]) -> dict:
    found = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            found.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return found


def _local_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenAIError(f"{path} must be a JSON object")
    return data


def _pick(cli, env_value, dotenv_value, file_value, default):
    for value, source in (
        (cli, "cli"),
        (env_value, "env"),
        (dotenv_value, "dotenv"),
        (file_value, "openai.local.json"),
    ):
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, source
    if default is None:
        return None, None
    return default, "default"


def load_config(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    environ: dict | None = None,
    config_path: Path | None = None,
    dotenv_paths: list[Path] | None = None,
) -> OpenAIConfig | None:
    """Return a config when a key exists. Otherwise return None."""
    env = os.environ if environ is None else environ
    path = LOCAL_CONFIG if config_path is None else Path(config_path)
    if dotenv_paths is None:
        dotenv_paths = [PACKAGE / ".env", REPO / ".env"]
    local = _local_file(path)
    dots = _dotenv(list(dotenv_paths))
    key, key_source = _pick(
        api_key,
        env.get("OPENAI_API_KEY"),
        dots.get("OPENAI_API_KEY"),
        local.get("api_key") or local.get("openai_api_key"),
        None,
    )
    if not key:
        return None
    resolved_base, _ = _pick(
        base_url,
        env.get("OPENAI_BASE_URL"),
        dots.get("OPENAI_BASE_URL"),
        local.get("base_url"),
        DEFAULT_BASE_URL,
    )
    resolved_model, _ = _pick(
        model,
        env.get("OPENAI_MODEL"),
        dots.get("OPENAI_MODEL"),
        local.get("model"),
        DEFAULT_MODEL,
    )
    return OpenAIConfig(key, resolved_base, resolved_model, key_source)


def _message_text(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAIError(f"unexpected response: {str(payload)[:400]}") from exc
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        text = "\n".join(parts).strip()
    else:
        text = ""
    if not text:
        raise OpenAIError("model returned empty content")
    return text


def _post_chat(config: OpenAIConfig, body: dict, timeout: float = 180) -> str:
    url = chat_url(config.base_url)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise OpenAIError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIError(f"cannot reach {url}: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"endpoint did not return JSON: {raw[:300]}") from exc
    return _message_text(payload)


def complete(config: OpenAIConfig, body: dict) -> str:
    try:
        return _post_chat(config, body)
    except OpenAIError as exc:
        if not str(exc).startswith("HTTP 400"):
            raise
        slim = {
            key: value
            for key, value in body.items()
            if key not in ("temperature", "response_format", "max_tokens")
        }
        return _post_chat(config, slim)


def extract_json(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise OpenAIError("model did not return JSON")
        try:
            data = json.loads(body[start : end + 1])
        except json.JSONDecodeError as exc:
            raise OpenAIError(f"model JSON could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenAIError("model JSON must be an object")
    return data


def _order_corners(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angle = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angle)]


def _parse_quad(value, width: int, height: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 4:
        raise OpenAIError("quad must be four [x, y] corners")
    points = []
    for corner in value:
        if not isinstance(corner, (list, tuple)) or len(corner) != 2:
            raise OpenAIError("each corner must be [x, y]")
        x, y = float(corner[0]), float(corner[1])
        if not np.isfinite(x) or not np.isfinite(y):
            raise OpenAIError("corner is not finite")
        if not (-20 <= x <= width + 20 and -20 <= y <= height + 20):
            raise OpenAIError(f"corner ({x:.1f}, {y:.1f}) is outside the image")
        points.append((x, y))
    quad = _order_corners(np.asarray(points, float))
    area = abs(
        0.5 * np.sum(quad[:, 0] * np.roll(quad[:, 1], -1) - quad[:, 1] * np.roll(quad[:, 0], -1))
    )
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    if area < 120 or area > 15000:
        raise OpenAIError(f"quad area {area:.0f}px is not a sticker")
    if min(edges) < 8:
        raise OpenAIError("quad is too thin to be a sticker")
    return quad


def parse_sticker_response(text: str, width: int, height: int, rois: dict) -> dict:
    data = extract_json(text)
    sides = {}
    errors = {}
    for side in ("left", "right"):
        node = data.get(side) or {}
        if not isinstance(node, dict):
            errors[side] = "response for this side is not an object"
            sides[side] = {"visible": False, "quad": None}
            continue
        visible = bool(node.get("visible")) and node.get("quad") not in (None, [])
        if not visible:
            sides[side] = {"visible": False, "quad": None}
            continue
        try:
            quad = _parse_quad(node.get("quad"), width, height)
        except (OpenAIError, TypeError, ValueError) as exc:
            errors[side] = str(exc)
            sides[side] = {"visible": False, "quad": None}
            continue
        center = quad.mean(axis=0)
        if side in rois:
            x0, y0, x1, y1 = rois[side]
            pad = 80
            inside = (x0 - pad) <= center[0] <= (x1 + pad) and (y0 - pad) <= center[1] <= (y1 + pad)
            if not inside:
                errors[side] = "quad center is outside that hand region"
                sides[side] = {"visible": False, "quad": None}
                continue
        sides[side] = {"visible": True, "quad": quad}
    left = sides["left"]
    right = sides["right"]
    if left["visible"] and right["visible"]:
        if left["quad"][:, 0].mean() >= right["quad"][:, 0].mean():
            errors["order"] = "left quad is not to the left of the right quad"
            sides["left"] = {"visible": False, "quad": None}
            sides["right"] = {"visible": False, "quad": None}
    return {"sides": sides, "errors": errors, "notes": str(data.get("notes") or "")}


def quad_mask(quad: np.ndarray, shape) -> np.ndarray:
    height, width = shape[:2]
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(map(float, point)) for point in quad], fill=1)
    return np.asarray(canvas, dtype=bool)


def _prompt(width: int, height: int, rois: dict) -> str:
    def box(side):
        if side not in rois:
            return "未知"
        x0, y0, x1, y1 = rois[side]
        return f"[{int(x0)}, {int(y0)}, {int(x1)}, {int(y1)}]"

    return f"""这是一张已经去畸变的头带左目画面，第一人称。图像左侧是操作者的左手，右侧是右手。每只手的黑色 Vive Tracker 顶面贴有一张白色矩形标签，实物约 5 cm × 3 cm。

只标出这张白色标签的四个角。不要标腕带上的字母、白手套、桌面上的纸或屏幕。标签侧面朝向相机、看不见正面时，visible 设为 false，quad 设为 null。

像素原点在图像左上角，x 向右，y 向下。quad 是四个角，顺序不限。
图像尺寸：{width}×{height}
左手搜索区域 [x0, y0, x1, y1]：{box("left")}
右手搜索区域 [x0, y0, x1, y1]：{box("right")}

只返回 JSON，不要其他文字：
{{"left": {{"visible": true, "quad": [[x, y], [x, y], [x, y], [x, y]]}}, "right": {{"visible": false, "quad": null}}, "notes": "一句话"}}
"""


def _jpeg_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image), "RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def locate_stickers(image: np.ndarray, rois: dict, config: OpenAIConfig, transport=None) -> dict:
    """Ask the vision model for the two sticker quads. transport(config, body) -> text, for tests."""
    height, width = image.shape[:2]
    body = {
        "model": config.model,
        "temperature": 0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _prompt(width, height, rois)},
                    {"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}},
                ],
            }
        ],
    }
    print(f"calling {config.model} at {chat_url(config.base_url)}", flush=True)
    content = (transport or complete)(config, body)
    parsed = parse_sticker_response(content, width, height, rois)
    parsed["raw"] = content
    return parsed


def main():
    config = load_config()
    print(f"default model: {DEFAULT_MODEL}")
    print(f"default base_url: {DEFAULT_BASE_URL}")
    if config is None:
        print("api key: not set; sticker_pipeline stays on the classical detector")
        return
    print(f"api key: set via {config.key_source}")
    print("activation: --detector ai --frame <index> (CV is the pipeline default)")
    print(f"model: {config.model}")
    print(f"base_url: {config.base_url}")


if __name__ == "__main__":
    main()
