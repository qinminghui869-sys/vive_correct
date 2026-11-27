"""Check repository Markdown links offline; section links use GitHub-compatible named anchors."""

from collections import Counter
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>]*>|[^\s)]*)(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
ANCHOR = re.compile(r'<a\s+name="([^"]+)"\s*>\s*</a>')


def prose(text):
    """Ignore fenced examples, while preserving line numbers for diagnostics."""
    result, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker[1]
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            result.append("\n")
        elif fence:
            result.append("\n")
        else:
            result.append(re.sub(r"(`+).*?\1", "", line))
    return "".join(result)


def check(root, tracked):
    documents = sorted(p for p in tracked if p.endswith(".md"))
    bodies = {p: prose((root / p).read_text(encoding="utf-8")) for p in documents}
    anchors = {p: Counter(ANCHOR.findall(body)) for p, body in bodies.items()}
    errors, count = [], 0
    for name, body in bodies.items():
        for match in re.finditer(r'<a\s+id="([^"\n]+)"', body):
            errors.append(f"{name}: use a named anchor for GitHub compatibility: #{match[1]}")
        for anchor, occurrences in anchors[name].items():
            if occurrences > 1:
                errors.append(f"{name}: duplicate anchor #{anchor}")
        for match in [*LINK.finditer(body), *REFERENCE.finditer(body)]:
            count += 1
            value = match[1].strip("<>")
            location = f"{name}:{body.count(chr(10), 0, match.start()) + 1}"
            if not value:
                errors.append(f"{location}: empty link")
                continue
            url = urlsplit(value)
            if url.scheme or url.netloc:
                continue  # External URLs and local service URLs are not fetched by CI.
            target = (root / name).parent / unquote(url.path) if url.path else root / name
            try:
                relative = target.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                errors.append(f"{location}: link escapes repository: {value}")
                continue
            if relative not in tracked or not target.is_file():
                errors.append(f"{location}: target is missing or untracked: {value}")
            elif url.fragment and unquote(url.fragment) not in anchors.get(relative, {}):
                errors.append(f"{location}: missing explicit anchor: {value}")
    return documents, count, errors


def main():
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    documents, count, errors = check(ROOT, tracked)
    for error in errors:
        print(error)
    print(f"Documentation: {len(documents)} files, {count} links, {len(errors)} errors")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
