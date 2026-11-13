"""Shared report integrity helpers."""

import hashlib
from pathlib import Path


def digest(path):
    """Hash large reports without reading the complete file into memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()
