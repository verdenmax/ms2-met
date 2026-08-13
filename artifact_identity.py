"""Content identities for immutable scientific inputs and result artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256_file(path: str | os.PathLike) -> str:
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | os.PathLike, *,
                     hash_content: bool = True) -> dict:
    """Return a JSON-safe path/size/time identity, optionally with content."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    result = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = sha256_file(resolved)
    return result
