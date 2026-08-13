"""Strict raw-to-DIA-cache adapter for reproducible Phase 2 extraction."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from artifact_identity import file_fingerprint
from manager.data_manager import DataManager
from spectrum.dia_data import DIAData
from workflows.flow_utils import get_filename_stem


def _same_source_identity(embedded: dict, current: dict) -> bool:
    return (
        os.path.realpath(embedded["path"])
        == os.path.realpath(current["path"])
        and embedded["sha256"] == current["sha256"]
        and int(embedded["size_bytes"]) == int(current["size_bytes"])
        and int(embedded["mtime_ns"]) == int(current["mtime_ns"])
    )


def _validate_cache(cache_path: Path, raw_path: Path,
                    manager: DataManager, raw_identity: dict | None) -> dict:
    centroid_enabled, centroid_threshold = manager.get_centroid_params()
    DIAData.validate_cache_params(
        str(cache_path),
        expected_centroid_enabled=centroid_enabled,
        expected_centroid_rel_threshold=centroid_threshold,
        expected_source_path=str(raw_path),
        require_source_identity=raw_identity is None,
    )
    embedded = DIAData.read_cache_source_identity(str(cache_path))
    if raw_identity is not None and not _same_source_identity(
            embedded, raw_identity):
        raise ValueError(
            "DIA cache source identity differs from the configured raw file")
    return embedded


def _rebuild_cache(cache_path: Path, raw_path: Path,
                   manager: DataManager, raw_identity: dict) -> None:
    """Build beside the old cache and replace it only after validation."""
    dia = manager.get_dia_data_object(str(raw_path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.staging.", suffix=".npz",
        dir=cache_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        dia.save_to_file(
            str(temporary), source_path=str(raw_path),
            source_fingerprint=raw_identity)
        _validate_cache(temporary, raw_path, manager, raw_identity)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_dia_cache(
    manager: DataManager,
    raw_path: str | Path,
    cache_root: str | Path,
    *,
    dataset: str,
) -> tuple[Path, dict]:
    """Return a strictly source-bound cache and complete provenance.

    Dataset-namespaced cache directories prevent same-basename acquisitions
    from different domains sharing an artifact. Cache-only operation is
    accepted only when the cache embeds the exact configured source path and
    a content digest; legacy caches must be rebuilt while the raw is present.
    """
    raw_path = Path(raw_path).expanduser().resolve()
    cache_directory = Path(cache_root).expanduser().resolve() / str(dataset)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{get_filename_stem(str(raw_path))}.dia.npz"
    raw_identity = file_fingerprint(raw_path) if raw_path.is_file() else None

    embedded = None
    if cache_path.is_file():
        try:
            embedded = _validate_cache(
                cache_path, raw_path, manager, raw_identity)
        except (OSError, ValueError):
            if raw_identity is None:
                # Never delete the only available cache on a failed strict
                # identity check. The user may still inspect/recover it.
                raise
    if embedded is None:
        if raw_identity is None:
            raise FileNotFoundError(
                "raw source is unavailable and no strictly matching DIA "
                f"cache exists: source={raw_path}, cache={cache_path}")
        _rebuild_cache(cache_path, raw_path, manager, raw_identity)
        embedded = _validate_cache(
            cache_path, raw_path, manager, raw_identity)

    provenance = {
        "dataset": str(dataset),
        "kind": "dia_cache",
        "cache": file_fingerprint(cache_path),
        "configured_raw_path": str(raw_path),
        "raw_source_available": raw_identity is not None,
        "embedded_raw_source": embedded,
    }
    if raw_identity is not None:
        provenance["current_raw_source"] = raw_identity
    return cache_path, provenance
