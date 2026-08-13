"""Strict raw-to-DIA-cache adapter for reproducible Phase 2 extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
from zipfile import BadZipFile

import numpy as np

from artifact_identity import file_fingerprint, sha256_file
from manager.data_manager import DataManager
from spectrum.dia_data import DIAData, _load_attrs
from workflows.flow_utils import get_filename_stem


MMAP_CACHE_SCHEMA = "phase2_dia_npy_mmap_cache_v1"


def _mmap_root(npz_path: Path) -> Path:
    name = npz_path.name[:-4] if npz_path.name.endswith(".npz") \
        else npz_path.name
    return npz_path.with_name(f"{name}.mmap-v1")


def _mmap_manifest(root: Path) -> dict:
    complete_path = root / "COMPLETE"
    manifest_path = root / "manifest.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"incomplete Phase 2 DIA mmap cache: {root}")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != MMAP_CACHE_SCHEMA or \
            complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"invalid Phase 2 DIA mmap cache marker: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MMAP_CACHE_SCHEMA:
        raise ValueError(f"unsupported Phase 2 DIA mmap cache: {root}")
    for name, declaration in manifest.get("arrays", {}).items():
        path = root / f"{name}.npy"
        if not path.is_file() or path.stat().st_size != declaration["size_bytes"]:
            raise ValueError(
                f"Phase 2 DIA mmap cache array changed: {path}")
    return manifest


def _build_mmap_cache(npz_path: Path, output: Path,
                      npz_identity: dict) -> None:
    staging = output.with_name(f".{output.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        arrays = {}
        with np.load(npz_path, allow_pickle=False) as archive:
            for name in archive.files:
                path = staging / f"{name}.npy"
                # np.save streams one decompressed member at a time. The final
                # training extraction then maps these .npy files without
                # materializing every DIA array in RAM together.
                np.save(path, archive[name], allow_pickle=False)
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                arrays[name] = {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "size_bytes": path.stat().st_size,
                }
                del value
        manifest = {
            "schema": MMAP_CACHE_SCHEMA,
            "source_npz": npz_identity,
            "arrays": arrays,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (staging / "COMPLETE").write_text(json.dumps({
            "schema": MMAP_CACHE_SCHEMA,
            "manifest_sha256": sha256_file(manifest_path),
        }, sort_keys=True) + "\n", encoding="utf-8")
        backup = None
        if output.exists():
            backup = output.with_name(
                f".{output.name}.backup.{uuid.uuid4().hex}")
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def resolve_mmap_dia_cache(
    npz_path: str | Path,
    *,
    npz_identity: dict | None = None,
) -> tuple[Path, dict]:
    """Return a source-bound directory of genuinely mmap-able DIA arrays."""
    npz_path = Path(npz_path).resolve()
    current = npz_identity or file_fingerprint(npz_path)
    if Path(current.get("path", "")).resolve() != npz_path or \
            not current.get("sha256"):
        raise ValueError("DIA NPZ fingerprint does not match mmap source")
    stat = npz_path.stat()
    if int(current.get("size_bytes", -1)) != stat.st_size or \
            int(current.get("mtime_ns", -1)) != stat.st_mtime_ns:
        raise ValueError("DIA NPZ fingerprint is stale")
    output = _mmap_root(npz_path)
    rebuild = True
    if output.is_dir():
        try:
            manifest = _mmap_manifest(output)
            rebuild = manifest.get("source_npz") != current
        except (OSError, ValueError, json.JSONDecodeError):
            rebuild = True
    if rebuild:
        _build_mmap_cache(npz_path, output, current)
    manifest = _mmap_manifest(output)
    if manifest.get("source_npz") != current:
        raise ValueError("Phase 2 DIA mmap cache source identity mismatch")
    return output, {
        "kind": "dia_mmap_cache",
        "path": str(output),
        "schema": MMAP_CACHE_SCHEMA,
        "source_npz": current,
        "manifest_sha256": sha256_file(output / "manifest.json"),
    }


def load_mmap_dia_cache(root: str | Path) -> DIAData:
    """Open a Phase 2 DIA cache while leaving large arrays as memmaps."""
    root = Path(root).resolve()
    manifest = _mmap_manifest(root)
    arrays = {
        name: np.load(
            root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in manifest["arrays"]
    }
    DIAData._check_format_version(str(root), arrays)
    result = DIAData()
    _load_attrs(result, arrays)
    # Keep the mapping alive explicitly for clarity and future non-NumPy
    # backends, although each np.memmap also owns its mapping handle.
    result._phase2_mmap_arrays = arrays
    return result


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
        except (OSError, ValueError, EOFError, BadZipFile):
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
