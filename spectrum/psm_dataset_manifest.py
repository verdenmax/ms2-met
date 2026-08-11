"""Sidecar provenance for custom PSM JSON datasets.

The PSM JSON itself deliberately remains a top-level list for backward
compatibility.  Chemistry and integrity metadata live in a sibling manifest
so consumers can reject a C13/N15 dataset produced under another labeling
assumption before any expensive raw-data work starts.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os

from spectrum.labeling import (
    IDEAL_FULL_LABEL_ISOTOPE_MODEL,
    HeavyType,
    canonical_labeling_name,
    parse_heavy_type,
)


MANIFEST_SCHEMA = "psm_dataset_manifest_v1"
MANIFEST_SUFFIX = ".manifest.json"


def manifest_path(dataset_path: str) -> str:
    """Return the stable sidecar name for a PSM JSON file."""
    return os.fspath(dataset_path) + MANIFEST_SUFFIX


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    dataset_path: str,
    psms: list,
    labeling: HeavyType | str,
    *,
    source_config_path: str | None = None,
) -> dict:
    """Build provenance after ``dataset_path`` has been serialized."""
    canonical = canonical_labeling_name(parse_heavy_type(labeling))
    label_counts = Counter(
        str(getattr(psm, "_label_type", None) or "unknown") for psm in psms)
    modified_counts = Counter(
        str(getattr(psm, "_label_type", None) or "unknown")
        for psm in psms if getattr(psm, "_modify", None))

    source = None
    if source_config_path:
        expanded = os.path.abspath(os.path.expanduser(source_config_path))
        source = {"path": expanded}
        if os.path.isfile(expanded):
            source.update({
                "sha256": _sha256(expanded),
                "size_bytes": os.path.getsize(expanded),
            })

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeling": canonical,
        "isotope_model": IDEAL_FULL_LABEL_ISOTOPE_MODEL,
        "dataset": {
            "path": os.path.abspath(os.path.expanduser(dataset_path)),
            "sha256": _sha256(dataset_path),
            "size_bytes": os.path.getsize(dataset_path),
            "n_psms": len(psms),
            "n_modified_psms": sum(modified_counts.values()),
            "counts_by_label_type": dict(sorted(label_counts.items())),
            "modified_counts_by_label_type": dict(
                sorted(modified_counts.items())),
        },
    }
    if source is not None:
        manifest["source_config"] = source
    return manifest


def write_manifest(
    dataset_path: str,
    psms: list,
    labeling: HeavyType | str,
    *,
    source_config_path: str | None = None,
) -> str:
    """Write and return the JSON sidecar path."""
    output = manifest_path(dataset_path)
    manifest = build_manifest(
        dataset_path, psms, labeling,
        source_config_path=source_config_path,
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logging.info("已写入 PSM 数据集 manifest: %s", output)
    return output


def validate_manifest(
    dataset_path: str,
    expected_labeling: HeavyType | str,
    *,
    require: bool,
) -> dict | None:
    """Validate schema, chemistry, and dataset digest.

    ``require=False`` keeps old SILAC JSON files usable with a warning.  The
    uniform-label paths require the sidecar because using the wrong chemistry
    would shift every precursor and fragment mass.
    """
    sidecar = manifest_path(dataset_path)
    if not os.path.isfile(sidecar):
        message = (
            f"PSM 数据集缺少 manifest: {sidecar}; 请用当前 "
            "tools/extract_common.py 重新生成")
        if require:
            raise ValueError(message)
        logging.warning("%s；按 legacy SILAC JSON 继续", message)
        return None

    with open(sidecar, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"不支持的 PSM manifest schema: {manifest.get('schema')!r}")

    expected = canonical_labeling_name(parse_heavy_type(expected_labeling))
    observed = manifest.get("labeling")
    if observed != expected:
        raise ValueError(
            "PSM 数据集标记类型与特征配置不一致: "
            f"manifest={observed!r}, [general] labeling={expected!r}")

    dataset = manifest.get("dataset", {})
    expected_digest = dataset.get("sha256")
    if not expected_digest:
        raise ValueError("PSM manifest 缺少 dataset.sha256")
    observed_digest = _sha256(dataset_path)
    if observed_digest != expected_digest:
        raise ValueError(
            "PSM JSON 与 manifest 摘要不一致，文件可能已被修改或配错: "
            f"{dataset_path}")
    return manifest
