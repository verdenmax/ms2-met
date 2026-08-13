"""Build the audited Phase 2 raw-XIC integrity pilot.

The frozen LightGBM bundle remains the sole owner of sample membership and
fold assignments.  This builder joins legacy PSM JSON rows to those IDs,
extracts raw signals, verifies selected legacy features from the saved tensor
contract, and publishes only when every identity and parity check passes.
"""

from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import pandas as pd
import yaml

from constant.keys import ConfigKeys
from manager.data_manager import DataManager
from spectrum.dia_data import DIAData
from spectrum.labeling import (
    HeavyType, canonical_labeling_name, parse_heavy_type,
)
from spectrum.psm_dataset_manifest import validate_manifest
from spectrum.psm_info import PSMInfo
from workflows.flow_utils import data_to_npz, get_filename_stem
from workflows.modified_psm_policy import apply_modified_psm_policy
from workflows.pred_store import build_pred_store, normalize_key

from tools.deep_trainer.spec_adapter import prepare_protocol

from .extraction import extract_signal_sample
from .matching import match_psms_to_protocol, select_pilot_rows
from .parity import compare_to_feature_row
from .schema import ExtractionSettings
from .store import write_signal_dataset


BUILD_CONFIG_SCHEMA = "phase2_xic_pilot_config_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path, *, hash_content: bool = True) -> dict:
    path = path.resolve()
    stat = path.stat()
    result = {
        "path": str(path), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        result["sha256"] = _sha256(path)
    return result


def _resolve(value: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_ini(path: Path) -> configparser.ConfigParser:
    if not path.is_file():
        raise FileNotFoundError(f"missing extraction config: {path}")
    config = configparser.ConfigParser()
    with path.open(encoding="utf-8") as handle:
        config.read_file(handle)
    if not config.has_section(ConfigKeys.INPUT):
        raise ValueError(f"extraction config lacks [input]: {path}")
    if not config.has_section(ConfigKeys.GENERAL):
        raise ValueError(f"extraction config lacks [general]: {path}")
    if config.getint(
            ConfigKeys.GENERAL, ConfigKeys.FEATURE_TYPE, fallback=0) != 0:
        raise ValueError("Phase 2 pilot requires feature_type=0 PSM extraction")
    return config


def _load_psms(config: configparser.ConfigParser, *, dataset: str,
               audit_documents: dict) -> tuple[list[PSMInfo], Path, HeavyType]:
    if config.getint(
            ConfigKeys.INPUT, ConfigKeys.SEARCH_ENGINE_TYPE, fallback=1) != 0:
        raise ValueError(
            f"dataset {dataset}: Phase 2 pilot requires custom PSM JSON "
            "(search_engine_type=0) so modifications are recoverable")
    path = _resolve(
        config[ConfigKeys.INPUT][ConfigKeys.LIGHT_RESULT_PATH],
        base=PROJECT_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"dataset {dataset}: missing PSM JSON: {path}")
    labeling = parse_heavy_type(config.get(
        ConfigKeys.GENERAL, ConfigKeys.LABELING, fallback="silac"))
    manifest = validate_manifest(
        str(path), labeling,
        require=labeling in (HeavyType.C13, HeavyType.N15))
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"dataset {dataset}: PSM JSON must be a top-level list")
    psms = [PSMInfo.from_dict(item) for item in payload]
    policy = config.get(
        ConfigKeys.GENERAL, ConfigKeys.MODIFIED_PSM_POLICY,
        fallback="reject")
    with tempfile.TemporaryDirectory(prefix="phase2-modified-audit-") as temp:
        audit_prefix = Path(temp) / "modified_psms"
        psms, audit = apply_modified_psm_policy(
            psms, labeling, policy, result_file=str(audit_prefix))
    audit_documents[f"modified_psms_{dataset}.json"] = audit
    if manifest is not None:
        audit_documents[f"psm_manifest_{dataset}.json"] = manifest
    return psms, path, labeling


def _raw_paths(config: configparser.ConfigParser) -> dict[str, Path]:
    count = config.getint(ConfigKeys.INPUT, ConfigKeys.RAW_NUM, fallback=1)
    result = {}
    for index in range(1, count + 1):
        key = f"{ConfigKeys.RAW_PATH}_{index}"
        if not config.has_option(ConfigKeys.INPUT, key):
            raise ValueError(f"extraction config lacks [input] {key}")
        path = _resolve(config[ConfigKeys.INPUT][key], base=PROJECT_ROOT)
        title = get_filename_stem(str(path))
        if title in result:
            raise ValueError(f"duplicate raw title in extraction config: {title}")
        result[title] = path
    return result


def _load_predictions(config: configparser.ConfigParser,
                      psms: list[PSMInfo], enabled: bool):
    if not enabled:
        return None
    required = (
        ConfigKeys.SPECLIB_DIR, ConfigKeys.SPECLIB_FASTA,
        ConfigKeys.SPECLIB_MOD,
    )
    if not config.has_section(ConfigKeys.SPECLIB):
        raise ValueError("prediction.include=true but [speclib] is absent")
    missing = [key for key in required
               if not config.has_option(ConfigKeys.SPECLIB, key)]
    if missing:
        raise ValueError(f"[speclib] lacks required keys: {missing}")
    from spectrum.speclib import SpecLib
    paths = {
        key: _resolve(config[ConfigKeys.SPECLIB][key], base=PROJECT_ROOT)
        for key in required
    }
    library = SpecLib.open_dir(
        str(paths[ConfigKeys.SPECLIB_DIR]),
        fasta_path=str(paths[ConfigKeys.SPECLIB_FASTA]),
        mod_path=str(paths[ConfigKeys.SPECLIB_MOD]),
    )
    wanted = {
        normalize_key(psm._sequence, psm._modify, psm._charge)
        for psm in psms
    }
    return build_pred_store(library, wanted)


def _source_row_metadata(row, psm: PSMInfo) -> dict:
    keep = (
        "sample_id", "source_sample_id", "dataset", "label",
        "negative_tier", "fixed_split", "outer_fold", "sequence", "charge",
        "precursor_mz", "rt", "raw_title1", "label_type", "q_value",
        "protein_names", "leakage_group_id", "group_id", "pair_id",
        "query_id", "parent_id", "candidate_family_id", "peptide_group_id",
    )
    metadata = {
        column: row[column] for column in keep
        if column in row.index and not pd.isna(row[column])
    }
    metadata["label"] = int(row["label"])
    metadata["charge"] = int(row["charge"])
    metadata["rt"] = float(row["rt"])
    metadata["precursor_mz"] = float(row["precursor_mz"])
    metadata["modifications_json"] = json.dumps(
        [[int(position), int(unimod)] for position, unimod in psm._modify],
        separators=(",", ":"))
    return metadata


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _assert_extraction_settings(config: configparser.ConfigParser,
                                settings: ExtractionSettings,
                                dataset: str) -> None:
    observed_window = config.getint(
        ConfigKeys.GENERAL, ConfigKeys.XIC_CYCLE_WINDOW)
    observed_ppm = config.getfloat(
        ConfigKeys.GENERAL, ConfigKeys.MASS_TOL_PPM)
    if observed_window != settings.xic_cycle_window:
        raise ValueError(
            f"dataset {dataset}: extraction config xic_cycle_window="
            f"{observed_window}, Phase 2 requires {settings.xic_cycle_window}")
    if observed_ppm != settings.mass_tol_ppm:
        raise ValueError(
            f"dataset {dataset}: extraction config mass_tol_ppm="
            f"{observed_ppm}, Phase 2 requires {settings.mass_tol_ppm}")


def build_signal_dataset(
    config_path: str,
    split_config_path: str,
    feature_root: str,
    protocol_root: str,
    output_root: str,
    *,
    cache_root: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Build and publish the balanced 1200-row Phase 2 integrity pilot."""
    config_path_obj = Path(config_path).resolve()
    with config_path_obj.open(encoding="utf-8") as handle:
        build_config = yaml.safe_load(handle)
    if build_config.get("schema") != BUILD_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported Phase 2 build config: {build_config.get('schema')!r}")

    extraction = build_config.get("extraction", {})
    settings = ExtractionSettings(
        xic_cycle_window=int(extraction.get("xic_cycle_window", 6)),
        mass_tol_ppm=float(extraction.get("mass_tol_ppm", 10.0)),
        fragment_charges=tuple(extraction.get("fragment_charges", [1, 2])),
    )
    pilot = build_config.get("pilot", {})
    correct_per_dataset = int(pilot.get("correct_per_dataset", 200))
    error_per_dataset = int(pilot.get("error_per_dataset", 200))
    pilot_seed = int(pilot.get("seed", 20260813))
    shard_size = int(build_config.get("storage", {}).get("shard_size", 256))
    prediction_enabled = bool(
        build_config.get("prediction", {}).get("include", False))

    protocol = prepare_protocol(
        split_config_path, feature_root, "combined", protocol_root)
    selected = select_pilot_rows(
        protocol.frame, correct_per_dataset=correct_per_dataset,
        error_per_dataset=error_per_dataset, seed=pilot_seed)
    expected_count = len(selected)
    datasets = build_config.get("datasets", {})
    expected_datasets = set(selected[protocol.dataset_col].astype(str))
    if set(datasets) != expected_datasets:
        raise ValueError(
            f"build config datasets={sorted(datasets)}, frozen protocol="
            f"{sorted(expected_datasets)}")

    cache_root_path = _resolve(
        cache_root or build_config.get("cache_root", "workspace"),
        base=PROJECT_ROOT)
    cache_root_path.mkdir(parents=True, exist_ok=True)
    audit_documents: dict[str, dict] = {}
    matching_tables = []
    samples = []
    parity_rows = []
    sources = []
    chemistry = {}

    for dataset in sorted(datasets):
        declaration = datasets[dataset]
        relative_config = declaration.get("extraction_config")
        if not relative_config:
            raise ValueError(
                f"dataset {dataset} lacks extraction_config")
        ini_path = _resolve(relative_config, base=Path(feature_root).resolve())
        ini = _load_ini(ini_path)
        _assert_extraction_settings(ini, settings, dataset)
        psms, psm_path, labeling = _load_psms(
            ini, dataset=dataset, audit_documents=audit_documents)
        chemistry[dataset] = canonical_labeling_name(labeling)
        domain_rows = selected[
            selected[protocol.dataset_col].astype(str).eq(dataset)
        ].copy()
        matched, match_audit = match_psms_to_protocol(
            domain_rows, psms, protocol.identity_cols)
        matching_tables.append(match_audit)
        bad = match_audit[~match_audit["status"].eq("matched")]
        if len(bad):
            failure = Path(output_root).resolve().with_suffix(
                ".identity_failure.csv")
            failure.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(matching_tables, ignore_index=True).to_csv(
                failure, index=False)
            raise ValueError(
                f"dataset {dataset}: {len(bad)} pilot PSM identities did not "
                f"match uniquely; audit={failure}")

        selected_psms = [matched[str(sample_id)]
                         for sample_id in domain_rows["sample_id"]]
        predictions = _load_predictions(
            ini, selected_psms, prediction_enabled)
        raw_paths = _raw_paths(ini)
        rows_by_raw = {}
        for _, row in domain_rows.iterrows():
            psm = matched[str(row["sample_id"])]
            rows_by_raw.setdefault(psm._raw_title, []).append((row, psm))
        unknown = sorted(set(rows_by_raw) - set(raw_paths))
        if unknown:
            raise ValueError(
                f"dataset {dataset}: PSM raw titles absent from extraction "
                f"config: {unknown[:10]}")

        manager = DataManager(ini)
        for raw_title in sorted(rows_by_raw):
            raw_path = raw_paths[raw_title]
            cache_path = cache_root_path / f"{raw_title}.dia.npz"
            if not raw_path.is_file() and not cache_path.is_file():
                raise FileNotFoundError(
                    f"raw source and DIA cache are both missing for "
                    f"{raw_title}: source={raw_path}, cache={cache_path}")
            _, shared_path = data_to_npz(
                manager, str(raw_path), str(cache_root_path))
            dia = DIAData.load_from_file(shared_path, use_mmap=True)
            for row, psm in rows_by_raw[raw_title]:
                pred_frags = None
                if predictions is not None:
                    record = predictions.get(normalize_key(
                        psm._sequence, psm._modify, psm._charge))
                    pred_frags = record["frags"] if record else None
                sample = extract_signal_sample(
                    psm, dia, settings, _source_row_metadata(row, psm),
                    labeling=labeling, pred_frags=pred_frags)
                samples.append(sample)
                parity_rows.extend(compare_to_feature_row(
                    sample, row, settings))

        sources.extend([
            {"dataset": dataset, "kind": "extraction_config",
             **_fingerprint(ini_path)},
            {"dataset": dataset, "kind": "psm_json",
             **_fingerprint(psm_path)},
        ])

    matching_audit = pd.concat(matching_tables, ignore_index=True)
    parity_audit = pd.DataFrame(parity_rows)
    if len(samples) != expected_count:
        raise ValueError(
            f"extracted {len(samples)} pilot samples; expected {expected_count}")
    if not len(parity_audit):
        raise ValueError("no parity features were available for verification")
    failed = parity_audit[~parity_audit["passed"]]
    if len(failed):
        failure = Path(output_root).resolve().with_suffix(
            ".parity_failure.csv")
        failure.parent.mkdir(parents=True, exist_ok=True)
        parity_audit.to_csv(failure, index=False)
        raise ValueError(
            f"Phase 2 parity failed for {len(failed)}/{len(parity_audit)} "
            f"sample-feature values; audit={failure}")

    build_metadata = {
        "mode": "balanced_integrity_pilot",
        "n_expected_samples": expected_count,
        "correct_per_dataset": correct_per_dataset,
        "incorrect_per_dataset": error_per_dataset,
        "pilot_seed": pilot_seed,
        "prediction_included": prediction_enabled,
        "chemistry_by_dataset": chemistry,
        "frozen_protocol_root": str(Path(protocol_root).resolve()),
        "frozen_sample_ids_exact_match": True,
        "parity_all_passed": True,
        "parity_comparisons": len(parity_audit),
        "source_fingerprints": sources,
        "build_config": _fingerprint(config_path_obj),
        "split_config": _fingerprint(Path(split_config_path).resolve()),
        "git_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return write_signal_dataset(
        samples, output_root, settings, build_metadata=build_metadata,
        shard_size=shard_size, overwrite=overwrite,
        audit_tables={
            "pilot_selection.csv": selected[[
                column for column in (
                    "sample_id", "dataset", "label", "negative_tier",
                    "fixed_split", "outer_fold", "sequence")
                if column in selected
            ]],
            "identity_matching.csv": matching_audit,
            "feature_parity.csv": parity_audit,
        },
        audit_documents=audit_documents,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Phase 2 raw-XIC integrity pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s")
    report = build_signal_dataset(
        args.config, args.split_config, args.feature_root,
        args.protocol_root, args.output_root, cache_root=args.cache_root,
        overwrite=args.overwrite)
    logging.info("Phase 2 pilot complete: %s", report)


if __name__ == "__main__":
    main()
