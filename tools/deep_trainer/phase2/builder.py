"""Build audited pilot or full Phase 2 raw-XIC datasets.

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
from artifact_identity import file_fingerprint
from manager.data_manager import DataManager
from spectrum.dia_data import DIAData
from spectrum.labeling import (
    COMPATIBLE_LEGACY_ISOTOPE_MODELS, IDEAL_FULL_LABEL_ISOTOPE_MODEL,
    HeavyType, canonical_labeling_name, parse_heavy_type,
)
from spectrum.psm_dataset_manifest import validate_manifest
from spectrum.psm_info import PSMInfo
from workflows.flow_utils import get_filename_stem
from workflows.modified_psm_policy import apply_modified_psm_policy
from workflows.pred_store import build_pred_store, normalize_key

from tools.deep_trainer.spec_adapter import prepare_protocol

from .extraction import extract_signal_sample
from .cache import resolve_dia_cache
from .matching import match_psms_to_protocol, select_pilot_rows
from .parity import compare_to_feature_row
from .schema import ExtractionSettings
from .store import StagedValidation, write_signal_dataset


PILOT_CONFIG_SCHEMA = "phase2_xic_pilot_config_v1"
DATASET_CONFIG_SCHEMA = "phase2_xic_dataset_config_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    for column in row.index:
        if column.startswith("inner_valid_for_fold_") and not pd.isna(row[column]):
            metadata[column] = bool(row[column])
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


def _feature_snapshot_isotope_models(frame: pd.DataFrame) -> list[str]:
    """Classify the frozen feature chemistry before expensive raw I/O."""
    if "isotope_model" not in frame:
        return ["undeclared_legacy"]
    normalized = set()
    for value in frame["isotope_model"]:
        text = "" if pd.isna(value) else str(value).strip()
        normalized.add(text or "undeclared_legacy")
    allowed = {
        "undeclared_legacy", IDEAL_FULL_LABEL_ISOTOPE_MODEL,
        *COMPATIBLE_LEGACY_ISOTOPE_MODELS,
    }
    unknown = sorted(normalized - allowed)
    if unknown:
        raise ValueError(
            "frozen feature snapshot declares unknown isotope_model values "
            f"{unknown}; allowed={sorted(allowed)}. Refuse before raw XIC "
            "extraction because isotope parity semantics are undefined.")
    return sorted(normalized)


def _select_rows(protocol, build_config: dict) -> tuple[pd.DataFrame, dict]:
    schema = build_config.get("schema")
    if schema == PILOT_CONFIG_SCHEMA:
        mode = "pilot"
        declaration = build_config.get("pilot", {})
    elif schema == DATASET_CONFIG_SCHEMA:
        mode = str(build_config.get(
            "selection", {}).get("mode", "full")).strip().lower()
        declaration = (
            build_config.get("pilot", {}) if mode == "pilot" else {})
    else:
        raise ValueError(f"unsupported Phase 2 build config: {schema!r}")
    if mode not in {"pilot", "full"}:
        raise ValueError("selection.mode must be pilot or full")
    if mode == "pilot":
        correct = int(declaration.get("correct_per_dataset", 200))
        error = int(declaration.get("error_per_dataset", 200))
        seed = int(declaration.get("seed", 20260813))
        selected = select_pilot_rows(
            protocol.frame, correct_per_dataset=correct,
            error_per_dataset=error, seed=seed)
        contract = {
            "mode": "balanced_integrity_pilot",
            "correct_per_dataset": correct,
            "incorrect_per_dataset": error,
            "seed": seed,
        }
    else:
        selected = protocol.frame.copy()
        selected[protocol.dataset_col] = selected[
            protocol.dataset_col].astype(str)
        selected[protocol.sample_id_col] = selected[
            protocol.sample_id_col].astype(str)
        selected = selected.sort_values(
            [protocol.dataset_col, protocol.sample_id_col]
        ).reset_index(drop=True)
        contract = {
            "mode": "full_frozen_protocol",
            "sample_ids_exactly_equal_frozen_protocol": True,
        }
    if selected[protocol.sample_id_col].duplicated().any():
        raise ValueError("selected Phase 2 rows contain duplicate sample IDs")
    return selected, contract


def _parity_rows(protocol, selected: pd.DataFrame,
                 build_config: dict) -> pd.DataFrame:
    validation = build_config.get("validation", {})
    if build_config.get("schema") == PILOT_CONFIG_SCHEMA or \
            build_config.get("selection", {}).get("mode", "full") == "pilot":
        return selected.copy()
    correct = int(validation.get("parity_correct_per_dataset", 200))
    error = int(validation.get("parity_error_per_dataset", 200))
    seed = int(validation.get("parity_seed", 20260813))
    return select_pilot_rows(
        selected, correct_per_dataset=correct,
        error_per_dataset=error, seed=seed)


def build_signal_dataset(
    config_path: str,
    split_config_path: str,
    feature_root: str,
    protocol_root: str,
    output_root: str,
    *,
    cache_root: str | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> dict:
    """Build and publish an immutable Phase 2 raw-XIC dataset."""
    config_path_obj = Path(config_path).resolve()
    with config_path_obj.open(encoding="utf-8") as handle:
        build_config = yaml.safe_load(handle)
    if build_config.get("schema") not in {
            PILOT_CONFIG_SCHEMA, DATASET_CONFIG_SCHEMA}:
        raise ValueError(
            f"unsupported Phase 2 build config: {build_config.get('schema')!r}")

    extraction = build_config.get("extraction", {})
    settings = ExtractionSettings(
        xic_cycle_window=int(extraction.get("xic_cycle_window", 6)),
        mass_tol_ppm=float(extraction.get("mass_tol_ppm", 10.0)),
        fragment_charges=tuple(extraction.get("fragment_charges", [1, 2])),
    )
    storage = build_config.get("storage", {})
    shard_size = int(storage.get("shard_size", 256))
    resume_enabled = bool(resume or storage.get("resume", False))
    prediction_enabled = bool(
        build_config.get("prediction", {}).get("include", False))

    protocol = prepare_protocol(
        split_config_path, feature_root, "combined", protocol_root)
    selected, selection_contract = _select_rows(protocol, build_config)
    parity_selection = _parity_rows(protocol, selected, build_config)
    snapshot_isotope_models = _feature_snapshot_isotope_models(selected)
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
    sources = []
    chemistry = {}

    def _record_source(provenance: dict) -> None:
        key = (
            str(provenance.get("dataset")), str(provenance.get("kind")),
            str(provenance.get("configured_raw_path", "")),
            str(provenance.get("path", "")),
        )
        existing = {
            (
                str(item.get("dataset")), str(item.get("kind")),
                str(item.get("configured_raw_path", "")),
                str(item.get("path", "")),
            ): item
            for item in sources
        }.get(key)
        if existing is not None and existing != provenance:
            raise ValueError(
                "Phase 2 resume source identity changed for "
                f"dataset={key[0]}, kind={key[1]}, path={key[2] or key[3]}")
        if existing is None:
            sources.append(provenance)

    def _sample_stream(completed_ids=frozenset(), checkpoint_metadata=None):
        checkpoint_metadata = checkpoint_metadata or {}
        for item in checkpoint_metadata.get("source_fingerprints", []):
            _record_source(item)
        for dataset, labeling in checkpoint_metadata.get(
                "chemistry_by_dataset", {}).items():
            chemistry[str(dataset)] = str(labeling)

        for dataset in sorted(datasets):
            declaration = datasets[dataset]
            relative_config = declaration.get("extraction_config")
            if not relative_config:
                raise ValueError(f"dataset {dataset} lacks extraction_config")
            ini_path = _resolve(
                relative_config, base=Path(feature_root).resolve())
            ini = _load_ini(ini_path)
            _assert_extraction_settings(ini, settings, dataset)
            psms, psm_path, labeling = _load_psms(
                ini, dataset=dataset, audit_documents=audit_documents)
            observed_chemistry = canonical_labeling_name(labeling)
            if dataset in chemistry and chemistry[dataset] != observed_chemistry:
                raise ValueError(
                    f"Phase 2 resume chemistry changed for {dataset}")
            chemistry[dataset] = observed_chemistry
            _record_source({
                "dataset": dataset, "kind": "extraction_config",
                **file_fingerprint(ini_path),
            })
            _record_source({
                "dataset": dataset, "kind": "psm_json",
                **file_fingerprint(psm_path),
            })
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
                    f"dataset {dataset}: {len(bad)} PSM identities did not "
                    f"match uniquely; audit={failure}")

            selected_psms = [
                matched[str(sample_id)] for sample_id in domain_rows["sample_id"]
                if str(sample_id) not in completed_ids
            ]
            predictions = _load_predictions(
                ini, selected_psms, prediction_enabled)
            raw_paths = _raw_paths(ini)
            rows_by_raw = {}
            for _, row in domain_rows.iterrows():
                psm = matched[str(row["sample_id"])]
                rows_by_raw.setdefault(
                    psm._raw_title, []).append((row, psm))
            unknown = sorted(set(rows_by_raw) - set(raw_paths))
            if unknown:
                raise ValueError(
                    f"dataset {dataset}: PSM raw titles absent from extraction "
                    f"config: {unknown[:10]}")

            manager = DataManager(ini)
            for raw_title in sorted(rows_by_raw):
                raw_path = raw_paths[raw_title]
                shared_path, cache_provenance = resolve_dia_cache(
                    manager, raw_path, cache_root_path, dataset=dataset)
                _record_source(cache_provenance)
                pending_rows = [
                    (row, psm) for row, psm in rows_by_raw[raw_title]
                    if str(row["sample_id"]) not in completed_ids
                ]
                if not pending_rows:
                    continue
                dia = DIAData.load_from_file(str(shared_path), use_mmap=True)
                for row, psm in pending_rows:
                    pred_frags = None
                    if predictions is not None:
                        record = predictions.get(normalize_key(
                            psm._sequence, psm._modify, psm._charge))
                        pred_frags = record["frags"] if record else None
                    yield extract_signal_sample(
                        psm, dia, settings, _source_row_metadata(row, psm),
                        labeling=labeling, pred_frags=pred_frags)
                del dia

    selected_by_id = selected.copy()
    selected_by_id["sample_id"] = selected_by_id["sample_id"].astype(str)
    selected_by_id = selected_by_id.set_index("sample_id", verify_integrity=True)
    parity_ids = set(parity_selection["sample_id"].astype(str))

    def _validate_serialized_dataset(dataset) -> StagedValidation:
        """Compare reconstructed features from the actual saved shards."""
        observed_ids = dataset.manifest["sample_id"].astype(str)
        observed_set = set(observed_ids)
        expected_set = set(selected_by_id.index)
        if observed_set != expected_set or len(dataset) != expected_count:
            raise ValueError(
                "serialized sample membership differs from frozen selection: "
                f"expected={expected_count}, observed={len(dataset)}, "
                f"missing={len(expected_set - observed_set)}, "
                f"unexpected={len(observed_set - expected_set)}")
        parity_rows = []
        parity_indices = [
            index for index, sample_id in enumerate(observed_ids)
            if sample_id in parity_ids
        ]
        for index in parity_indices:
            sample = dataset.sample(index)
            sample_id = str(sample.metadata["sample_id"])
            if sample_id not in selected_by_id.index:
                raise ValueError(
                    f"serialized sample is absent from frozen pilot: {sample_id}")
            sample.validate(settings)
            parity_rows.extend(compare_to_feature_row(
                sample, selected_by_id.loc[sample_id], settings))
        parity_audit = pd.DataFrame(parity_rows)
        if not len(parity_audit):
            raise ValueError(
                "no parity features were available for serialized verification")
        required = parity_audit.get(
            "required_for_publish",
            pd.Series(True, index=parity_audit.index, dtype=bool),
        ).astype(bool)
        failed = parity_audit[required & ~parity_audit["passed"]]
        migration_mismatches = parity_audit[
            ~required & ~parity_audit["passed"]]
        if len(failed):
            failure = Path(output_root).resolve().with_suffix(
                ".parity_failure.csv")
            failure.parent.mkdir(parents=True, exist_ok=True)
            parity_audit.to_csv(failure, index=False)
            raise ValueError(
                f"Phase 2 serialized parity failed for "
                f"{len(failed)}/{int(required.sum())} required "
                f"sample-feature values; "
                f"audit={failure}")
        return StagedValidation(
            audit_tables={
                "feature_parity.csv": parity_audit,
                "identity_matching.csv": pd.concat(
                    matching_tables, ignore_index=True),
            },
            summary={
                "serialized_shards_validated": True,
                "frozen_membership_exact_match": True,
                "required_feature_parity_all_passed": True,
                "feature_parity_comparisons": len(parity_audit),
                "required_feature_parity_comparisons": int(required.sum()),
                "legacy_isotope_audit_comparisons": int((~required).sum()),
                "legacy_isotope_audit_mismatches": len(
                    migration_mismatches),
            },
        )

    protocol_root_path = Path(protocol_root).resolve()
    protocol_summary = protocol_root_path / "summary.json"

    build_metadata = {
        **selection_contract,
        "n_expected_samples": expected_count,
        "n_parity_samples": len(parity_selection),
        "prediction_included": prediction_enabled,
        "chemistry_by_dataset": chemistry,
        "feature_snapshot_isotope_models": snapshot_isotope_models,
        "legacy_isotope_parity_policy": (
            "isotope_correlation_audited_not_publish_blocking_until_feature_"
            "snapshot_uses_current_exact_mass_model"
        ),
        "frozen_protocol_root": str(protocol_root_path),
        "frozen_protocol_contract": protocol.validation.get(
            "frozen_protocol", {}),
        "target_fprs": list(protocol.target_fprs),
        "split_group_col": protocol.group_col,
        "frozen_protocol_summary": file_fingerprint(protocol_summary),
        "frozen_sample_ids_exact_match": True,
        "parity_validation": "required_on_serialized_staging_shards",
        "source_fingerprints": sources,
        "build_config": file_fingerprint(config_path_obj),
        "split_config": file_fingerprint(Path(split_config_path).resolve()),
        "git_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    selection_digest = hashlib.sha256("\n".join(
        sorted(selected_by_id.index)).encode("utf-8")).hexdigest()
    resume_identity = {
        "schema": "phase2_builder_resume_identity_v1",
        "build_config": {
            "path": str(config_path_obj),
            "sha256": file_fingerprint(config_path_obj)["sha256"],
        },
        "split_config": {
            "path": str(Path(split_config_path).resolve()),
            "sha256": file_fingerprint(
                Path(split_config_path).resolve())["sha256"],
        },
        "frozen_protocol_summary": {
            "path": str(protocol_summary),
            "sha256": file_fingerprint(protocol_summary)["sha256"],
        },
        "selection_sample_ids_sha256": selection_digest,
        "n_expected_samples": expected_count,
    }
    return write_signal_dataset(
        _sample_stream, output_root, settings, build_metadata=build_metadata,
        shard_size=shard_size, overwrite=overwrite,
        audit_tables={
            "selection.csv": selected[[
                column for column in (
                    "sample_id", "dataset", "label", "negative_tier",
                    "fixed_split", "outer_fold", "sequence")
                if column in selected
            ]],
            "parity_selection.csv": parity_selection[[
                column for column in (
                    "sample_id", "dataset", "label", "negative_tier",
                    "fixed_split", "outer_fold", "sequence")
                if column in parity_selection
            ]],
        },
        audit_documents=audit_documents,
        staged_validator=_validate_serialized_dataset,
        resume=resume_enabled,
        resume_identity=resume_identity if resume_enabled else None,
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s")
    report = build_signal_dataset(
        args.config, args.split_config, args.feature_root,
        args.protocol_root, args.output_root, cache_root=args.cache_root,
        overwrite=args.overwrite, resume=args.resume)
    logging.info("Phase 2 XIC dataset complete: %s", report)


if __name__ == "__main__":
    main()
