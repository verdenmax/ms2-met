"""Prepare audited, heavy-confirmed parents for counterfactual generation.

The public interface is ``prepare_counterfactual_parents``.  It deliberately
does not derive a heavy-confirmation rule from signal features: callers supply
an explicit confirmation table and a versioned rule name.  The module owns
identity matching, raw-split enforcement, parent eligibility, peptide-family
identity, provenance, and audit output.
"""

from __future__ import annotations

import argparse
import configparser
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Sequence

import pandas as pd

from spectrum.labeling import (
    HeavyType,
    canonical_labeling_name,
    has_label_site,
    parse_heavy_type,
)
from spectrum.psm_dataset_manifest import validate_manifest, write_manifest
from spectrum.psm_identity import (
    PEPTIDE_GROUP_ID_SCHEMA,
    li_normalize_sequence,
    peptide_group_id,
)
from spectrum.psm_info import PSMInfo


PREPARATION_SCHEMA = "counterfactual_parent_preparation_v1"
AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")
_TRUE_VALUES = frozenset({"1", "true", "yes", "confirmed"})
_FALSE_VALUES = frozenset({"0", "false", "no", "unconfirmed"})


@dataclass(frozen=True)
class ParentPreparationConfig:
    """Scientific selection contract independent of input/output paths."""

    dataset_split: str
    confirmation_rule: str
    confirmation_column: str = "heavy_confirmed"
    labeling: HeavyType = HeavyType.SILAC
    min_length: int = 7
    max_length: int = 40
    require_tryptic_c_terminus: bool = True
    exclude_modified: bool = True


@dataclass(frozen=True)
class ParentPreparationJob:
    """File adapter configuration for one preparation run."""

    input_psms: str
    confirmation_table: str
    raw_split_table: str
    output_psms: str
    output_manifest: str
    output_audit: str
    prepare: ParentPreparationConfig


@dataclass(frozen=True)
class ParentPreparationResult:
    """Audited in-memory parents returned through the module interface."""

    psms: tuple[PSMInfo, ...]
    manifest: pd.DataFrame
    audit: dict


def _validate_config(cfg: ParentPreparationConfig) -> None:
    if not str(cfg.dataset_split).strip():
        raise ValueError("dataset_split is required")
    if not str(cfg.confirmation_rule).strip():
        raise ValueError("confirmation_rule is required")
    if not str(cfg.confirmation_column).strip():
        raise ValueError("confirmation_column is required")
    if cfg.labeling != HeavyType.SILAC:
        raise ValueError(
            "counterfactual parent preparation currently supports SILAC only")
    if cfg.min_length < 1 or cfg.max_length < cfg.min_length:
        raise ValueError("invalid parent length range")
    if not cfg.exclude_modified:
        raise ValueError(
            "counterfactual parent preparation requires exclude_modified=true")


def _table_raw_column(frame: pd.DataFrame) -> str:
    for column in ("raw_title", "raw_title1", "Run", "run"):
        if column in frame:
            return column
    raise ValueError(
        "table requires one raw column: raw_title, raw_title1, Run, or run")


def _parse_confirmation(value) -> bool:
    if pd.isna(value):
        raise ValueError("missing value")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(f"unsupported value {value!r}")


def _confirmation_index(
    frame: pd.DataFrame,
    confirmation_column: str,
) -> dict[tuple[str, int, str], dict]:
    required = {"sequence", "charge", confirmation_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"confirmation table missing columns: {missing}")
    raw_column = _table_raw_column(frame)
    index: dict[tuple[str, int, str], dict] = {}
    invalid = []
    for row_index, row in frame.iterrows():
        try:
            sequence = str(row["sequence"]).strip().upper()
            charge = int(row["charge"])
            raw_title = str(row[raw_column]).strip()
            if not sequence or charge <= 0 or not raw_title:
                raise ValueError("empty/invalid identity")
            confirmed = _parse_confirmation(row[confirmation_column])
        except (TypeError, ValueError) as exc:
            invalid.append({"row": int(row_index), "reason": str(exc)})
            continue
        key = (sequence, charge, raw_title)
        if key in index:
            raise ValueError(
                "confirmation table has duplicate parent identity "
                f"{key!r}; one authoritative row is required")
        index[key] = {
            "confirmed": confirmed,
            "label_type": row.get("label_type"),
            "row_index": int(row_index),
        }
    if invalid:
        raise ValueError(
            "confirmation table has invalid rows: "
            f"{invalid[:5]}")
    return index


def _raw_split_index(frame: pd.DataFrame) -> dict[str, str]:
    if "dataset_split" not in frame:
        raise ValueError("raw split table missing column 'dataset_split'")
    raw_column = _table_raw_column(frame)
    index: dict[str, str] = {}
    for row_index, row in frame.iterrows():
        raw_title = str(row[raw_column]).strip()
        split = str(row["dataset_split"]).strip()
        if not raw_title or not split:
            raise ValueError(
                f"raw split table row {row_index} has empty raw/split")
        if raw_title in index:
            raise ValueError(
                f"raw split table has duplicate raw_title {raw_title!r}")
        index[raw_title] = split
    if not index:
        raise ValueError("raw split table is empty")
    return index


def _parent_key(psm: PSMInfo) -> tuple[str, int, str]:
    return (
        str(psm._sequence).strip().upper(),
        int(psm._charge),
        str(psm._raw_title).strip(),
    )


def _eligibility_reason(
    psm: PSMInfo,
    cfg: ParentPreparationConfig,
) -> str | None:
    sequence = str(psm._sequence).strip().upper()
    if psm._label_type != "positive":
        return "not_positive"
    if cfg.exclude_modified and psm._modify:
        return "modified"
    if not cfg.min_length <= len(sequence) <= cfg.max_length:
        return "length"
    if not sequence or not set(sequence) <= AA_SET:
        return "nonstandard_sequence"
    if cfg.require_tryptic_c_terminus and sequence[-1] not in "KR":
        return "nontryptic_terminus"
    if not has_label_site(sequence, cfg.labeling):
        return "no_label_site"
    if int(psm._charge) <= 0:
        return "invalid_charge"
    if not math.isfinite(float(psm._rt)):
        return "invalid_rt"
    if (not math.isfinite(float(psm._precursor_mz))
            or float(psm._precursor_mz) <= 0):
        return "invalid_precursor_mz"
    return None


def _clone_prepared_parent(
    psm: PSMInfo,
    cfg: ParentPreparationConfig,
) -> PSMInfo:
    clone = PSMInfo.from_dict(psm.to_dict())
    expected_group = peptide_group_id(clone._sequence)
    if clone._peptide_group_id not in (None, expected_group):
        raise ValueError(
            "input PSM has incompatible peptide_group_id: "
            f"sequence={clone._sequence!r}, "
            f"observed={clone._peptide_group_id!r}, expected={expected_group!r}")
    clone._sequence = str(clone._sequence).strip().upper()
    clone._heavy_confirmed = True
    clone._dataset_split = cfg.dataset_split
    clone._peptide_group_id = expected_group
    return clone


def prepare_counterfactual_parents(
    psms: Sequence[PSMInfo],
    confirmations: pd.DataFrame,
    raw_splits: pd.DataFrame,
    cfg: ParentPreparationConfig,
) -> ParentPreparationResult:
    """Return parents that satisfy truth, split, and family contracts."""
    _validate_config(cfg)
    confirmation_index = _confirmation_index(
        confirmations, cfg.confirmation_column)
    raw_split_index = _raw_split_index(raw_splits)

    input_raws = {str(psm._raw_title).strip() for psm in psms}
    unmapped_raws = sorted(input_raws - set(raw_split_index))
    if unmapped_raws:
        raise ValueError(
            "input PSM raws are missing from raw split table: "
            f"{unmapped_raws[:10]}")

    positive_keys = [_parent_key(psm) for psm in psms
                     if psm._label_type == "positive"]
    duplicate_positive_keys = sorted(
        key for key, count in Counter(positive_keys).items() if count > 1)
    if duplicate_positive_keys:
        raise ValueError(
            "input PSM JSON has duplicate positive parent identities: "
            f"{duplicate_positive_keys[:5]}")

    failures: Counter = Counter()
    selected: list[tuple[PSMInfo, int]] = []
    used_confirmation_rows: set[int] = set()
    for psm in psms:
        reason = _eligibility_reason(psm, cfg)
        if reason is not None:
            failures[reason] += 1
            continue
        key = _parent_key(psm)
        split = raw_split_index[key[2]]
        if split != cfg.dataset_split:
            failures["outside_dataset_split"] += 1
            continue
        confirmation = confirmation_index.get(key)
        if confirmation is None:
            failures["confirmation_missing"] += 1
            continue
        label_type = confirmation["label_type"]
        if pd.notna(label_type) and str(label_type).strip() \
                and str(label_type).strip() != "positive":
            failures["confirmation_not_positive"] += 1
            continue
        if not confirmation["confirmed"]:
            failures["not_heavy_confirmed"] += 1
            continue
        clone = _clone_prepared_parent(psm, cfg)
        selected.append((clone, confirmation["row_index"]))
        used_confirmation_rows.add(confirmation["row_index"])

    if not selected:
        raise ValueError("no eligible heavy-confirmed parents remained")
    selected.sort(key=lambda item: (
        item[0]._peptide_group_id,
        item[0]._sequence,
        int(item[0]._charge),
        item[0]._raw_title,
        float(item[0]._rt),
    ))

    manifest_rows = []
    prepared_psms = []
    for psm, confirmation_row in selected:
        key = _parent_key(psm)
        observation_payload = "\x1f".join(map(str, key)).encode("utf-8")
        prepared_psms.append(psm)
        manifest_rows.append({
            "parent_observation_id": (
                "PO" + hashlib.sha256(observation_payload).hexdigest()[:24]),
            "peptide_group_id": psm._peptide_group_id,
            "peptide_group_id_schema": PEPTIDE_GROUP_ID_SCHEMA,
            "dataset_split": cfg.dataset_split,
            "sequence": psm._sequence,
            "sequence_li_normalized": li_normalize_sequence(psm._sequence),
            "charge": int(psm._charge),
            "raw_title": psm._raw_title,
            "rt": float(psm._rt),
            "precursor_mz": float(psm._precursor_mz),
            "heavy_confirmed": 1,
            "confirmation_column": cfg.confirmation_column,
            "confirmation_rule": cfg.confirmation_rule,
            "confirmation_row": confirmation_row,
        })
    manifest = pd.DataFrame(manifest_rows)
    audit = {
        "schema": PREPARATION_SCHEMA,
        "dataset_split": cfg.dataset_split,
        "labeling": canonical_labeling_name(cfg.labeling),
        "confirmation": {
            "column": cfg.confirmation_column,
            "rule": cfg.confirmation_rule,
            "derived_by_tool": False,
        },
        "peptide_group_id_schema": PEPTIDE_GROUP_ID_SCHEMA,
        "counts": {
            "input_psms": len(psms),
            "confirmation_rows": len(confirmations),
            "used_confirmation_rows": len(used_confirmation_rows),
            "unused_confirmation_rows": (
                len(confirmations) - len(used_confirmation_rows)),
            "prepared_parents": len(prepared_psms),
            "peptide_groups": manifest["peptide_group_id"].nunique(),
            "raws": manifest["raw_title"].nunique(),
        },
        "failures": {
            str(key): int(value) for key, value in sorted(failures.items())
        },
        "split_contract": {
            "raw_mapping_complete": True,
            "selected_split": cfg.dataset_split,
            "selected_raw_titles": sorted(manifest["raw_title"].unique()),
        },
    }
    return ParentPreparationResult(
        psms=tuple(prepared_psms), manifest=manifest, audit=audit)


def _read_table(path: str) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    return pd.read_csv(path, sep="\t" if suffix in {".tsv", ".txt"} else ",")


def _load_psms(path: str, labeling: HeavyType) -> list[PSMInfo]:
    validate_manifest(path, labeling, require=False)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("input PSM JSON must contain a top-level list")
    return [PSMInfo.from_dict(row) for row in payload]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _fingerprint(path: str) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": os.path.abspath(os.path.expanduser(path)),
        "sha256": digest.hexdigest(),
        "size_bytes": os.path.getsize(path),
    }


def load_job(path: str) -> ParentPreparationJob:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"parent preparation config not found: {path}")
    if "counterfactual_parents" not in parser:
        raise ValueError("config is missing [counterfactual_parents]")
    section = parser["counterfactual_parents"]
    cfg = ParentPreparationConfig(
        dataset_split=section["dataset_split"].strip(),
        confirmation_rule=section["confirmation_rule"].strip(),
        confirmation_column=section.get(
            "confirmation_column", "heavy_confirmed").strip(),
        labeling=parse_heavy_type(section.get("labeling", "silac")),
        min_length=section.getint("min_length", fallback=7),
        max_length=section.getint("max_length", fallback=40),
        require_tryptic_c_terminus=section.getboolean(
            "require_tryptic_c_terminus", fallback=True),
        exclude_modified=section.getboolean(
            "exclude_modified", fallback=True),
    )
    return ParentPreparationJob(
        input_psms=section["input_psms"],
        confirmation_table=section["confirmation_table"],
        raw_split_table=section["raw_split_table"],
        output_psms=section["output_psms"],
        output_manifest=section["output_manifest"],
        output_audit=section["output_audit"],
        prepare=cfg,
    )


def run_job(job: ParentPreparationJob, *, source_config_path: str | None) -> dict:
    cfg = job.prepare
    result = prepare_counterfactual_parents(
        _load_psms(job.input_psms, cfg.labeling),
        _read_table(job.confirmation_table),
        _read_table(job.raw_split_table),
        cfg,
    )
    for path in (job.output_psms, job.output_manifest, job.output_audit):
        _ensure_parent(path)
    with open(job.output_psms, "w", encoding="utf-8") as handle:
        json.dump([psm.to_dict() for psm in result.psms], handle,
                  indent=2, ensure_ascii=False)
        handle.write("\n")
    write_manifest(
        job.output_psms, list(result.psms), cfg.labeling,
        source_config_path=source_config_path)
    result.manifest.to_csv(job.output_manifest, sep="\t", index=False)
    audit = dict(result.audit)
    audit["inputs"] = {
        "psms": _fingerprint(job.input_psms),
        "confirmation_table": _fingerprint(job.confirmation_table),
        "raw_split_table": _fingerprint(job.raw_split_table),
    }
    audit["outputs"] = {
        "psms": job.output_psms,
        "manifest": job.output_manifest,
        "audit": job.output_audit,
    }
    with open(job.output_audit, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare heavy-confirmed counterfactual parent PSMs")
    parser.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    job = load_job(args.config)
    audit = run_job(job, source_config_path=args.config)
    print(json.dumps(audit["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
