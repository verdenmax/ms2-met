"""Preflight policy for modified PSMs under uniform metabolic labeling."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os

from spectrum.labeling import (
    HeavyType,
    canonical_labeling_name,
    parse_heavy_type,
    supports_modified_peptide,
)


AUDIT_SCHEMA = "modified_psm_audit_v1"
DROP_WITH_AUDIT = "drop_with_audit"
REJECT = "reject"
SUPPORTED_POLICIES = frozenset({DROP_WITH_AUDIT, REJECT})


def parse_modified_psm_policy(value: str | None) -> str:
    """Normalize a configured policy; safety defaults to ``reject``."""
    policy = str(value or REJECT).strip().lower()
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"非法 modified_psm_policy={value!r}; "
            f"支持 {sorted(SUPPORTED_POLICIES)}")
    return policy


def _audit_path(result_file: str) -> str:
    return os.fspath(result_file) + ".modified_psm_audit.json"


def apply_modified_psm_policy(
    psms: list,
    labeling: HeavyType | str,
    policy: str | None,
    *,
    result_file: str,
) -> tuple[list, dict]:
    """Apply the chemistry guard and always write a machine-readable audit.

    SILAC retains modified PSMs because its K/R label mass is independent of
    PTM elemental composition.  C13/N15 cannot yet decide whether every PTM
    atom originated before or after metabolic labeling, so modified PSMs are
    either rejected or explicitly dropped before feature workers are started.
    """
    selected = parse_heavy_type(labeling)
    normalized_policy = parse_modified_psm_policy(policy)
    modified = [psm for psm in psms if getattr(psm, "_modify", None)]

    if supports_modified_peptide(selected):
        kept = list(psms)
        action = "retained_supported"
        dropped = []
    elif modified and normalized_policy == REJECT:
        raise ValueError(
            f"{canonical_labeling_name(selected)} 输入包含 {len(modified)} 条"
            "带修饰 PSM；当前无法确定修饰基团是否参与代谢标记。请设置 "
            "modified_psm_policy = drop_with_audit 显式过滤，或先实现按修饰来源"
            "区分的原子组成模型")
    else:
        dropped = modified
        kept = [psm for psm in psms if not getattr(psm, "_modify", None)]
        action = "dropped" if dropped else "none_present"

    label_counts = Counter(
        str(getattr(psm, "_label_type", None) or "unknown")
        for psm in dropped)
    unimod_counts = Counter()
    dropped_rows = []
    for psm in dropped:
        mods = []
        seen_ids = set()
        for position, unimod_id in psm._modify:
            uid = int(unimod_id)
            mods.append({"position": int(position), "unimod_id": uid})
            seen_ids.add(uid)
        for uid in seen_ids:
            unimod_counts[str(uid)] += 1
        dropped_rows.append({
            "sequence": psm._sequence,
            "charge": int(psm._charge),
            "raw_title": psm._raw_title,
            "label_type": str(psm._label_type or "unknown"),
            "modifications": mods,
        })

    audit = {
        "schema": AUDIT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "labeling": canonical_labeling_name(selected),
        "modified_psm_policy": normalized_policy,
        "action": action,
        "n_input_psms": len(psms),
        "n_modified_psms": len(modified),
        "n_dropped_psms": len(dropped),
        "n_output_psms": len(kept),
        "dropped_counts_by_label_type": dict(sorted(label_counts.items())),
        "dropped_psm_counts_by_unimod_id": dict(sorted(unimod_counts.items())),
        "dropped_psms": dropped_rows,
        "reason": (
            "PTM atom origin/timing is not represented by the current ideal "
            "C13/N15 full-label model"
        ) if dropped else None,
    }
    output = _audit_path(result_file)
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logging.info(
        "修饰 PSM 审计: policy=%s input=%d modified=%d dropped=%d output=%d; %s",
        normalized_policy, len(psms), len(modified), len(dropped), len(kept),
        output,
    )
    if not kept:
        raise ValueError("修饰 PSM 策略执行后没有剩余 PSM；详见 " + output)
    return kept, audit
