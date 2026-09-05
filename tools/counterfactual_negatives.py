"""Build counterfactual negative PSM hypotheses from curated positives.

The generated rows are *candidate hypotheses*, not search-engine results and
not FDR decoys.  A child hypothesis inherits its parent PSM's observed raw,
precursor m/z, retention time, and charge, but replaces the peptide sequence.
Running the ordinary ``feature_type=0`` workflow then asks whether that wrong
sequence can explain the parent's observed light signal and the sequence-
derived heavy coordinates in the same metabolically labelled raw file.

The public interface is ``build_counterfactual_negatives``.  File parsing and
serialization are kept in the CLI adapter at the bottom of this module.

Version 1 deliberately supports unmodified SILAC parents only.  Its
``local_mass_gap`` generator is a conservative precursor to spectrum-guided
local de novo: it replaces an internal short mass gap with a different,
mass-compatible residue string.  It does not claim to use observed fragment
anchors yet; real light/heavy feature extraction and later hardness mining
decide whether the proposal is useful.
"""
from __future__ import annotations

import argparse
import bisect
import configparser
import hashlib
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import pandas as pd
from pyteomics import mass

from spectrum.entrapment_classifier import (
    TargetIndex,
    classify_peptide,
    load_target_fasta,
)
from spectrum.labeling import (
    HeavyType,
    canonical_labeling_name,
    get_heavy_increase_mass,
    has_label_site,
    parse_heavy_type,
)
from spectrum.psm_dataset_manifest import write_manifest
from spectrum.psm_dataset_manifest import validate_manifest
from spectrum.psm_identity import li_normalize_sequence, peptide_group_id
from spectrum.psm_info import PROTON_MASS, PSMInfo


AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = frozenset(AA_ALPHABET)

SOURCE_COMPOSITION_SHUFFLE = "synthetic_composition_shuffle"
SOURCE_KR_POSITION_SHUFFLE = "synthetic_kr_position_shuffle"
SOURCE_LOCAL_MASS_GAP = "synthetic_local_mass_gap"

BUILD_SCHEMA = "counterfactual_negative_build_v1"
LOCAL_PROPOSAL_SCHEMA = "local_mass_gap_v1"


@dataclass(frozen=True)
class CounterfactualConfig:
    """Scientific and sampling contract for hypothesis generation."""

    dataset_split: str
    labeling: HeavyType = HeavyType.SILAC
    seed: int = 42
    composition_shuffle_per_parent: int = 1
    kr_position_shuffle_per_parent: int = 1
    local_denovo_per_parent: int = 1
    max_attempts_per_source: int = 500
    min_length: int = 7
    max_length: int = 40
    precursor_mass_tolerance_ppm: float = 20.0
    fragment_mass_tolerance_da: float = 0.02
    min_sequence_difference: int = 2
    min_changed_fragment_positions: int = 2
    local_min_segment_length: int = 2
    local_max_segment_length: int = 4
    local_min_shared_fragment_fraction: float = 0.5
    require_positive_label: bool = True
    require_tryptic_c_terminus: bool = True
    exclude_modified: bool = True
    include_parent_positives: bool = True
    require_prepared_parents: bool = True


@dataclass(frozen=True)
class CounterfactualJob:
    """CLI adapter configuration; paths stay outside the build interface."""

    parents: str
    target_fasta: str
    output_psms: str
    output_manifest: str
    output_audit: str
    build: CounterfactualConfig
    contaminant_fasta: str | None = None


@dataclass(frozen=True)
class CounterfactualBuildResult:
    """In-memory result returned at the module interface."""

    psms: tuple[PSMInfo, ...]
    manifest: pd.DataFrame
    audit: dict


@dataclass(frozen=True)
class _Proposal:
    sequence: str
    generator: str
    local_start: int | None = None
    local_end: int | None = None
    local_original: str | None = None
    local_replacement: str | None = None


def _stable_id(*parts: object, prefix: str) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return prefix + hashlib.sha1(payload).hexdigest()[:16]


def _sequence_difference(left: str, right: str) -> int:
    a = li_normalize_sequence(left)
    b = li_normalize_sequence(right)
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def _neutral_mass(sequence: str) -> float:
    return float(mass.fast_mass(sequence))


def _precursor_mz(sequence: str, charge: int) -> float:
    return (_neutral_mass(sequence) + charge * PROTON_MASS) / charge


def _ppm_error(observed: float, theoretical: float) -> float:
    if not math.isfinite(observed) or observed <= 0:
        return float("nan")
    return (theoretical - observed) / observed * 1e6


def _fragment_masses(sequence: str) -> dict[str, list[float]]:
    return {
        "b": [
            float(mass.fast_mass(sequence[:i], ion_type="b"))
            for i in range(1, len(sequence))
        ],
        "y": [
            float(mass.fast_mass(sequence[-i:], ion_type="y"))
            for i in range(1, len(sequence))
        ],
    }


def _count_mass_matches(
    reference: Sequence[float], candidate: Sequence[float], tolerance: float,
) -> int:
    """Greedily count one-to-one mass matches between sorted ion lists."""
    left = sorted(float(value) for value in reference)
    right = sorted(float(value) for value in candidate)
    i = j = matches = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches


def _fragment_relationship(
    parent: str, candidate: str, tolerance: float,
) -> tuple[int, int, float]:
    """Return positional changes, shared-ion count, and shared fraction."""
    parent_ions = _fragment_masses(parent)
    candidate_ions = _fragment_masses(candidate)
    changed = 0
    shared = 0
    total = 0
    for ion_type in ("b", "y"):
        parent_values = parent_ions[ion_type]
        candidate_values = candidate_ions[ion_type]
        changed += sum(
            abs(a - b) > tolerance
            for a, b in zip(parent_values, candidate_values)
        )
        shared += _count_mass_matches(
            parent_values, candidate_values, tolerance)
        total += len(candidate_values)
    fraction = shared / total if total else 0.0
    return changed, shared, fraction


def _kr_position_match(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    return all(
        (a in "KR") == (b in "KR") and (a == b if a in "KR" else True)
        for a, b in zip(left, right)
    )


class _MassGapIndex:
    """Sorted residue-string mass index for bounded local proposals."""

    def __init__(self, lengths: Iterable[int]):
        self._entries: dict[int, list[tuple[float, str]]] = {}
        self._masses: dict[int, list[float]] = {}
        for length in sorted(set(int(value) for value in lengths)):
            if length < 1 or length > 4:
                raise ValueError(
                    "local mass-gap segment length must be between 1 and 4")
            entries = [
                (
                    float(sum(mass.std_aa_mass[aa] for aa in residues)),
                    "".join(residues),
                )
                for residues in product(AA_ALPHABET, repeat=length)
            ]
            entries.sort(key=lambda item: (item[0], item[1]))
            self._entries[length] = entries
            self._masses[length] = [item[0] for item in entries]

    def alternatives(
        self,
        segment: str,
        tolerance_da: float,
    ) -> list[str]:
        entries = self._entries[len(segment)]
        masses = self._masses[len(segment)]
        target = sum(mass.std_aa_mass[aa] for aa in segment)
        lo = bisect.bisect_left(masses, target - tolerance_da)
        hi = bisect.bisect_right(masses, target + tolerance_da)
        original_composition = sorted(segment)
        original_li = li_normalize_sequence(segment)
        return [
            sequence
            for _, sequence in entries[lo:hi]
            if sorted(sequence) != original_composition
            and li_normalize_sequence(sequence) != original_li
        ]


def _shuffle_composition(
    sequence: str, rng: random.Random,
) -> str | None:
    """Shuffle all residues except the terminal cleavage residue."""
    if len(sequence) < 3:
        return None
    prefix = list(sequence[:-1])
    if len(set(prefix)) < 2:
        return None
    rng.shuffle(prefix)
    candidate = "".join(prefix) + sequence[-1]
    return candidate if candidate != sequence else None


def _shuffle_fixed_kr(
    sequence: str, rng: random.Random,
) -> str | None:
    movable = [index for index, aa in enumerate(sequence) if aa not in "KR"]
    if len(movable) < 2:
        return None
    original = [sequence[index] for index in movable]
    if len(set(original)) < 2:
        return None
    shuffled = original[:]
    rng.shuffle(shuffled)
    if shuffled == original:
        return None
    chars = list(sequence)
    for index, aa in zip(movable, shuffled):
        chars[index] = aa
    return "".join(chars)


def _local_mass_gap_proposal(
    sequence: str,
    rng: random.Random,
    index: _MassGapIndex,
    cfg: CounterfactualConfig,
) -> _Proposal | None:
    # Keep the terminal cleavage residue outside the local replacement.
    available = len(sequence) - 1
    max_length = min(cfg.local_max_segment_length, available)
    if max_length < cfg.local_min_segment_length:
        return None
    length = rng.randint(cfg.local_min_segment_length, max_length)
    start = rng.randint(0, available - length)
    end = start + length
    original = sequence[start:end]
    neutral_tolerance = (
        _neutral_mass(sequence) * cfg.precursor_mass_tolerance_ppm / 1e6)
    alternatives = index.alternatives(original, neutral_tolerance)
    if not alternatives:
        return None
    replacement = alternatives[rng.randrange(len(alternatives))]
    candidate = sequence[:start] + replacement + sequence[end:]
    return _Proposal(
        sequence=candidate,
        generator=SOURCE_LOCAL_MASS_GAP,
        local_start=start,
        local_end=end,
        local_original=original,
        local_replacement=replacement,
    )


def _clone_parent(parent: PSMInfo, parent_id: str) -> PSMInfo:
    clone = PSMInfo.from_dict(parent.to_dict())
    clone._sequence = str(clone._sequence).upper()
    clone._parent_id = parent_id
    clone._group_id = parent_id
    clone._candidate_family_id = parent_id
    return clone


def _eligible_parent(
    parent: PSMInfo, cfg: CounterfactualConfig,
) -> str | None:
    sequence = str(parent._sequence).strip().upper()
    if cfg.require_prepared_parents:
        if getattr(parent, "_heavy_confirmed", None) is not True:
            return "not_heavy_confirmed"
        if getattr(parent, "_dataset_split", None) != cfg.dataset_split:
            return "dataset_split_mismatch"
        expected_group = peptide_group_id(sequence)
        if getattr(parent, "_peptide_group_id", None) != expected_group:
            return "peptide_group_id_mismatch"
    if cfg.require_positive_label and parent._label_type != "positive":
        return "not_positive"
    if cfg.exclude_modified and parent._modify:
        return "modified"
    if not cfg.min_length <= len(sequence) <= cfg.max_length:
        return "length"
    if not sequence or not set(sequence) <= AA_SET:
        return "nonstandard_sequence"
    if cfg.require_tryptic_c_terminus and sequence[-1] not in "KR":
        return "nontryptic_terminus"
    if not has_label_site(sequence, cfg.labeling):
        return "no_label_site"
    if int(parent._charge) <= 0:
        return "invalid_charge"
    if not math.isfinite(float(parent._rt)):
        return "invalid_rt"
    if not math.isfinite(float(parent._precursor_mz)) \
            or float(parent._precursor_mz) <= 0:
        return "invalid_precursor_mz"
    return None


def _proposal_validity(
    proposal: _Proposal,
    parent: PSMInfo,
    target: TargetIndex,
    contaminant: TargetIndex | None,
    generated: set[str],
    cfg: CounterfactualConfig,
) -> tuple[dict | None, str | None]:
    sequence = proposal.sequence
    parent_sequence = str(parent._sequence).upper()
    if not sequence or not set(sequence) <= AA_SET:
        return None, "nonstandard_sequence"
    if sequence in generated:
        return None, "duplicate"
    sequence_difference = _sequence_difference(sequence, parent_sequence)
    if sequence_difference < cfg.min_sequence_difference:
        return None, "too_close_to_parent"
    if cfg.require_tryptic_c_terminus and sequence[-1] not in "KR":
        return None, "nontryptic_terminus"
    if not has_label_site(sequence, cfg.labeling):
        return None, "no_label_site"
    if classify_peptide(sequence, target) != "L4":
        return None, "target_exact_or_li"
    if contaminant is not None \
            and classify_peptide(sequence, contaminant) != "L4":
        return None, "contaminant_exact_or_li"

    theoretical_mz = _precursor_mz(sequence, int(parent._charge))
    observed_error = _ppm_error(float(parent._precursor_mz), theoretical_mz)
    if not math.isfinite(observed_error) \
            or abs(observed_error) > cfg.precursor_mass_tolerance_ppm:
        return None, "precursor_mass_tolerance"

    changed, shared, shared_fraction = _fragment_relationship(
        parent_sequence, sequence, cfg.fragment_mass_tolerance_da)
    if changed < cfg.min_changed_fragment_positions:
        return None, "insufficient_distinguishing_fragments"
    if proposal.generator == SOURCE_LOCAL_MASS_GAP \
            and shared_fraction < cfg.local_min_shared_fragment_fraction:
        return None, "local_fragment_overlap"

    parent_theoretical_mz = _precursor_mz(
        parent_sequence, int(parent._charge))
    parent_neutral_mass = _neutral_mass(parent_sequence)
    candidate_neutral_mass = _neutral_mass(sequence)
    parent_shift = get_heavy_increase_mass(parent_sequence, cfg.labeling)
    candidate_shift = get_heavy_increase_mass(sequence, cfg.labeling)
    metadata = {
        "sequence_difference_li_normalized": sequence_difference,
        "candidate_theoretical_mz": theoretical_mz,
        "candidate_observed_mass_error_ppm": observed_error,
        "parent_theoretical_mz": parent_theoretical_mz,
        "parent_observed_mass_error_ppm": _ppm_error(
            float(parent._precursor_mz), parent_theoretical_mz),
        "candidate_parent_neutral_mass_delta_da": (
            candidate_neutral_mass - parent_neutral_mass),
        "parent_kr_count": parent_sequence.count("K")
        + parent_sequence.count("R"),
        "candidate_kr_count": sequence.count("K") + sequence.count("R"),
        "k_count_delta": sequence.count("K") - parent_sequence.count("K"),
        "r_count_delta": sequence.count("R") - parent_sequence.count("R"),
        "kr_count_delta": (
            sequence.count("K") + sequence.count("R")
            - parent_sequence.count("K") - parent_sequence.count("R")),
        "kr_position_match": int(
            _kr_position_match(parent_sequence, sequence)),
        "parent_label_shift": parent_shift,
        "candidate_label_shift": candidate_shift,
        "label_shift_delta": candidate_shift - parent_shift,
        "n_changed_fragment_positions": changed,
        "n_shared_theoretical_fragments": shared,
        "shared_theoretical_fragment_fraction": shared_fraction,
        "target_exclusion_scope": "exact_or_li_substring_v1",
    }
    return metadata, None


def _proposal_for_source(
    source: str,
    parent_sequence: str,
    rng: random.Random,
    mass_gap_index: _MassGapIndex | None,
    cfg: CounterfactualConfig,
) -> _Proposal | None:
    if source == SOURCE_COMPOSITION_SHUFFLE:
        sequence = _shuffle_composition(parent_sequence, rng)
        return (_Proposal(sequence, source) if sequence is not None else None)
    if source == SOURCE_KR_POSITION_SHUFFLE:
        sequence = _shuffle_fixed_kr(parent_sequence, rng)
        return (_Proposal(sequence, source) if sequence is not None else None)
    if source == SOURCE_LOCAL_MASS_GAP:
        if mass_gap_index is None:
            return None
        return _local_mass_gap_proposal(
            parent_sequence, rng, mass_gap_index, cfg)
    raise ValueError(f"unknown counterfactual generator: {source}")


def _validate_config(cfg: CounterfactualConfig) -> None:
    if not str(cfg.dataset_split).strip():
        raise ValueError(
            "dataset_split is required; split raws before generating children")
    if parse_heavy_type(cfg.labeling) is not HeavyType.SILAC:
        raise ValueError(
            "counterfactual_negative_build_v1 currently supports SILAC only")
    if not cfg.exclude_modified:
        raise ValueError(
            "counterfactual_negative_build_v1 requires exclude_modified=true")
    counts = (
        cfg.composition_shuffle_per_parent,
        cfg.kr_position_shuffle_per_parent,
        cfg.local_denovo_per_parent,
    )
    if any(value < 0 for value in counts):
        raise ValueError("per-parent candidate counts must be >= 0")
    if sum(counts) == 0:
        raise ValueError("at least one candidate generator must be enabled")
    if cfg.max_attempts_per_source <= 0:
        raise ValueError("max_attempts_per_source must be > 0")
    if cfg.min_length < 1 or cfg.min_length > cfg.max_length:
        raise ValueError("invalid min_length/max_length")
    if cfg.precursor_mass_tolerance_ppm <= 0:
        raise ValueError("precursor_mass_tolerance_ppm must be > 0")
    if cfg.fragment_mass_tolerance_da < 0:
        raise ValueError("fragment_mass_tolerance_da must be >= 0")
    if cfg.min_sequence_difference < 1:
        raise ValueError("min_sequence_difference must be >= 1")
    if cfg.min_changed_fragment_positions < 1:
        raise ValueError("min_changed_fragment_positions must be >= 1")
    if not 0 <= cfg.local_min_shared_fragment_fraction <= 1:
        raise ValueError(
            "local_min_shared_fragment_fraction must be in [0, 1]")
    if cfg.local_min_segment_length > cfg.local_max_segment_length:
        raise ValueError(
            "local_min_segment_length cannot exceed local_max_segment_length")


def build_counterfactual_negatives(
    parents: Sequence[PSMInfo],
    target: TargetIndex,
    cfg: CounterfactualConfig,
    contaminant: TargetIndex | None = None,
) -> CounterfactualBuildResult:
    """Generate deterministic child hypotheses and their provenance.

    The caller must pass a raw/peptide-pre-split parent collection.  Children
    always receive ``label_type='negative'`` (stored label 0 downstream), and
    share their parent's stable ``group_id`` for leak-free splitting.
    """
    _validate_config(cfg)
    source_counts = {
        SOURCE_COMPOSITION_SHUFFLE: cfg.composition_shuffle_per_parent,
        SOURCE_KR_POSITION_SHUFFLE: cfg.kr_position_shuffle_per_parent,
        SOURCE_LOCAL_MASS_GAP: cfg.local_denovo_per_parent,
    }
    mass_gap_index = None
    if cfg.local_denovo_per_parent:
        mass_gap_index = _MassGapIndex(range(
            cfg.local_min_segment_length,
            cfg.local_max_segment_length + 1))

    failures: Counter = Counter()
    eligible: list[PSMInfo] = []
    seen_parents: set[tuple[str, int, str]] = set()
    for parent in parents:
        reason = _eligible_parent(parent, cfg)
        if reason is not None:
            failures[f"parent:{reason}"] += 1
            continue
        key = (
            str(parent._sequence).upper(), int(parent._charge),
            str(parent._raw_title),
        )
        if key in seen_parents:
            failures["parent:duplicate"] += 1
            continue
        seen_parents.add(key)
        eligible.append(parent)

    if not eligible:
        raise ValueError("no eligible positive parents remained")

    output_psms: list[PSMInfo] = []
    manifest_rows: list[dict] = []
    if cfg.include_parent_positives:
        for parent in eligible:
            parent_id = _stable_id(
                str(parent._sequence).upper(), int(parent._charge), prefix="P")
            output_psms.append(_clone_parent(parent, parent_id))

    for parent in eligible:
        parent_sequence = str(parent._sequence).upper()
        parent_id = _stable_id(
            parent_sequence, int(parent._charge), prefix="P")
        seed_payload = (
            f"{cfg.seed}:{parent_id}:{parent._raw_title}".encode("utf-8"))
        parent_seed = int(hashlib.sha1(seed_payload).hexdigest()[:8], 16)
        rng = random.Random(parent_seed)
        generated: set[str] = set()

        for source, wanted in source_counts.items():
            made = 0
            for _ in range(cfg.max_attempts_per_source):
                if made >= wanted:
                    break
                proposal = _proposal_for_source(
                    source, parent_sequence, rng, mass_gap_index, cfg)
                if proposal is None:
                    failures[f"{source}:no_proposal"] += 1
                    continue
                metadata, reason = _proposal_validity(
                    proposal, parent, target, contaminant, generated, cfg)
                if reason is not None:
                    failures[f"{source}:{reason}"] += 1
                    continue
                assert metadata is not None

                query_id = _stable_id(
                    parent_id, parent._raw_title, source,
                    proposal.sequence, parent._charge,
                    prefix="Q",
                )
                child = PSMInfo(
                    sequence=proposal.sequence,
                    charge=int(parent._charge),
                    modify=[],
                    rt=parent._rt,
                    precursor_mz=parent._precursor_mz,
                    raw_title=str(parent._raw_title),
                    protein_names="SYNTHETIC_COUNTERFACTUAL",
                    label_type="negative",
                    query_id=query_id,
                    parent_id=parent_id,
                    group_id=parent_id,
                    candidate_family_id=parent_id,
                    peptide_group_id=parent._peptide_group_id,
                    dataset_split=cfg.dataset_split,
                )
                output_psms.append(child)
                generated.add(proposal.sequence)
                manifest_rows.append({
                    "query_id": query_id,
                    "parent_id": parent_id,
                    "group_id": parent_id,
                    "candidate_family_id": parent_id,
                    "peptide_group_id": parent._peptide_group_id,
                    "dataset_split": cfg.dataset_split,
                    "generator": source,
                    "negative_source": source,
                    "negative_confidence": "silver",
                    "generator_seed": parent_seed,
                    "labeling": canonical_labeling_name(cfg.labeling),
                    "sequence": proposal.sequence,
                    "charge": int(parent._charge),
                    "raw_title": str(parent._raw_title),
                    "rt": float(parent._rt),
                    "precursor_mz": float(parent._precursor_mz),
                    "parent_sequence": parent_sequence,
                    "local_proposal_schema": (
                        LOCAL_PROPOSAL_SCHEMA
                        if source == SOURCE_LOCAL_MASS_GAP else None),
                    "local_uses_observed_fragment_anchors": False,
                    "local_start": proposal.local_start,
                    "local_end": proposal.local_end,
                    "local_original": proposal.local_original,
                    "local_replacement": proposal.local_replacement,
                    **metadata,
                })
                made += 1
            if made < wanted:
                failures[f"{source}:shortfall"] += wanted - made

    if not manifest_rows:
        raise ValueError("counterfactual generation produced zero valid children")
    manifest = pd.DataFrame(manifest_rows)
    by_source = manifest["negative_source"].value_counts().to_dict()
    audit = {
        "schema": BUILD_SCHEMA,
        "dataset_split": cfg.dataset_split,
        "labeling": canonical_labeling_name(cfg.labeling),
        "scope": {
            "modified_peptides": "excluded",
            "parent_truth": (
                "prepared_parent_contract_v2: upstream-filtered "
                "label_type=positive, matching dataset_split, and canonical "
                "peptide_group_id required"
                if cfg.require_prepared_parents else
                "explicit exploratory bypass; prepared parent contract not "
                "required"),
            "target_exclusion": "exact_or_li_substring_v1",
            "local_denovo": (
                "mass-gap proposals only; observed fragment anchors are not "
                "used until a later spectrum-guided implementation"),
            "hardness": (
                "not assigned at generation; extract real light/heavy "
                "features before mining"),
        },
        "counts": {
            "input_parents": len(parents),
            "eligible_parents": len(eligible),
            "output_parent_rows": (
                len(eligible) if cfg.include_parent_positives else 0),
            "negative_children": len(manifest),
            "by_source": {
                str(key): int(value) for key, value in by_source.items()
            },
        },
        "failures": {
            str(key): int(value) for key, value in sorted(failures.items())
        },
        "split_contract": (
            "input parents must already be split by raw before generation; "
            "parent and children share group_id and peptide_group_id"),
    }
    return CounterfactualBuildResult(
        psms=tuple(output_psms), manifest=manifest, audit=audit)


def _load_parent_psms(path: str) -> list[PSMInfo]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if not path.lower().endswith(".json"):
        raise ValueError(
            "counterfactual v1 expects custom PSM JSON from extract_common")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("parent PSM JSON must contain a top-level list")
    return [PSMInfo.from_dict(item) for item in payload]


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_job(path: str) -> CounterfactualJob:
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(f"counterfactual config not found: {path}")
    if "counterfactual" not in parser:
        raise ValueError("config is missing [counterfactual]")
    section = parser["counterfactual"]
    build = CounterfactualConfig(
        dataset_split=section["dataset_split"].strip(),
        labeling=parse_heavy_type(section.get("labeling", "silac")),
        seed=section.getint("seed", fallback=42),
        composition_shuffle_per_parent=section.getint(
            "composition_shuffle_per_parent", fallback=1),
        kr_position_shuffle_per_parent=section.getint(
            "kr_position_shuffle_per_parent", fallback=1),
        local_denovo_per_parent=section.getint(
            "local_denovo_per_parent", fallback=1),
        max_attempts_per_source=section.getint(
            "max_attempts_per_source", fallback=500),
        min_length=section.getint("min_length", fallback=7),
        max_length=section.getint("max_length", fallback=40),
        precursor_mass_tolerance_ppm=section.getfloat(
            "precursor_mass_tolerance_ppm", fallback=20.0),
        fragment_mass_tolerance_da=section.getfloat(
            "fragment_mass_tolerance_da", fallback=0.02),
        min_sequence_difference=section.getint(
            "min_sequence_difference", fallback=2),
        min_changed_fragment_positions=section.getint(
            "min_changed_fragment_positions", fallback=2),
        local_min_segment_length=section.getint(
            "local_min_segment_length", fallback=2),
        local_max_segment_length=section.getint(
            "local_max_segment_length", fallback=4),
        local_min_shared_fragment_fraction=section.getfloat(
            "local_min_shared_fragment_fraction", fallback=0.5),
        require_positive_label=section.getboolean(
            "require_positive_label", fallback=True),
        require_tryptic_c_terminus=section.getboolean(
            "require_tryptic_c_terminus", fallback=True),
        exclude_modified=section.getboolean(
            "exclude_modified", fallback=True),
        include_parent_positives=section.getboolean(
            "include_parent_positives", fallback=True),
        require_prepared_parents=section.getboolean(
            "require_prepared_parents", fallback=True),
    )
    return CounterfactualJob(
        parents=os.path.expanduser(section["parents"]),
        target_fasta=os.path.expanduser(section["target_fasta"]),
        contaminant_fasta=(
            os.path.expanduser(
                section.get("contaminant_fasta", "").strip()) or None),
        output_psms=os.path.expanduser(section["output_psms"]),
        output_manifest=os.path.expanduser(section["output_manifest"]),
        output_audit=os.path.expanduser(section["output_audit"]),
        build=build,
    )


def run_job(job: CounterfactualJob, *, source_config_path: str | None) -> dict:
    validate_manifest(
        job.parents, job.build.labeling,
        require=job.build.require_prepared_parents)
    parents = _load_parent_psms(job.parents)
    target = load_target_fasta(job.target_fasta)
    contaminant = (
        load_target_fasta(job.contaminant_fasta, log_label="contaminant FASTA")
        if job.contaminant_fasta else None)
    result = build_counterfactual_negatives(
        parents, target, job.build, contaminant=contaminant)

    for path in (job.output_psms, job.output_manifest, job.output_audit):
        _ensure_parent(path)
    with open(job.output_psms, "w", encoding="utf-8") as handle:
        json.dump(
            [psm.to_dict() for psm in result.psms], handle,
            indent=2, ensure_ascii=False)
        handle.write("\n")
    write_manifest(
        job.output_psms, list(result.psms), job.build.labeling,
        source_config_path=source_config_path)
    result.manifest.to_csv(job.output_manifest, sep="\t", index=False)
    audit = dict(result.audit)
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
        description="Generate counterfactual negative PSM hypotheses")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audit = run_job(
        load_job(args.config), source_config_path=args.config)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
