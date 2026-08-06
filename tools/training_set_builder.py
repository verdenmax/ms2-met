"""Build synthetic DIA queries and assemble a hard-negative training set.

The module has two public phases separated by an external DIA-search seam:

1. ``generate_queries`` writes a synthetic precursor manifest and FASTA.
   DIA-NN/pFind-DIA search is intentionally external: this repository does
   not own those executables or their large raw-data inputs.
2. ``assemble_training_set`` consumes the ordinary ``features.csv`` emitted
   after those queries have been searched and passed through ms2-met.  It
   applies physical-signal gates, target/L-I exclusion, distribution matching,
   provenance annotation, and generator-signature auditing.

Every output row remains an independently scorable PSM. ``group_id`` is only
for leak-free splitting: a synthetic negative shares its parent positive's
group so the pair cannot straddle train/test folds.

Usage::

    python -m tools.training_set_builder generate --config config.ini
    # Run the generated FASTA through the external DIA search + ms2-met.
    python -m tools.training_set_builder assemble --config config.ini
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pyteomics import mass

from spectrum.entrapment_classifier import (
    TargetIndex,
    classify_peptide,
    load_target_fasta,
)
from spectrum.labeling import parse_heavy_type, supports_modified_peptide
from spectrum.psm_info import (
    HeavyType,
    PROTON_MASS,
    get_heavy_increase_mass,
    has_label_site,
)


logger = logging.getLogger(__name__)

AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
SOURCE_POSITIVE = "gold_positive"
SOURCE_GOLD = "gold_entrapment"
SOURCE_SHUFFLE = "silver_synthetic_shuffle"
SOURCE_MARKOV = "silver_synthetic_markov"

@dataclass(frozen=True)
class QueryBuildConfig:
    positives: str
    target_fasta: str
    output_manifest: str
    output_fasta: str
    contaminant_fasta: str | None = None
    labeling: HeavyType = HeavyType.SILAC
    confirmation_column: str = "heavy_confirmed"
    require_confirmation: bool = True
    shuffle_per_parent: int = 1
    markov_per_parent: int = 1
    markov_order: int = 2
    seed: int = 42
    min_length: int = 7
    max_length: int = 40
    max_attempts: int = 500
    exclude_modified: bool = True
    require_tryptic_c_terminus: bool = True
    mz_bin_width: float = 50.0
    shift_bin_width: float = 20.0


@dataclass(frozen=True)
class AssemblyConfig:
    positive_features: tuple[str, ...]
    gold_features: tuple[str, ...]
    silver_features: tuple[str, ...]
    query_manifest: str
    target_fasta: str
    output_features: str
    output_audit: str
    contaminant_fasta: str | None = None
    labeling: HeavyType = HeavyType.SILAC
    heldout_features: tuple[str, ...] = ()
    require_heldout: bool = True
    confirmation_column: str = "heavy_confirmed"
    require_confirmation: bool = True
    light_fragment_column: str = "all_count"
    heavy_fragment_column: str = "q1a_TP_shifted"
    min_light_fragments: int = 2
    min_heavy_fragments: int = 2
    min_light_precursor_intensity: float = 0.0
    min_heavy_precursor_intensity: float = 0.0
    silver_per_positive_per_source: float = 1.0
    seed: int = 42
    distribution_columns: tuple[str, ...] = (
        "charge",
        "precursor_mz",
        "sequence_len",
        "total_silac_shift",
        "psm_is_split_window",
        "rt",
    )
    precursor_mz_bin_width: float = 50.0
    sequence_len_bin_width: float = 3.0
    total_shift_bin_width: float = 20.0
    rt_bin_width: float = 5.0


def _csv_list(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def load_query_config(path: str) -> QueryBuildConfig:
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError(f"training-set config not found: {path}")
    if "queries" not in cp:
        raise ValueError("config is missing [queries]")
    s = cp["queries"]
    return QueryBuildConfig(
        positives=s["positives"],
        target_fasta=s["target_fasta"],
        contaminant_fasta=s.get("contaminant_fasta", "").strip() or None,
        output_manifest=s["output_manifest"],
        output_fasta=s["output_fasta"],
        labeling=parse_heavy_type(s.get("labeling", "silac")),
        confirmation_column=s.get(
            "confirmation_column", "heavy_confirmed").strip(),
        require_confirmation=s.getboolean(
            "require_confirmation", fallback=True),
        shuffle_per_parent=s.getint("shuffle_per_parent", fallback=1),
        markov_per_parent=s.getint("markov_per_parent", fallback=1),
        markov_order=s.getint("markov_order", fallback=2),
        seed=s.getint("seed", fallback=42),
        min_length=s.getint("min_length", fallback=7),
        max_length=s.getint("max_length", fallback=40),
        max_attempts=s.getint("max_attempts", fallback=500),
        exclude_modified=s.getboolean("exclude_modified", fallback=True),
        require_tryptic_c_terminus=s.getboolean(
            "require_tryptic_c_terminus", fallback=True),
        mz_bin_width=s.getfloat("mz_bin_width", fallback=50.0),
        shift_bin_width=s.getfloat("shift_bin_width", fallback=20.0),
    )


def load_assembly_config(path: str) -> AssemblyConfig:
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError(f"training-set config not found: {path}")
    if "assembly" not in cp:
        raise ValueError("config is missing [assembly]")
    s = cp["assembly"]
    return AssemblyConfig(
        positive_features=_csv_list(s["positive_features"]),
        gold_features=_csv_list(s.get("gold_features", "")),
        silver_features=_csv_list(s["silver_features"]),
        query_manifest=s["query_manifest"],
        target_fasta=s["target_fasta"],
        contaminant_fasta=s.get("contaminant_fasta", "").strip() or None,
        labeling=parse_heavy_type(s.get("labeling", "silac")),
        output_features=s["output_features"],
        output_audit=s["output_audit"],
        heldout_features=_csv_list(s.get("heldout_features", "")),
        require_heldout=s.getboolean("require_heldout", fallback=True),
        confirmation_column=s.get(
            "confirmation_column", "heavy_confirmed").strip(),
        require_confirmation=s.getboolean(
            "require_confirmation", fallback=True),
        light_fragment_column=s.get(
            "light_fragment_column", "all_count").strip(),
        heavy_fragment_column=s.get(
            "heavy_fragment_column", "q1a_TP_shifted").strip(),
        min_light_fragments=s.getint("min_light_fragments", fallback=2),
        min_heavy_fragments=s.getint("min_heavy_fragments", fallback=2),
        min_light_precursor_intensity=s.getfloat(
            "min_light_precursor_intensity", fallback=0.0),
        min_heavy_precursor_intensity=s.getfloat(
            "min_heavy_precursor_intensity", fallback=0.0),
        silver_per_positive_per_source=s.getfloat(
            "silver_per_positive_per_source", fallback=1.0),
        seed=s.getint("seed", fallback=42),
        distribution_columns=_csv_list(s.get(
            "distribution_columns",
            "charge,precursor_mz,sequence_len,total_silac_shift,"
            "psm_is_split_window,rt")),
        precursor_mz_bin_width=s.getfloat(
            "precursor_mz_bin_width", fallback=50.0),
        sequence_len_bin_width=s.getfloat(
            "sequence_len_bin_width", fallback=3.0),
        total_shift_bin_width=s.getfloat(
            "total_shift_bin_width", fallback=20.0),
        rt_bin_width=s.getfloat("rt_bin_width", fallback=5.0),
    )


def _read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"expected a JSON list in {path}")
        return pd.DataFrame(data)
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def _read_many(paths: Sequence[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames = [_read_table(p) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_fasta_sequences(path: str) -> list[str]:
    # load_target_fasta validates and gives consistent uppercase semantics.
    index = load_target_fasta(path)
    return [s for s in index.raw_text.split("|") if s]


def _is_confirmed(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "yes", "confirmed"})


def _positive_rows(
    df: pd.DataFrame,
    confirmation_column: str,
    require_confirmation: bool,
) -> pd.DataFrame:
    if "sequence" not in df or "charge" not in df:
        raise ValueError("positive input requires sequence and charge columns")
    out = df.copy()
    if "label_type" in out:
        out = out[out["label_type"].eq("positive")]
    elif "label" in out:
        out = out[pd.to_numeric(out["label"], errors="coerce").eq(1)]
    if require_confirmation:
        if not confirmation_column or confirmation_column not in out:
            raise ValueError(
                f"confirmed positives require column "
                f"{confirmation_column!r}; set require_confirmation=false "
                "only for an explicit exploratory run")
        out = out[_is_confirmed(out[confirmation_column])]
    return out.reset_index(drop=True)


def _has_modification(row: pd.Series) -> bool:
    if "modification_count" in row.index:
        value = pd.to_numeric(
            pd.Series([row["modification_count"]]), errors="coerce").iloc[0]
        if pd.notna(value):
            return value > 0
    for column in ("modify", "modifications", "Modifications"):
        if column not in row.index:
            continue
        value = row[column]
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        if pd.isna(value):
            return False
        text = str(value).strip()
        return text not in {"", "[]", "nan", "None"}
    return False


def _validate_supported_modifications(
    named_frames: Sequence[tuple[str, pd.DataFrame]],
    labeling: HeavyType,
) -> None:
    """Reject rows whose PTM atoms are outside the uniform-label model."""
    if supports_modified_peptide(labeling):
        return
    for name, frame in named_frames:
        if frame.empty:
            continue
        modified = frame.apply(_has_modification, axis=1)
        if bool(modified.any()):
            raise ValueError(
                f"modified C13/N15 rows are unsupported in {name}: "
                f"{int(modified.sum())} row(s); PTM elemental compositions "
                "are not included in label shifts")


def _stable_id(*parts: object, prefix: str = "") -> str:
    payload = "\x1f".join(str(x) for x in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:16]
    return f"{prefix}{digest}"


def _neutral_mass(sequence: str) -> float:
    return float(mass.fast_mass(sequence))


def _precursor_mz(sequence: str, charge: int) -> float:
    neutral = _neutral_mass(sequence)
    return float((neutral + charge * PROTON_MASS) / charge)


def _label_shift(sequence: str, labeling: HeavyType) -> float:
    return float(get_heavy_increase_mass(sequence, labeling))


def _same_bin(a: float, b: float, width: float) -> bool:
    if width <= 0:
        return True
    return math.floor(float(a) / width) == math.floor(float(b) / width)


def _sequence_difference(a: str, b: str) -> int:
    aa = a.replace("I", "L")
    bb = b.replace("I", "L")
    if len(aa) != len(bb):
        return max(len(aa), len(bb))
    return sum(x != y for x, y in zip(aa, bb))


def _valid_synthetic(
    candidate: str,
    parent: str,
    target: TargetIndex,
    contaminant: TargetIndex | None,
    generated: set[str],
    labeling: HeavyType,
    min_length: int,
    max_length: int,
    require_tryptic_c_terminus: bool,
) -> bool:
    if not min_length <= len(candidate) <= max_length:
        return False
    if not candidate or not set(candidate) <= AA_ALPHABET:
        return False
    if candidate in generated:
        return False
    if _sequence_difference(candidate, parent) < 2:
        return False
    if require_tryptic_c_terminus and candidate[-1] not in "KR":
        return False
    if not has_label_site(candidate, labeling):
        return False
    if classify_peptide(candidate, target) != "L4":
        return False
    if contaminant is not None and classify_peptide(
            candidate, contaminant) != "L4":
        return False
    return True


def _cleavage_preserving_shuffle(
    sequence: str, rng: random.Random, attempts: int,
) -> str | None:
    """Shuffle non-cleavage residues while keeping all K/R positions fixed."""
    movable = [i for i, aa in enumerate(sequence) if aa not in "KR"]
    if len(movable) < 2:
        return None
    original = [sequence[i] for i in movable]
    if len(set(original)) < 2:
        return None
    for _ in range(attempts):
        values = original[:]
        rng.shuffle(values)
        if values == original:
            continue
        chars = list(sequence)
        for i, aa in zip(movable, values):
            chars[i] = aa
        return "".join(chars)
    return None


class _MarkovSampler:
    """Small internal adapter: sample proteome-like sequences from n-grams."""

    def __init__(self, proteins: Iterable[str], order: int):
        if order not in (1, 2):
            raise ValueError("markov_order must be 1 or 2")
        self.order = order
        self.transitions: dict[str, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()
        start = "^" * order
        for protein in proteins:
            seq = "".join(aa for aa in protein.upper() if aa in AA_ALPHABET)
            if not seq:
                continue
            padded = start + seq
            for i, aa in enumerate(seq, start=order):
                self.transitions[padded[i - order:i]][aa] += 1
                self.global_counts[aa] += 1
        if not self.global_counts:
            raise ValueError("target FASTA contains no standard amino acids")

    @staticmethod
    def _weighted_choice(counter: Counter, rng: random.Random,
                         allowed: set[str]) -> str:
        items = [(aa, n) for aa, n in counter.items()
                 if aa in allowed and n > 0]
        if not items:
            raise LookupError("no allowed residue in distribution")
        total = sum(n for _, n in items)
        pick = rng.uniform(0, total)
        upto = 0.0
        for aa, n in items:
            upto += n
            if pick <= upto:
                return aa
        return items[-1][0]

    def sample(self, length: int, terminal: str,
               rng: random.Random) -> str:
        if length < 2:
            raise ValueError("synthetic peptide length must be >=2")
        context = "^" * self.order
        chars: list[str] = []
        # A strict tryptic synthetic peptide has no internal K/R. This does
        # not force its K/R count to equal the parent; it only makes the FASTA
        # query survive an ordinary no-missed-cleavage digest.
        internal_allowed = set(AA_ALPHABET) - {"K", "R"}
        for _ in range(length - 1):
            dist = self.transitions.get(context, self.global_counts)
            try:
                aa = self._weighted_choice(dist, rng, internal_allowed)
            except LookupError:
                aa = self._weighted_choice(
                    self.global_counts, rng, internal_allowed)
            chars.append(aa)
            context = (context + aa)[-self.order:]
        chars.append(terminal if terminal in "KR" else rng.choice("KR"))
        return "".join(chars)


def _parent_records(df: pd.DataFrame, cfg: QueryBuildConfig) -> list[dict]:
    positives = _positive_rows(
        df, cfg.confirmation_column, cfg.require_confirmation)
    seen: set[tuple] = set()
    records: list[dict] = []
    for _, row in positives.iterrows():
        sequence = str(row["sequence"]).strip().upper()
        try:
            charge = int(row["charge"])
        except (TypeError, ValueError):
            continue
        if charge <= 0 or not cfg.min_length <= len(sequence) <= cfg.max_length:
            continue
        if cfg.exclude_modified and _has_modification(row):
            continue
        if cfg.require_tryptic_c_terminus and sequence[-1:] not in "KR":
            continue
        if not has_label_site(sequence, cfg.labeling):
            continue
        key = (sequence, charge)
        if key in seen:
            continue
        seen.add(key)
        observed_mz = pd.to_numeric(
            pd.Series([row.get("precursor_mz")]), errors="coerce").iloc[0]
        parent_mz = (float(observed_mz) if pd.notna(observed_mz)
                     else _precursor_mz(sequence, charge))
        records.append({
            "parent_id": _stable_id(sequence, charge, prefix="P"),
            "sequence": sequence,
            "charge": charge,
            "precursor_mz": parent_mz,
            "label_shift": _label_shift(sequence, cfg.labeling),
        })
    if not records:
        raise ValueError(
            "no eligible confirmed, unmodified parent positives remained")
    return records


def generate_queries(cfg: QueryBuildConfig) -> dict:
    """Generate query manifest + FASTA; return a JSON-serialisable summary."""
    if cfg.shuffle_per_parent < 0 or cfg.markov_per_parent < 0:
        raise ValueError("per-parent query counts must be >=0")
    if cfg.shuffle_per_parent + cfg.markov_per_parent == 0:
        raise ValueError("at least one synthetic generator must be enabled")
    if (not cfg.exclude_modified
            and not supports_modified_peptide(cfg.labeling)):
        raise ValueError(
            "modified C13/N15 peptides are unsupported because PTM "
            "elemental compositions are not included in label shifts; "
            "set exclude_modified=true")

    input_df = _read_table(cfg.positives)
    parents = _parent_records(input_df, cfg)
    target = load_target_fasta(cfg.target_fasta)
    contaminant = (load_target_fasta(
        cfg.contaminant_fasta, log_label="contaminant FASTA")
        if cfg.contaminant_fasta else None)
    proteins = _read_fasta_sequences(cfg.target_fasta)
    markov = _MarkovSampler(proteins, cfg.markov_order)

    generated: set[str] = set()
    rows: list[dict] = []
    failures = Counter()

    for parent in parents:
        sequence = parent["sequence"]
        charge = parent["charge"]
        parent_seed = int(
            hashlib.sha1(f"{cfg.seed}:{parent['parent_id']}".encode()).hexdigest()[:8],
            16)
        rng = random.Random(parent_seed)

        for generator, wanted in (
            (SOURCE_SHUFFLE, cfg.shuffle_per_parent),
            (SOURCE_MARKOV, cfg.markov_per_parent),
        ):
            made = 0
            for attempt in range(cfg.max_attempts):
                if made >= wanted:
                    break
                if generator == SOURCE_SHUFFLE:
                    candidate = _cleavage_preserving_shuffle(
                        sequence, rng, attempts=10)
                    if candidate is None:
                        failures["shuffle_unshufflable"] += 1
                        break
                else:
                    candidate = markov.sample(
                        len(sequence), sequence[-1], rng)

                if not _valid_synthetic(
                    candidate, sequence, target, contaminant, generated,
                    cfg.labeling, cfg.min_length, cfg.max_length,
                    cfg.require_tryptic_c_terminus,
                ):
                    failures[f"{generator}:invalid"] += 1
                    continue

                candidate_mz = _precursor_mz(candidate, charge)
                candidate_shift = _label_shift(candidate, cfg.labeling)
                if generator == SOURCE_MARKOV:
                    if not _same_bin(
                            candidate_mz, parent["precursor_mz"],
                            cfg.mz_bin_width):
                        failures["markov:mz_bin"] += 1
                        continue
                    if not _same_bin(
                            candidate_shift, parent["label_shift"],
                            cfg.shift_bin_width):
                        failures["markov:shift_bin"] += 1
                        continue

                generated.add(candidate)
                query_id = _stable_id(
                    parent["parent_id"], generator, candidate, charge,
                    prefix="Q")
                rows.append({
                    "query_id": query_id,
                    "parent_id": parent["parent_id"],
                    "group_id": parent["parent_id"],
                    "generator": generator,
                    "generator_seed": parent_seed,
                    "sequence": candidate,
                    "charge": charge,
                    "precursor_mz": candidate_mz,
                    "heavy_precursor_mz": (
                        candidate_mz + candidate_shift / charge),
                    "sequence_len": len(candidate),
                    "kr_count": candidate.count("K") + candidate.count("R"),
                    "label_shift": candidate_shift,
                    "parent_sequence": sequence,
                    "parent_precursor_mz": parent["precursor_mz"],
                    "parent_label_shift": parent["label_shift"],
                    "labeling": cfg.labeling.name,
                    "negative_source": generator,
                    "negative_confidence": "silver",
                })
                made += 1
            if made < wanted:
                failures[f"{generator}:shortfall"] += wanted - made

    if not rows:
        raise ValueError("synthetic generation produced zero valid queries")
    manifest = pd.DataFrame(rows)
    _ensure_parent(cfg.output_manifest)
    _ensure_parent(cfg.output_fasta)
    manifest.to_csv(cfg.output_manifest, sep="\t", index=False)
    with open(cfg.output_fasta, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(
                f">SYNTH_{row['query_id']} generator={row['generator']} "
                f"parent={row['parent_id']}\n{row['sequence']}\n")

    summary = {
        "n_input_rows": int(len(input_df)),
        "n_parent_positives": int(len(parents)),
        "n_queries": int(len(manifest)),
        "by_generator": {
            str(k): int(v)
            for k, v in manifest["generator"].value_counts().items()
        },
        "failures": {str(k): int(v) for k, v in failures.items()},
        "manifest": cfg.output_manifest,
        "fasta": cfg.output_fasta,
    }
    summary_path = f"{cfg.output_manifest}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("generated %d queries from %d parents",
                len(manifest), len(parents))
    return summary


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _raw_column(df: pd.DataFrame) -> str | None:
    for column in ("raw_title1", "raw_title", "Run"):
        if column in df:
            return column
    return None


def _check_heldout_disjoint(
    train_frames: Sequence[pd.DataFrame],
    heldout: pd.DataFrame,
    require_heldout: bool,
) -> dict:
    if heldout.empty:
        if require_heldout:
            raise ValueError(
                "immutable heldout_features are required; set "
                "require_heldout=false only for an explicit exploratory run")
        return {"checked": False, "reason": "not configured"}
    held_col = _raw_column(heldout)
    if held_col is None:
        raise ValueError("heldout features have no raw-title column")
    held_raws = set(heldout[held_col].dropna().astype(str))
    train_raws: set[str] = set()
    for frame in train_frames:
        col = _raw_column(frame)
        if col is not None:
            train_raws.update(frame[col].dropna().astype(str))
    overlap = sorted(train_raws & held_raws)
    if overlap:
        raise ValueError(
            f"raw leakage between training and immutable heldout: "
            f"{overlap[:10]}")
    return {
        "checked": True,
        "n_train_raws": len(train_raws),
        "n_heldout_raws": len(held_raws),
    }


def _gold_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "label_type" in out:
        out = out[out["label_type"].eq("negative")]
    elif "label" in out:
        out = out[pd.to_numeric(out["label"], errors="coerce").eq(0)]
    else:
        raise ValueError("gold features require label or label_type")
    return out.reset_index(drop=True)


def _filter_silver_signal(
    df: pd.DataFrame, cfg: AssemblyConfig,
) -> tuple[pd.DataFrame, dict]:
    required = {
        "precursor_light_max_int",
        "precursor_heavy_max_int",
        cfg.light_fragment_column,
        cfg.heavy_fragment_column,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"silver features missing physical-signal columns: {missing}")
    mask = (
        pd.to_numeric(df["precursor_light_max_int"], errors="coerce")
        .gt(cfg.min_light_precursor_intensity)
        & pd.to_numeric(df["precursor_heavy_max_int"], errors="coerce")
        .gt(cfg.min_heavy_precursor_intensity)
        & pd.to_numeric(df[cfg.light_fragment_column], errors="coerce")
        .ge(cfg.min_light_fragments)
        & pd.to_numeric(df[cfg.heavy_fragment_column], errors="coerce")
        .ge(cfg.min_heavy_fragments)
    )
    optional_gates = {}
    if "precursor_xic_empty" in df:
        gate = pd.to_numeric(
            df["precursor_xic_empty"], errors="coerce").fillna(1).eq(0)
        mask &= gate
        optional_gates["precursor_xic_nonempty"] = int(gate.sum())
    if "heavy_in_raw" in df:
        gate = pd.to_numeric(
            df["heavy_in_raw"], errors="coerce").fillna(0).eq(1)
        mask &= gate
        optional_gates["heavy_in_raw"] = int(gate.sum())
    if "heavy_out_of_range" in df:
        gate = pd.to_numeric(
            df["heavy_out_of_range"], errors="coerce").fillna(1).eq(0)
        mask &= gate
        optional_gates["heavy_not_out_of_range"] = int(gate.sum())
    kept = df[mask].copy()
    return kept, {
        "input": int(len(df)),
        "kept": int(len(kept)),
        "optional_gate_pass_counts": optional_gates,
    }


def _exclude_target_like(
    df: pd.DataFrame,
    target: TargetIndex,
    contaminant: TargetIndex | None,
) -> tuple[pd.DataFrame, dict]:
    levels = df["sequence"].fillna("").astype(str).map(
        lambda s: classify_peptide(s, target))
    keep = levels.eq("L4")
    contaminant_levels = None
    if contaminant is not None:
        contaminant_levels = df["sequence"].fillna("").astype(str).map(
            lambda s: classify_peptide(s, contaminant))
        keep &= contaminant_levels.eq("L4")
    out = df[keep].copy()
    report = {
        "input": int(len(df)),
        "kept": int(len(out)),
        "target_levels": {
            str(k): int(v) for k, v in levels.value_counts().items()
        },
    }
    if contaminant_levels is not None:
        report["contaminant_levels"] = {
            str(k): int(v)
            for k, v in contaminant_levels.value_counts().items()
        }
    return out, report


def _filter_gold_domain(
    df: pd.DataFrame,
    target: TargetIndex,
    contaminant: TargetIndex | None,
    labeling: HeavyType,
) -> tuple[pd.DataFrame, dict]:
    """Keep real entrapments but remove physically invalid/ambiguous rows."""
    if df.empty:
        return df.copy(), {"input": 0, "kept": 0}
    clean, exclusion = _exclude_target_like(df, target, contaminant)
    mask = clean["sequence"].fillna("").astype(str).map(
        lambda seq: has_label_site(seq, labeling))
    if "heavy_out_of_range" in clean:
        mask &= pd.to_numeric(
            clean["heavy_out_of_range"], errors="coerce").fillna(1).eq(0)
    elif "heavy_in_raw" in clean:
        mask &= pd.to_numeric(
            clean["heavy_in_raw"], errors="coerce").fillna(0).eq(1)
    out = clean[mask].copy()
    return out, {
        "input": int(len(df)),
        "after_target_exclusion": int(len(clean)),
        "kept": int(len(out)),
        "exclusion": exclusion,
    }


def _numeric_bin(series: pd.Series, width: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if width <= 0:
        return numeric
    return np.floor(numeric / width)


def _distribution_keys(
    df: pd.DataFrame, columns: Sequence[str], cfg: AssemblyConfig,
) -> tuple[pd.Series, list[str]]:
    usable = [c for c in columns if c in df.columns]
    parts: list[pd.Series] = []
    used: list[str] = []
    widths = {
        "precursor_mz": cfg.precursor_mz_bin_width,
        "sequence_len": cfg.sequence_len_bin_width,
        "total_silac_shift": cfg.total_shift_bin_width,
        "rt": cfg.rt_bin_width,
    }
    for column in usable:
        if column in widths:
            part = _numeric_bin(df[column], widths[column])
        else:
            part = df[column]
        parts.append(part.fillna("__NA__").astype(str))
        used.append(column)
    if not parts:
        return pd.Series(["__ALL__"] * len(df), index=df.index), []
    keys = parts[0]
    for part in parts[1:]:
        keys = keys.str.cat(part, sep="|")
    return keys, used


def _match_silver_distribution(
    positives: pd.DataFrame,
    silver: pd.DataFrame,
    cfg: AssemblyConfig,
) -> tuple[pd.DataFrame, dict]:
    if silver.empty:
        return silver.copy(), {
            "input": 0, "kept": 0, "used_columns": []}
    common_columns = [
        c for c in cfg.distribution_columns
        if c in positives.columns and c in silver.columns
    ]
    pos_keys, used = _distribution_keys(positives, common_columns, cfg)
    sil_keys, _ = _distribution_keys(silver, common_columns, cfg)
    pos_counts = pos_keys.value_counts().to_dict()
    work = silver.copy()
    work["_dist_key"] = sil_keys
    rng = np.random.default_rng(cfg.seed)
    selected: list[pd.DataFrame] = []
    by_source = {}
    for source, source_df in work.groupby("negative_source", sort=True):
        source_parts: list[pd.DataFrame] = []
        for key, bucket in source_df.groupby("_dist_key", sort=False):
            n_pos = int(pos_counts.get(key, 0))
            if n_pos <= 0:
                continue
            quota = int(math.ceil(
                n_pos * cfg.silver_per_positive_per_source))
            if quota <= 0:
                continue
            if len(bucket) > quota:
                take = rng.choice(bucket.index.to_numpy(),
                                  size=quota, replace=False)
                bucket = bucket.loc[take]
            source_parts.append(bucket)
        kept = (pd.concat(source_parts, ignore_index=False)
                if source_parts else source_df.iloc[0:0])
        selected.append(kept)
        by_source[str(source)] = {
            "input": int(len(source_df)),
            "kept": int(len(kept)),
        }
    out = (pd.concat(selected, ignore_index=True)
           if selected else work.iloc[0:0].copy())
    return out.drop(columns=["_dist_key"], errors="ignore"), {
        "input": int(len(silver)),
        "kept": int(len(out)),
        "used_columns": used,
        "by_source": by_source,
    }


def _metadata_audit(
    positives: pd.DataFrame,
    silver: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
) -> dict:
    common = [
        c for c in columns
        if c in positives.columns and c in silver.columns
        and pd.api.types.is_numeric_dtype(
            pd.concat([positives[c], silver[c]], ignore_index=True))
    ]
    smd = {}
    for column in common:
        p = pd.to_numeric(positives[column], errors="coerce")
        n = pd.to_numeric(silver[column], errors="coerce")
        pooled = math.sqrt(
            (float(p.var(ddof=1) or 0) + float(n.var(ddof=1) or 0)) / 2)
        smd[column] = (
            float((p.mean() - n.mean()) / pooled)
            if pooled > 0 else 0.0)

    signature_auc = None
    if len(positives) >= 10 and len(silver) >= 10 and common:
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            sample = pd.concat(
                [positives[common].assign(_source=0),
                 silver[common].assign(_source=1)],
                ignore_index=True)
            X = sample[common].replace([np.inf, -np.inf], np.nan)
            y = sample["_source"].to_numpy()
            n_splits = min(5, int(np.bincount(y).min()))
            if n_splits >= 2:
                oof = np.full(len(sample), np.nan)
                splitter = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed)
                for tr, te in splitter.split(X, y):
                    model = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(max_iter=1000,
                                           random_state=seed),
                    )
                    model.fit(X.iloc[tr], y[tr])
                    oof[te] = model.predict_proba(X.iloc[te])[:, 1]
                signature_auc = float(roc_auc_score(y, oof))
        except Exception as exc:  # audit must not destroy a valid dataset
            logger.warning("generator signature audit skipped: %s", exc)
    return {
        "columns": common,
        "standardized_mean_difference": smd,
        "generator_signature_auc": signature_auc,
        "interpretation": (
            "AUC near 0.5 means metadata alone cannot identify synthetic "
            "negatives; a high AUC indicates a generator shortcut."),
    }


def _annotate_sources(
    positives: pd.DataFrame,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = positives.copy()
    p["label"] = 1
    p["label_type"] = "positive"
    p["negative_source"] = SOURCE_POSITIVE
    p["negative_confidence"] = "gold"
    p["parent_id"] = [
        _stable_id(seq, charge, prefix="P")
        for seq, charge in zip(p["sequence"], p["charge"])
    ]
    p["group_id"] = p["parent_id"]

    g = gold.copy()
    if not g.empty:
        g["label"] = 0
        g["label_type"] = "negative"
        g["negative_source"] = SOURCE_GOLD
        g["negative_confidence"] = "gold"
        g["parent_id"] = pd.NA
        g["group_id"] = [
            _stable_id(seq, charge, raw, prefix="G")
            for seq, charge, raw in zip(
                g["sequence"], g["charge"],
                g[_raw_column(g)] if _raw_column(g) else [""] * len(g))
        ]

    s = silver.copy()
    if not s.empty:
        s["label"] = 0
        s["label_type"] = "negative"
        s["negative_confidence"] = "silver"
        s["group_id"] = s["parent_id"]
    return p, g, s


def _deduplicate_training(df: pd.DataFrame) -> pd.DataFrame:
    raw_col = _raw_column(df)
    keys = ["sequence", "charge"]
    if raw_col:
        keys.append(raw_col)
    work = df.copy()
    priority = {
        SOURCE_POSITIVE: 0,
        SOURCE_GOLD: 1,
        SOURCE_SHUFFLE: 2,
        SOURCE_MARKOV: 3,
    }
    work["_source_priority"] = work["negative_source"].map(
        priority).fillna(99)
    work = (work.sort_values("_source_priority")
            .drop_duplicates(keys, keep="first")
            .drop(columns="_source_priority"))
    return work.reset_index(drop=True)


def assemble_training_set(cfg: AssemblyConfig) -> dict:
    """Assemble independent PSM rows into a leak-audited training CSV."""
    positives_raw = _read_many(cfg.positive_features)
    gold_raw = _read_many(cfg.gold_features)
    silver_raw = _read_many(cfg.silver_features)
    heldout = _read_many(cfg.heldout_features)

    _validate_supported_modifications((
        ("positive features", positives_raw),
        ("Gold features", gold_raw),
        ("Silver features", silver_raw),
    ), cfg.labeling)
    positives = _positive_rows(
        positives_raw, cfg.confirmation_column, cfg.require_confirmation)
    gold = _gold_rows(gold_raw)

    split_report = _check_heldout_disjoint(
        [positives, gold, silver_raw], heldout, cfg.require_heldout)

    manifest = pd.read_csv(cfg.query_manifest, sep="\t")
    required_manifest = {
        "query_id", "parent_id", "sequence", "charge",
        "generator", "negative_source",
    }
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(
            f"query manifest missing columns: {missing_manifest}")
    for frame, name in ((silver_raw, "silver features"),
                        (manifest, "query manifest")):
        if not {"sequence", "charge"} <= set(frame.columns):
            raise ValueError(f"{name} requires sequence and charge")
    silver = silver_raw.merge(
        manifest[list(required_manifest)
                 + [c for c in ("generator_seed",)
                    if c in manifest.columns]],
        on=["sequence", "charge"],
        how="inner",
        suffixes=("", "_manifest"),
        validate="many_to_one",
    )
    # The manifest is authoritative for provenance.
    for column in ("query_id", "parent_id", "generator",
                   "negative_source", "generator_seed"):
        manifest_col = f"{column}_manifest"
        if manifest_col in silver:
            silver[column] = silver[manifest_col]
            silver.drop(columns=manifest_col, inplace=True)

    silver_signal, signal_report = _filter_silver_signal(silver, cfg)
    target = load_target_fasta(cfg.target_fasta)
    contaminant = (load_target_fasta(
        cfg.contaminant_fasta, log_label="contaminant FASTA")
        if cfg.contaminant_fasta else None)
    silver_clean, exclusion_report = _exclude_target_like(
        silver_signal, target, contaminant)
    gold, gold_domain_report = _filter_gold_domain(
        gold, target, contaminant, cfg.labeling)
    silver_matched, distribution_report = _match_silver_distribution(
        positives, silver_clean, cfg)

    positives, gold, silver_matched = _annotate_sources(
        positives, gold, silver_matched)
    audit = _metadata_audit(
        positives, silver_matched, cfg.distribution_columns, cfg.seed)
    combined = _deduplicate_training(pd.concat(
        [positives, gold, silver_matched],
        ignore_index=True, sort=False))

    _ensure_parent(cfg.output_features)
    _ensure_parent(cfg.output_audit)
    combined.to_csv(cfg.output_features, index=False)
    summary = {
        "counts": {
            "positive": int((combined["label"] == 1).sum()),
            "negative": int((combined["label"] == 0).sum()),
            "by_source": {
                str(k): int(v)
                for k, v in combined["negative_source"].value_counts().items()
            },
        },
        "heldout": split_report,
        "silver_join": {
            "input": int(len(silver_raw)),
            "matched_manifest": int(len(silver)),
        },
        "silver_signal_filter": signal_report,
        "gold_domain_filter": gold_domain_report,
        "target_exclusion": exclusion_report,
        "distribution_matching": distribution_report,
        "metadata_generator_audit": audit,
        "output_features": cfg.output_features,
    }
    with open(cfg.output_audit, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("assembled training set: %s (%d rows)",
                cfg.output_features, len(combined))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic DIA query + hard-negative training-set builder")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "assemble"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--logpath", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.logpath:
        _ensure_parent(args.logpath)
        handlers.append(logging.FileHandler(args.logpath, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    if args.command == "generate":
        summary = generate_queries(load_query_config(args.config))
    else:
        summary = assemble_training_set(load_assembly_config(args.config))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
