"""Sanity gate (spec §4.0): does the library's predicted fragment intensity
agree with the observed *light* spectrum on confident PSMs? Run this BEFORE
building any predicted-intensity features — if it fails, fix alignment /
units / mod mapping first.

The pure core (similarity_distribution, gate_pass, build_pairs_from_maps) is
unit-tested; main() wires it to a real library + raw for the manual go/no-go
run.
"""
import argparse
import configparser
import logging
import os
import tempfile

import numpy as np

from spectrum.speclib import SpecLib
from spectrum.psm_info import HeavyType
from manager.light_result_manager import LightResultManager
from manager import data_manager
from workflows.pred_features import spectral_angle, spearman_sim
from workflows.pred_store import (
    build_pred_store, normalize_key, frag_key, frag_pos_for_ion)

logger = logging.getLogger(__name__)

_METRICS = {"spectral_angle": spectral_angle, "spearman": spearman_sim}


def similarity_distribution(pairs, metric=spectral_angle) -> dict:
    """pairs: iterable of (pred_vec, obs_vec) aligned over a fragment set.
    Returns {n, median, p25, p75} over finite similarities."""
    sims = []
    for pred_vec, obs_vec in pairs:
        s = metric(pred_vec, obs_vec)
        if np.isfinite(s):
            sims.append(float(s))
    if not sims:
        logger.debug("similarity_distribution: no finite similarities")
        return {"n": 0, "median": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    arr = np.asarray(sims, dtype=float)
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def gate_pass(stats, min_sim) -> bool:
    """Gate passes iff we have data and the median similarity clears min_sim."""
    return bool(stats["n"] > 0 and np.isfinite(stats["median"])
                and stats["median"] > min_sim)


def build_pairs_from_maps(pred_map: dict, obs_map: dict):
    """Align a predicted {frag_key: intensity} map with an observed
    {frag_key: intensity} map on their common fragments; return
    (pred_vec, obs_vec) as parallel lists in a stable fragment order."""
    common = sorted(set(pred_map) & set(obs_map))
    pred_vec = [pred_map[k] for k in common]
    obs_vec = [obs_map[k] for k in common]
    return pred_vec, obs_vec


def _observed_light_map(psm, dia_data, xic_cycle_window, mass_tol_ppm) -> dict:
    """{frag_key: apex_intensity} for the PSM's light b/y fragments
    (singly-charged, matching the dominant predicted fragments).

    The b/y ordinal from get_fragment_ions is mapped to the library's
    0-indexed cleavage site via frag_pos_for_ion (y is reversed) so observed
    and predicted fragments key identically.
    """
    out = {}
    seq_len = len(psm._sequence)
    # Only the light masses are consumed below; they are chemistry-independent.
    # SILAC is intentional here to avoid imposing the uniform-label PTM guard
    # on this light-spectrum-only diagnostic.
    b_ions, y_ions = psm.get_fragment_ions(HeavyType.SILAC)
    for ion_type, ion_num, light_mass, _heavy_mass in (b_ions + y_ions):
        xic, _all = dia_data.xic_ms2_peaks_extract(
            psm._rt, xic_cycle_window,
            precursor_mz=psm._precursor_mz,
            ions_mass=light_mass, mass_tol_ppm=mass_tol_ppm)
        if xic is None or len(xic) == 0:
            continue
        if not np.any(np.isfinite(xic["intensity"])):
            continue
        apex = float(np.nanmax(xic["intensity"]))
        if apex > 0:
            frag_pos = frag_pos_for_ion(ion_type, ion_num, seq_len)
            out[frag_key(ion_type, frag_pos, 1)] = apex
    return out


def filter_psms_by_raw(psms, raw_title):
    """Keep only PSMs whose _raw_title matches raw_title. If raw_title is None,
    return the list unchanged. Lets the gate process a single raw at a time
    (memory-frugal) while the PSM file may span many raws."""
    if raw_title is None:
        return list(psms)
    return [p for p in psms if p._raw_title == raw_title]


def main():
    parser = argparse.ArgumentParser(
        description="Speclib sanity gate: predicted vs observed light similarity")
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--mod", required=True)
    parser.add_argument("--psm-file", required=True,
                        help="confident light PSMs (same loader as main pipeline)")
    parser.add_argument("--search-engine-type", type=int, default=3)
    parser.add_argument("--raw", default=None,
                        help="DIA mzML for observed light (parsed into RAM; "
                             "heavy). Prefer --dia-npz when a cache exists.")
    parser.add_argument("--dia-npz", default=None,
                        help="prebuilt .dia.npz; mmap-loaded (low memory). "
                             "Overrides --raw when given.")
    parser.add_argument("--raw-title", default=None,
                        help="only score PSMs whose raw_title matches (so one "
                             "raw is processed at a time)")
    parser.add_argument("--metric", choices=list(_METRICS), default="spectral_angle")
    parser.add_argument("--min-sim", type=float, default=0.7)
    parser.add_argument("--mass-tol-ppm", type=float, default=10.0)
    parser.add_argument("--xic-cycle-window", type=int, default=6)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.raw and not args.dia_npz:
        parser.error("provide either --dia-npz (preferred, mmap) or --raw")

    cfg = configparser.ConfigParser()
    cfg["input"] = {
        "search_engine_type": str(args.search_engine_type),
        "light_result_file": args.psm_file,
        "pfind_qvalue_threshold": "0.01",
    }
    cfg["general"] = {
        "mass_tol_ppm": str(args.mass_tol_ppm),
        "xic_cycle_window": str(args.xic_cycle_window),
        "centroid_enabled": "true",
        "centroid_rel_threshold": "0.001",
    }

    lib = SpecLib.open_dir(args.library_dir, fasta_path=args.fasta,
                           mod_path=args.mod)

    lrm = LightResultManager(config=cfg)
    light_result = lrm.get_light_result_object(args.psm_file)
    psms = filter_psms_by_raw(list(light_result.psm_info), args.raw_title)
    psms = psms[:args.limit]
    logging.info("PSMs after raw_title=%s filter + limit: %d",
                 args.raw_title, len(psms))

    wanted = {normalize_key(p._sequence, p._modify, p._charge) for p in psms}
    store = build_pred_store(lib, wanted)
    logging.info("speclib coverage: hit=%d miss=%d", store.n_hit, store.n_miss)

    if args.dia_npz:
        from spectrum.dia_data import DIAData
        logging.info("loading DIA via mmap npz: %s", args.dia_npz)
        dia_data = DIAData.load_from_file(args.dia_npz, use_mmap=True)
    else:
        tmp_pickle = os.path.join(tempfile.mkdtemp(), "raw_manager.pkl")
        dm = data_manager.DataManager(cfg, path=tmp_pickle)
        dia_data = dm.get_dia_data_object(args.raw)

    pairs = []
    for p in psms:
        rec = store.get(normalize_key(p._sequence, p._modify, p._charge))
        if rec is None:
            continue
        obs_map = _observed_light_map(p, dia_data, args.xic_cycle_window,
                                      args.mass_tol_ppm)
        pred_vec, obs_vec = build_pairs_from_maps(rec["frags"], obs_map)
        if len(pred_vec) >= 2:
            pairs.append((pred_vec, obs_vec))

    stats = similarity_distribution(pairs, metric=_METRICS[args.metric])
    passed = gate_pass(stats, args.min_sim)
    logging.info("sanity stats: %s", stats)
    logging.info("GATE %s (min_sim=%.2f, metric=%s)",
                 "PASS" if passed else "FAIL", args.min_sim, args.metric)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
