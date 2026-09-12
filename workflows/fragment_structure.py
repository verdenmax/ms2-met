"""Candidate-local fragment evidence; independent of labels and other PSMs.

Q: same-charge pairing. D: exact observed-trace deduplication. S: sequence
positions supported by a common light AND heavy chromatographic peak group.
Existing pooled features and their Q1A definitions are not replaced.
"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from workflows.q1a_helpers import (
    is_separable_fragment, is_signal_present_heavy,
    is_signal_present_light, xic_pair_pearson,
)
from spectrum.dia_data import pool_fragment_charges


VERSION = 'qds_v1'
CHARGE_FEATURES = (
    'ms2_charge_paired_target_count', 'ms2_charge_paired_ion_count',
    'ms2_charge_b_paired_ion_count', 'ms2_charge_y_paired_ion_count',
    'ms2_charge_shifted_paired_ion_count', 'ms2_charge_paired_fraction',
    'ms2_charge_pooled_only_fraction', 'ms2_charge_dominant_mismatch_fraction',
    'ms2_charge_z1_pearson_median', 'ms2_charge_z2_pearson_median',
    'ms2_charge_z1_log_lh_mad', 'ms2_charge_z2_log_lh_mad',
    'ms2_charge_paired_effective_points_median',
)
DEDUP_FEATURES = (
    'ms2_dedup_paired_trace_count', 'ms2_dedup_reused_target_fraction',
    'ms2_dedup_reused_intensity_fraction',
)
STRUCTURE_FEATURES = (
    'ms2_structure_main_trace_count', 'ms2_structure_main_trace_fraction',
    'ms2_structure_main_shifted_trace_count', 'ms2_structure_main_cut_fraction',
    'ms2_structure_main_b_run_fraction', 'ms2_structure_main_y_run_fraction',
    'ms2_structure_main_complementary_cut_fraction',
    'ms2_structure_main_longest_gap_fraction',
    'ms2_structure_main_internal_kr_bracket_fraction',
    'ms2_structure_main_anchor_offset_cycles',
    'ms2_structure_outside_main_intensity_fraction',
    'ms2_structure_main_ambiguous_cut_fraction',
)
FEATURE_NAMES = CHARGE_FEATURES + DEDUP_FEATURES + STRUCTURE_FEATURES


@dataclass(frozen=True)
class FragmentEvidence:
    ion_type: str
    ordinal: int
    light_mass: float
    heavy_mass: float
    light: dict[int, np.ndarray]
    heavy: dict[int, np.ndarray]


def unavailable_features(reason: str) -> dict:
    return {**dict.fromkeys(FEATURE_NAMES, float('nan')),
            'fragment_structure_valid': 0,
            'fragment_structure_version': VERSION,
            'fragment_structure_status': reason}


def _fraction(a, b):
    return float(a / b) if b else float('nan')


def _median(values):
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float('nan')


def _mad(values):
    finite = np.asarray([v for v in values if np.isfinite(v)])
    return float(np.median(abs(finite - np.median(finite)))) if len(finite) >= 3 else float('nan')


def _effective_points(x):
    y = np.asarray(x['intensity'], dtype='f8')
    return _fraction(y.sum() ** 2, np.square(y).sum())


def _signature(x):
    # Equality of theoretical masses or intensity numbers is insufficient.
    # Both sides must use exactly the same positive centroid sets, in the
    # same actual scans, over the entire extraction window. This deliberately
    # does not claim to remove all partial/intermittent peak sharing.
    return tuple((int(scan), tuple(int(i) for i in indices))
                 for scan, indices in x['peak_ids'] if len(indices))


def _peak_interval(x):
    """Connected half-height interval about the dominant sampled apex.

    Half a cycle on each side allows adjacent samples to share a peak group.
    A broad peak retains its full measured interval. No transitive chaining
    of nearby groups is used. Flat background can span a broad interval;
    this is a descriptive representation, not a correctness assertion.
    """
    y = x['intensity']; cycles = x['cycle_idx']
    apex = int(np.argmax(y)); lo = hi = apex
    half = float(y[apex]) * .5
    while lo > 0 and y[lo - 1] >= half and cycles[lo] - cycles[lo - 1] == 1:
        lo -= 1
    while hi + 1 < len(y) and y[hi + 1] >= half and cycles[hi + 1] - cycles[hi] == 1:
        hi += 1
    return float(cycles[lo]) - .5, float(cycles[hi]) + .5


def _longest_run(values):
    best = current = 0; previous = -2
    for v in sorted(values):
        current = current + 1 if v == previous + 1 else 1
        best = max(best, current); previous = v
    return best


def _main_group(groups, center_rt):
    """Largest set with a common point in BOTH peak intervals.

    Interval endpoints cover every possible maximal-overlap set, including
    an intersection between adjacent cycles. Ties favor the candidate RT,
    then the earlier acquisition coordinate. Distinct exact-trace classes
    each get one vote irrespective of intensity or number of ion names.
    """
    intervals = np.asarray([[_peak_interval(g[0]['light']),
                             _peak_interval(g[0]['heavy'])] for g in groups])
    axes = [np.unique(intervals[:, side, :]) for side in (0, 1)]
    cover_l = ((axes[0][None, :] >= intervals[:, 0, 0, None]) &
               (axes[0][None, :] <= intervals[:, 0, 1, None]))
    cover_h = ((axes[1][None, :] >= intervals[:, 1, 0, None]) &
               (axes[1][None, :] <= intervals[:, 1, 1, None]))
    counts = cover_l.astype('i4').T @ cover_h.astype('i4')
    centers = []
    for side in ('light', 'heavy'):
        x = groups[0][0][side]
        centers.append(float(x['cycle_idx'][np.argmin(abs(x['rt'] - center_rt))]))
    points = np.argwhere(counts == counts.max())
    a, b = min(points, key=lambda p:(abs(axes[0][p[0]]-centers[0]) +
                                    abs(axes[1][p[1]]-centers[1]),
                                    axes[0][p[0]], axes[1][p[1]]))
    selected = np.flatnonzero(cover_l[:, a] & cover_h[:, b])
    offset = (abs(axes[0][a]-centers[0]) + abs(axes[1][b]-centers[1])) / 2
    return [groups[i] for i in selected], float(offset)


def fragment_structure_features(
    sequence: str, precursor_charge: int, fragments: list[FragmentEvidence], *,
    split_window: bool | None, center_rt: float, silac: bool = True,
) -> dict:
    """Compute 28 numeric Q/D/S features from a single PSM's observations.

    Only same-run acquisition coordinates are supported. Callers representing
    cross-run pairs must emit ``unavailable_features('cross_run')`` instead.
    Missing acquisition/identity data produces unavailable features; acquired
    but empty signal produces zero evidence counts and undefined ratios.
    """
    n = len(sequence)
    if n < 2 or precursor_charge < 1 or not np.isfinite(center_rt):
        raise ValueError('A sequence, positive precursor charge and finite RT are required')
    if split_window is None:
        return unavailable_features('missing_window')
    eligible = [f for f in fragments if is_separable_fragment(
        f.light_mass, f.heavy_mass, split_window)]
    if not eligible:
        return unavailable_features('no_separable_targets')
    for f in eligible:
        if f.ion_type not in ('b', 'y') or not 1 <= f.ordinal < n:
            raise ValueError('Fragment positions must refer to internal b/y cuts')
        for side in (f.light, f.heavy):
            for z in (1, 2):
                if z not in side or not len(side[z]):
                    return unavailable_features('no_ms2_scans')
                x = side[z]
                if not {'rt','intensity','cycle_idx','peak_ids'} <= set(x.dtype.names or ()):
                    return unavailable_features('missing_peak_identity')
                if (not np.isfinite(x['rt']).all() or not np.isfinite(x['intensity']).all()
                        or (x['intensity'] < 0).any() or (x['cycle_idx'] < 0).any()
                        or (np.diff(x['cycle_idx']) <= 0).any()):
                    return unavailable_features('invalid_acquisition_rows')
    out = unavailable_features('ok')
    out['fragment_structure_valid'] = 1
    paired = []; light_opportunities = 0; pooled_count = pooled_only = 0
    dominant_count = dominant_mismatch = 0
    correlations = {1:[], 2:[]}; ratios = {1:[], 2:[]}
    effective = []; supported = set(); shifted_supported = set()
    for f in eligible:
        name = (f.ion_type, f.ordinal)
        lp, hp = pool_fragment_charges(f.light), pool_fragment_charges(f.heavy)
        pooled_pass = is_signal_present_light(lp) and is_signal_present_heavy(lp, hp)
        pooled_count += int(pooled_pass)
        passed = False; areas_l = {}; areas_h = {}
        for z in (1, 2):
            if z > precursor_charge:
                continue
            l, h = f.light[z], f.heavy[z]
            areas_l[z], areas_h[z] = float(l['intensity'].sum()), float(h['intensity'].sum())
            present_l, present_h = is_signal_present_light(l), is_signal_present_light(h)
            light_opportunities += int(present_l)
            if not (present_l and present_h):
                continue
            correlations[z].append(xic_pair_pearson(l, h))
            if not is_signal_present_heavy(l, h):
                continue
            passed = True; supported.add(name)
            shifted = abs(f.heavy_mass - f.light_mass) >= .001
            if shifted:
                shifted_supported.add(name)
            ratios[z].append(float(np.log2(areas_l[z]/areas_h[z])))
            effective.append(min(_effective_points(l), _effective_points(h)))
            paired.append({'fragment':f, 'charge':z, 'light':l, 'heavy':h,
                           'shifted':shifted, 'weight':areas_l[z]+areas_h[z]})
        pooled_only += int(pooled_pass and not passed)
        if areas_l and max(areas_l.values()) > 0 and max(areas_h.values()) > 0:
            dominant_count += 1
            dominant_mismatch += int(max(areas_l, key=areas_l.get) != max(areas_h, key=areas_h.get))
    out.update({
        'ms2_charge_paired_target_count':len(paired),
        'ms2_charge_paired_ion_count':len(supported),
        'ms2_charge_b_paired_ion_count':sum(t=='b' for t,_ in supported),
        'ms2_charge_y_paired_ion_count':sum(t=='y' for t,_ in supported),
        'ms2_charge_shifted_paired_ion_count':len(shifted_supported),
        'ms2_charge_paired_fraction':_fraction(len(paired),light_opportunities),
        'ms2_charge_pooled_only_fraction':_fraction(pooled_only,pooled_count),
        'ms2_charge_dominant_mismatch_fraction':_fraction(dominant_mismatch,dominant_count),
        'ms2_charge_paired_effective_points_median':_median(effective),
    })
    for z in (1, 2):
        out[f'ms2_charge_z{z}_pearson_median'] = _median(correlations[z])
        out[f'ms2_charge_z{z}_log_lh_mad'] = _mad(ratios[z])
    groups = defaultdict(list)
    for p in paired:
        sig = (_signature(p['light']), _signature(p['heavy']))
        if not sig[0] or not sig[1]:
            return unavailable_features('inconsistent_peak_identity')
        groups[sig].append(p)
    groups = list(groups.values())
    total_weight = sum(p['weight'] for p in paired)
    unique_weight = sum(g[0]['weight'] for g in groups)
    out.update({
        'ms2_dedup_paired_trace_count':len(groups),
        'ms2_dedup_reused_target_fraction':_fraction(len(paired)-len(groups),len(paired)),
        'ms2_dedup_reused_intensity_fraction':_fraction(total_weight-unique_weight,total_weight),
    })
    # Deduplicate BEFORE removing ordinal 1: b1+/b2++ remains ambiguous.
    informative = [g for g in groups if any(p['fragment'].ordinal >= 2 for p in g)]
    main, offset = _main_group(informative, center_rt) if informative else ([],float('nan'))
    bcuts = set(); ycuts = set(); cuts = set(); ambiguous = shifted_count = 0
    for g in main:
        possible = {p['fragment'].ordinal if p['fragment'].ion_type=='b'
                    else n-p['fragment'].ordinal for p in g}
        shifted_count += int(all(p['shifted'] for p in g))
        if len(possible) != 1:
            ambiguous += 1
            continue
        cut = next(iter(possible))
        cuts.add(cut)
        types = {p['fragment'].ion_type for p in g}
        # A single peak equally explained by b and y is not independent
        # complementary support, even when both interpretations share a cut.
        if len(types) == 1:
            (bcuts if next(iter(types))=='b' else ycuts).add(cut)
    boundaries = sorted({0,n} | cuts)
    internal_kr = [i for i,aa in enumerate(sequence,1) if aa in 'KR' and 1<i<n]
    main_weight = sum(g[0]['weight'] for g in main)
    informative_weight = sum(g[0]['weight'] for g in informative)
    out.update({
        'ms2_structure_main_trace_count':len(main),
        'ms2_structure_main_trace_fraction':_fraction(len(main),len(informative)),
        'ms2_structure_main_shifted_trace_count':shifted_count,
        'ms2_structure_main_cut_fraction':len(cuts)/(n-1),
        'ms2_structure_main_b_run_fraction':_longest_run(bcuts)/(n-1),
        'ms2_structure_main_y_run_fraction':_longest_run(ycuts)/(n-1),
        'ms2_structure_main_complementary_cut_fraction':len(bcuts & ycuts)/(n-1),
        'ms2_structure_main_longest_gap_fraction':max(b-a for a,b in zip(boundaries,boundaries[1:]))/n,
        'ms2_structure_main_internal_kr_bracket_fraction':(
            _fraction(sum(i-1 in cuts and i in cuts for i in internal_kr),len(internal_kr))
            if silac else float('nan')),
        'ms2_structure_main_anchor_offset_cycles':offset,
        'ms2_structure_outside_main_intensity_fraction':_fraction(informative_weight-main_weight,informative_weight),
        'ms2_structure_main_ambiguous_cut_fraction':_fraction(ambiguous,len(main)),
    })
    return out
