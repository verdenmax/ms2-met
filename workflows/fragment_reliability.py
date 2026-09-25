"""Reliability of sequence cuts in the original, deduplicated QDS main group.

Remove each trace's strongest common light/heavy acquisition cycle and apply
the ordinary pairing rules again. This describes evidence sensitivity; it
does not change observations, labels, or select a new chromatographic group.
"""
import numpy as np

from workflows.q1a_helpers import is_signal_present_light, is_signal_present_heavy


VERSION = 'r_v1'
FEATURE_NAMES = (
    'ms2_reliability_stable_cut_fraction',
    'ms2_reliability_stable_longest_gap_fraction',
)
STATUS_COLUMNS = (
    'fragment_reliability_valid', 'fragment_reliability_version',
    'fragment_reliability_status',
)


def unavailable_features(reason):
    return {**dict.fromkeys(FEATURE_NAMES, float('nan')),
            'fragment_reliability_valid': 0,
            'fragment_reliability_version': VERSION,
            'fragment_reliability_status': reason}


def _stable_pair(light, heavy):
    """Preserve actual cycle coordinates, including gaps after removal."""
    common, li, hi = np.intersect1d(
        light['cycle_idx'], heavy['cycle_idx'], return_indices=True)
    if not len(common) or not (is_signal_present_light(light)
                              and is_signal_present_light(heavy)):
        return False
    joint = ((light['intensity'][li] / light['intensity'].max())
             * (heavy['intensity'][hi] / heavy['intensity'].max()))
    # intersect1d sorts cycles; argmax breaks ties at the earliest cycle.
    strongest = common[int(np.argmax(joint))]
    remaining_light = light[light['cycle_idx'] != strongest]
    remaining_heavy = heavy[heavy['cycle_idx'] != strongest]
    return bool(is_signal_present_light(remaining_light)
                and is_signal_present_heavy(remaining_light, remaining_heavy))


def reliability_features(sequence_length, main_groups):
    """Consume validated QDS groups, after exact-peak dedup and ordinal filtering.

An ambiguous alias group never creates multiple cuts. Multiple independent
groups supporting one cut count once, and that cut survives if any does.
Acquired but empty/ordinal-one-only evidence yields zero cuts and gap one.
"""
    if sequence_length < 2:
        raise ValueError('Reliability requires a sequence of length >= 2')
    cuts = set()
    for group in main_groups:
        possible = {p['fragment'].ordinal if p['fragment'].ion_type == 'b'
                    else sequence_length - p['fragment'].ordinal for p in group}
        if len(possible) != 1:
            continue
        representative = group[0]
        if _stable_pair(representative['light'], representative['heavy']):
            cuts.update(possible)
    boundaries = sorted({0, sequence_length} | cuts)
    return {
        FEATURE_NAMES[0]: len(cuts) / (sequence_length - 1),
        FEATURE_NAMES[1]: max(b-a for a, b in zip(boundaries, boundaries[1:])) / sequence_length,
        'fragment_reliability_valid': 1,
        'fragment_reliability_version': VERSION,
        'fragment_reliability_status': 'ok',
    }
