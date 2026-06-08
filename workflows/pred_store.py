"""Peptide identity normalization + streaming peptide->prediction lookup.

Builds an in-memory {normalize_key -> predictions} store by scanning the
spectral library once and keeping only the wanted (identified) peptides.
See docs/specs/2026-06-08-speclib-predicted-intensity-features-design.md §4.1.
"""
import logging

logger = logging.getLogger(__name__)


def _as_pairs(mods):
    """Yield (pos:int, mod_id:int) from either (pos, mod_id) tuples or
    ModSite objects (with .pos / .mod_id)."""
    for m in mods:
        if hasattr(m, "pos") and hasattr(m, "mod_id"):
            yield int(m.pos), int(m.mod_id)
        else:
            pos, mid = m
            yield int(pos), int(mid)


def normalize_mods(mods) -> tuple:
    """Canonical, hashable, position-sorted modification tuple."""
    return tuple(sorted(_as_pairs(mods)))


def normalize_key(sequence, mods, charge) -> tuple:
    """Canonical peptide-variant key: (sequence, sorted-mods, int charge)."""
    return (sequence, normalize_mods(mods), int(charge))


def frag_key(ion_type, frag_pos, frag_charge) -> tuple:
    """Canonical fragment key shared by predicted and observed sides."""
    return (str(ion_type), int(frag_pos), int(frag_charge))


class PredStore:
    """In-memory {normalize_key -> {'frags': {frag_key: intensity},
    'pred_rt': float}} for the identified peptides only."""

    def __init__(self):
        self._d = {}
        self.wanted = set()
        self.n_hit = 0
        self.n_miss = 0

    def get(self, key):
        return self._d.get(key)


def _frag_map(frag_ions) -> dict:
    """Build {frag_key: intensity} from a list of FragIon (objects mode)."""
    out = {}
    for fi in frag_ions:
        out[frag_key(fi.ion_type, fi.frag_pos, fi.frag_charge)] = float(fi.intensity)
    return out


def build_pred_store(lib, wanted_keys, decode_ms2: str = "objects") -> PredStore:
    """Scan `lib` once; keep predictions only for peptides in `wanted_keys`
    (a set of normalize_key tuples). Memory is O(hits)."""
    store = PredStore()
    store.wanted = set(wanted_keys)

    want_by_id = {}
    for (seq, norm_mods, chg) in store.wanted:
        want_by_id.setdefault((seq, norm_mods), set()).add(chg)

    for pep in lib.iter_peptides(decode_ms2=decode_ms2):
        pid = (pep.sequence, normalize_mods(pep.mods))
        charges = want_by_id.get(pid)
        if not charges:
            continue
        for chg in charges:
            frags = pep.pred_ms2.get(chg)
            if frags is None:
                logger.debug("pred_store: %s matched but charge %s missing",
                             pid, chg)
                continue
            store._d[(pep.sequence, normalize_mods(pep.mods), chg)] = {
                "frags": _frag_map(frags),
                "pred_rt": float(pep.pred_rt) if pep.pred_rt is not None else float("nan"),
            }

    store.n_hit = len(store._d)
    store.n_miss = len(store.wanted) - store.n_hit
    logger.info("pred_store coverage: hit=%d miss=%d", store.n_hit, store.n_miss)
    return store
