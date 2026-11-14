"""Step 5 of OFEX: deduplication within one provision (normalisation of null-like values is done when fields are stored)."""
from .config import CONTAINMENT


def deduplicate(records: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Records of ONE provision, each with start/end. Returns (kept, [(removed, reason)]).
    1. identical spans: keep one (the first in document order);
    2. a span at least 90% inside a longer span of the same provision is removed."""
    kept, removed, seen = [], [], set()
    for p in sorted(records, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if (p["start"], p["end"]) in seen:
            removed.append((p, "identical_span")); continue
        seen.add((p["start"], p["end"])); kept.append(p)
    inside = lambda a, b: (min(a["end"], b["end"]) - max(a["start"], b["start"])) / max(1, a["end"] - a["start"])
    out = []
    for p in kept:
        if any(q is not p and (q["end"] - q["start"]) > (p["end"] - p["start"]) and inside(p, q) >= CONTAINMENT for q in kept):
            removed.append((p, "contained_in_longer_record"))
        else:
            out.append(p)
    return out, removed
