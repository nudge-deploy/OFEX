"""Marker categories and deterministic stratified sampling (stratum = regulation × marker category).
Proportional allocation by largest remainder, at least one provision per non-empty stratum."""
import random, re
from collections import defaultdict
from fractions import Fraction

_PROHIBITION = re.compile(r"\b(dilarang|tidak boleh|tidak diperbolehkan|tidak diizinkan)\b", re.I)
_OBLIGATION = re.compile(r"\b(wajib|diwajibkan|harus|paling lambat|paling lama)\b", re.I)
_PERMISSION = re.compile(r"\b(dapat|berhak|berwenang|diperbolehkan|diizinkan)\b", re.I)


def category(text: str) -> str:
    """Descriptive marker category used ONLY for stratification. It is not a label."""
    if _PROHIBITION.search(text): return "prohibition"
    if _OBLIGATION.search(text): return "obligation"
    if _PERMISSION.search(text): return "permission"
    return "other"


def largest_remainder(sizes: dict, total: int, minimum: int = 0) -> dict:
    population = sum(sizes.values()); total = min(total, population)
    if total <= 0 or population == 0: return {k: 0 for k in sizes}
    keys = sorted(sizes); alloc = {k: 0 for k in keys}
    if minimum and total >= minimum * sum(1 for k in keys if sizes[k] > 0):
        for k in keys: alloc[k] = min(minimum, sizes[k])
    remaining = total - sum(alloc.values()); capacity = {k: sizes[k] - alloc[k] for k in keys}; cap_total = sum(capacity.values())
    if remaining > 0 and cap_total > 0:
        quotas = {k: Fraction(capacity[k] * remaining, cap_total) for k in keys}
        floors = {k: min(capacity[k], int(quotas[k])) for k in keys}
        for k in keys: alloc[k] += floors[k]
        left = remaining - sum(floors.values())
        order = sorted(keys, key=lambda k: (-(quotas[k] - int(quotas[k])), k))
        while left > 0:
            progressed = False
            for k in order:
                if left == 0: break
                if alloc[k] < sizes[k]: alloc[k] += 1; left -= 1; progressed = True
            if not progressed: break
    return alloc


def stratified_sample(units: list[tuple[str, str]], size: int, seed: int) -> list[str]:
    """units: (provision_id, stratum). Returns the sampled provision ids, sorted."""
    groups = defaultdict(list)
    for pid, stratum in sorted(units): groups[stratum].append(pid)
    alloc = largest_remainder({k: len(v) for k, v in groups.items()}, size, minimum=1)
    rng = random.Random(seed); chosen = []
    for stratum in sorted(groups):
        if alloc[stratum]: chosen.extend(rng.sample(groups[stratum], alloc[stratum]))
    return sorted(chosen)
