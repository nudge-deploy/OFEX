"""Evaluation — comparison of system records with the reference.

Unit: one obligation instance. A record matches a reference obligation when both are in the same provision and their
character spans have intersection-over-union ≥ 0.5; matching is one-to-one, highest overlap first. Precision, recall,
and F1 are micro-averaged over provisions and then averaged over runs; 95% percentile bootstrap over provisions."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
import numpy as np
from .config import IOU_THRESHOLD, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL, FIELDS
from .text import norm_text, null_norm


def span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_spans(gold: list[dict], pred: list[dict], threshold: float = IOU_THRESHOLD) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching within one provision; pairs ranked by IoU (ties: gold index, then prediction index)."""
    cands = sorted((-span_iou((g["start"], g["end"]), (p["start"], p["end"])), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)
                   if span_iou((g["start"], g["end"]), (p["start"], p["end"])) >= threshold)
    used_g, used_p, out = set(), set(), []
    for neg, gi, pi in cands:
        if gi in used_g or pi in used_p: continue
        used_g.add(gi); used_p.add(pi); out.append((gi, pi, -neg))
    return out


def by_provision(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows: out[r["provision_id"]].append(r)
    return out


def fval(obj: dict, field: str):
    return null_norm((obj.get("fields") or {}).get(field))


def _overlap(a, b): return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def classify_fp(pred, ref, matched_ref) -> str:
    if not ref: return "no_reference_obligation_in_provision"
    length = max(1, pred["end"] - pred["start"])
    over = [(_overlap(pred, g), g) for g in ref if _overlap(pred, g) > 0]
    if not over: return "span_outside_all_reference_obligations"
    best, g = max(over, key=lambda x: x[0])
    if sum(1 for o, _ in over if o > 0) >= 2 and best / length < 0.9: return "merges_several_reference_obligations"
    if g in matched_ref:
        if span_iou((pred["start"], pred["end"]), (g["start"], g["end"])) >= IOU_THRESHOLD: return "duplicate_of_matched_reference"
        return "fragment_of_matched_reference" if best / length >= 0.9 else "partial_overlap_with_matched_reference"
    return "fragment_of_unmatched_reference" if best / length >= 0.9 else "partial_overlap_with_unmatched_reference"


def classify_fn(ref, pred) -> str:
    length = max(1, ref["end"] - ref["start"])
    covered = [_overlap(ref, p) for p in pred if _overlap(ref, p) > 0]
    if not covered: return "not_extracted"
    if len(covered) >= 2 and sum(covered) / length >= 0.5: return "split_into_several_predictions"
    if any(span_iou((ref["start"], ref["end"]), (p["start"], p["end"])) >= IOU_THRESHOLD for p in pred): return "lost_in_one_to_one_matching"
    return "span_too_short_or_too_long"


class RunScore:
    """Counts of one run over an ordered list of provisions."""
    def __init__(self, ids: list[str], ref_by: dict, pred_by: dict):
        self.ids = ids; n = len(ids)
        self.tp, self.npred, self.nref = np.zeros(n), np.zeros(n), np.zeros(n)
        self.pairs, self.detected, self.fp, self.fn = [], {}, [], []
        for i, pid in enumerate(ids):
            ref, pred = ref_by.get(pid, []), pred_by.get(pid, [])
            ms = match_spans(ref, pred)
            self.tp[i], self.npred[i], self.nref[i] = len(ms), len(pred), len(ref)
            mg, mp = {g for g, _, _ in ms}, {p for _, p, _ in ms}
            for g, p, _ in ms: self.pairs.append((i, ref[g], pred[p]))
            for g, obj in enumerate(ref):
                self.detected[obj["gold_id"]] = g in mg
                if g not in mg: self.fn.append((i, classify_fn(obj, pred), obj))
            for p, obj in enumerate(pred):
                if p not in mp: self.fp.append((i, classify_fp(obj, ref, [ref[g] for g in mg]), obj))

    def prf(self, idx=None):
        idx = slice(None) if idx is None else idx
        tp, n_pred, n_ref = self.tp[idx].sum(), self.npred[idx].sum(), self.nref[idx].sum()
        p = tp / n_pred if n_pred else float("nan"); r = tp / n_ref if n_ref else float("nan")
        f = 2 * tp / (n_pred + n_ref) if (n_pred + n_ref) else float("nan")
        return p, r, f


def bootstrap_ci(n_items: int, stat, B: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED, level: float = CONFIDENCE_LEVEL):
    if n_items == 0: return float("nan"), float("nan")
    rng = np.random.default_rng(seed); vals = []
    for _ in range(B):
        v = stat(rng.integers(0, n_items, n_items))
        if not (v is None or (isinstance(v, float) and math.isnan(v))): vals.append(v)
    if not vals: return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def mean_over_runs(runs, fn) -> float:
    vals = [v for v in (fn(r) for r in runs) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def prf_with_ci(runs: list[RunScore], idx_filter=None) -> dict:
    n = len(runs[0].ids); base = np.arange(n) if idx_filter is None else np.array(idx_filter); out = {}
    for k, name in enumerate(("precision", "recall", "f1")):
        stat = lambda idx, k=k: mean_over_runs(runs, lambda r: r.prf(base[idx])[k])
        lo, hi = bootstrap_ci(len(base), stat)
        out[name] = {"value": stat(np.arange(len(base))), "ci_lo": lo, "ci_hi": hi}
    out["n_pred_mean"] = float(np.mean([r.npred[base].sum() for r in runs])); out["n_ref"] = int(runs[0].nref[base].sum())
    out["tp_mean"] = float(np.mean([r.tp[base].sum() for r in runs]))
    return out


def paired_difference(a: list[RunScore], b: list[RunScore]) -> dict:
    """Paired bootstrap over provisions of a − b for precision, recall, F1."""
    n = len(a[0].ids); out = {}
    for k, lab in enumerate(("precision", "recall", "f1")):
        diff = lambda idx: mean_over_runs(a, lambda r: r.prf(idx)[k]) - mean_over_runs(b, lambda r: r.prf(idx)[k])
        rng = np.random.default_rng(BOOTSTRAP_SEED); vals = np.array([diff(rng.integers(0, n, n)) for _ in range(BOOTSTRAP_RESAMPLES)]); vals = vals[~np.isnan(vals)]
        out[lab] = {"diff": float(diff(np.arange(n))), "ci_lo": float(np.quantile(vals, 0.025)), "ci_hi": float(np.quantile(vals, 0.975))}
    return out


def mcnemar(a: list[RunScore], b: list[RunScore]) -> dict:
    """Exact McNemar test on detection of each reference obligation (detected = by at least half of the runs)."""
    maj = lambda runs: {k: sum(r.detected[k] for r in runs) * 2 >= len(runs) for k in runs[0].detected}
    da, db = maj(a), maj(b); keys = sorted(da)
    x = sum(1 for k in keys if da[k] and not db[k]); y = sum(1 for k in keys if db[k] and not da[k]); n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return {"gained": x, "lost": y, "p": p}


def field_scores(runs: list[RunScore]) -> dict:
    """Exact match on matched pairs; overall and where the reference has a value."""
    out = {}
    for f in FIELDS:
        c = Counter()
        for r in runs:
            for _, g, p in r.pairs:
                gv, pv = fval(g, f), fval(p, f); ex = norm_text(gv) == norm_text(pv)
                c["n"] += 1; c["exact"] += ex
                if gv is not None: c["pos"] += 1; c["pos_exact"] += ex
        out[f] = {"exact": c["exact"] / c["n"] if c["n"] else None, "positive_case_exact": c["pos_exact"] / c["pos"] if c["pos"] else None, "positive_n_per_run": c["pos"] / len(runs)}
    return out


def error_taxonomy(runs: list[RunScore]) -> dict:
    tax = Counter()
    for r in runs:
        for _, k, _ in r.fp: tax["FP: " + k] += 1 / len(runs)
        for _, k, _ in r.fn: tax["FN: " + k] += 1 / len(runs)
    return {k: round(v, 2) for k, v in sorted(tax.items())}


def cohen_kappa(a: list, b: list) -> float:
    n = len(a); labels = sorted(set(a) | set(b), key=str); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1 else (1.0 if po == 1 else float("nan"))
