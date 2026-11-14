"""Evaluation: compare OFEX runs with the reference. No model call.
Usage: python scripts/04_evaluate.py --set test --runs test1 test2 test3
       python scripts/04_evaluate.py --set dev  --runs dev
Writes results/<set>_results.json and results/<set>_summary.md."""
import _path, argparse, json
from collections import Counter
import numpy as np
from ofex import evaluate as E
from ofex.config import DATA, RUNS, RESULTS, FIELDS
from ofex.corpus import Corpus
from ofex.extract import final_records
from ofex.io import read_jsonl
from ofex.keyword_baseline import extract as keyword_extract
from ofex.sampling import category

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["dev", "test"]); ap.add_argument("--runs", nargs="+", required=True); a = ap.parse_args()
corpus = Corpus()
prov = read_jsonl(DATA / f"{a.set}_provisions.jsonl"); ids = sorted(p["provision_id"] for p in prov)
ref = E.by_provision(read_jsonl(DATA / f"{a.set}_reference_obligations.jsonl")); nref = sum(len(ref.get(p, [])) for p in ids)
raw = json.loads((DATA / f"{a.set}_annotator_raw_labels.json").read_text())
rows = {r: read_jsonl(RUNS / r / "candidates.jsonl") for r in a.runs}
r4 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)

def summary(runs):
    m = E.prf_with_ci(runs)
    return {"runs": len(runs), "predictions": r4(m["n_pred_mean"]), "TP": r4(m["tp_mean"]), "FP": r4(m["n_pred_mean"] - m["tp_mean"]), "FN": r4(nref - m["tp_mean"]),
            **{k: {"value": r4(m[k]["value"]), "ci": [r4(m[k]["ci_lo"]), r4(m[k]["ci_hi"])]} for k in ("precision", "recall", "f1")}}

res = {"set": a.set, "provisions": len(ids), "reference_obligations": nref, "reference_by_modality": dict(Counter(g["fields"]["modality"] for p in ids for g in ref.get(p, []))),
       "provisions_without_reference": sum(not ref.get(p) for p in ids), "reference_status": "AI-generated (two model annotators + model adjudicator); not expert-validated"}
outputs = {"all_records (primary)": [final_records(rows[r], "all_records") for r in a.runs], "accept_only": [final_records(rows[r], "accept_only") for r in a.runs]}
S = {k: [E.RunScore(ids, ref, E.by_provision(p)) for p in v] for k, v in outputs.items()}
S["keyword baseline"] = [E.RunScore(ids, ref, E.by_provision([x for p in ids for x in keyword_extract(p, corpus.text(p))]))]
res["main"] = {k: summary(v) for k, v in S.items()}
P = S["all_records (primary)"]
res["per_run"] = [{"run": r, **dict(zip(("precision", "recall", "f1"), map(r4, s.prf()))), "TP": int(s.tp.sum()), "FP": int(s.npred.sum() - s.tp.sum()), "FN": int(nref - s.tp.sum())} for r, s in zip(a.runs, P)]
res["source_grounding"] = r4(np.mean([sum(corpus.text(x["provision_id"])[x["start"]:x["end"]] == x["quote"] for x in o) / max(1, len(o)) for o in outputs["all_records (primary)"]]))
res["identical_spans_across_runs"] = r4(np.mean([len({tuple(sorted((x["start"], x["end"]) for x in o if x["provision_id"] == pid)) for o in outputs["all_records (primary)"]}) == 1 for pid in ids])) if len(a.runs) > 1 else None
res["vs_keyword_baseline"] = {**E.paired_difference(P, S["keyword baseline"]), "mcnemar": E.mcnemar(P, S["keyword baseline"])}
# annotator agreement (yardstick)
S_ = lambda o: {"start": o["span_start"], "end": o["span_end"]}
def agree(pred_of, gold_of):
    tp = sum(len(E.match_spans(gold_of(p), pred_of(p))) for p in ids); x = sum(len(pred_of(p)) for p in ids); y = sum(len(gold_of(p)) for p in ids)
    return {"precision": r4(tp / x), "recall": r4(tp / y), "f1": r4(2 * tp / (x + y))}
A = lambda p: [S_(o) for o in raw[p]["a"]]; B = lambda p: [S_(o) for o in raw[p]["b"]]; G = lambda p: ref.get(p, [])
res["annotator_agreement"] = {"A_vs_B": agree(B, A), "A_vs_reference": agree(A, G), "B_vs_reference": agree(B, G),
                              "presence_kappa": r4(E.cohen_kappa([bool(raw[p]["a"]) for p in ids], [bool(raw[p]["b"]) for p in ids])), "adjudicated_provisions": sum(raw[p].get("adjudicated", None) is True for p in ids) if a.set == "test" else None}
# breakdowns
def status(p):
    x, y = raw[p]["a"], raw[p]["b"]; m = E.match_spans([S_(o) for o in x], [S_(o) for o in y])
    if not x and not y: return "agree"
    if bool(x) != bool(y) or len(x) != len(y) or len(m) != len(x): return "disagree"
    return "agree"
groups = {"annotators agree": [i for i, p in enumerate(ids) if status(p) == "agree"], "annotators disagree": [i for i, p in enumerate(ids) if status(p) == "disagree"]}
groups.update({f"marker: {c}": [i for i, p in enumerate(ids) if category(corpus.text(p)) == c] for c in ("obligation", "prohibition", "permission", "other")})
groups.update({f"regulation: {g}": [i for i, p in enumerate(ids) if p.startswith(g + "/")] for g in sorted({p.split('/')[0] for p in ids})})
res["breakdown"] = {k: {"provisions": len(j), **{x: v for x, v in summary_sub.items()}} for k, j in groups.items()
                    for summary_sub in [(lambda m: {"reference": m["n_ref"], "predictions": r4(m["n_pred_mean"]), "TP": r4(m["tp_mean"]), "precision": r4(m["precision"]["value"]), "recall": r4(m["recall"]["value"]), "f1": r4(m["f1"]["value"])})(E.prf_with_ci(P, j))]}
res["recall_by_modality"] = {mod: r4(np.mean([sum(r.detected[g["gold_id"]] for p in ids for g in ref.get(p, []) if g["fields"]["modality"] == mod) for r in P]) / max(1, res["reference_by_modality"].get(mod, 0)))
                             for mod in sorted(res["reference_by_modality"])}
res["fields"] = {f: {k: r4(v) for k, v in x.items()} for f, x in E.field_scores(P).items()}
res["error_taxonomy"] = E.error_taxonomy(P)
# funnel
F = Counter()
for r in a.runs:
    rr = rows[r]; fn = {}
    for x in rr: fn.setdefault(x["provision_id"], x.get("provision_function"))
    for p in ids: F["1 provision function: " + str(fn.get(p))] += 1
    F["2 norms listed"] += sum(x["status"] in ("located", "quote_not_verbatim") for x in rr)
    for x in rr:
        if x["status"] == "located": F["3 quotation aligned: " + x["aligned_by"]] += 1; F["4 " + x["decision"]] += 1
        if x["status"] == "quote_not_verbatim": F["3 quotation rejected"] += 1
    F["5 removed by deduplication"] += sum(x["status"] == "located" for x in rr) - len(final_records(rr))
    F["6 final records"] += len(final_records(rr))
    calls = read_jsonl(next(iter([RUNS / r / "calls.jsonl"] if (RUNS / r / "calls.jsonl").exists() else [RUNS / r / "calls.jsonl.gz"])))
    F["calls"] += len(calls); F["cost_usd"] += sum(c.get("cost_usd") or 0 for c in calls); F["tokens_in"] += sum(c.get("tokens_in") or 0 for c in calls); F["tokens_out"] += sum(c.get("tokens_out") or 0 for c in calls)
res["funnel_mean_per_run"] = {k: r4(v / len(a.runs)) for k, v in sorted(F.items())}
RESULTS.mkdir(exist_ok=True); (RESULTS / f"{a.set}_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
m = res["main"]["all_records (primary)"]
lines = [f"# Results on the {a.set} set ({len(ids)} provisions, {nref} reference obligations)", "", "| Output | Runs | Predictions | TP | FP | FN | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in res["main"].items():
    lines.append(f"| {k} | {v['runs']} | {v['predictions']} | {v['TP']} | {v['FP']} | {v['FN']} | {v['precision']['value']} {v['precision']['ci']} | {v['recall']['value']} {v['recall']['ci']} | {v['f1']['value']} {v['f1']['ci']} |")
(RESULTS / f"{a.set}_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print("annotator agreement", res["annotator_agreement"]); print("funnel", res["funnel_mean_per_run"])
