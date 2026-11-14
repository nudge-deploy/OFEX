"""Per-run scores split by annotator agreement (secondary analysis fixed before the test run). No model call.
Usage: python scripts/05_strata_per_run.py --set test --runs test1 test2 test3
Writes results/<set>_agreement_status_per_provision.csv (one row per provision: status and per-run TP/FP/FN)
and results/<set>_strata_per_run.csv (per run and group: TP, FP, FN, precision, recall, F1)."""
import _path, argparse, csv, json
from ofex import evaluate as E
from ofex.config import DATA, RUNS, RESULTS
from ofex.extract import final_records
from ofex.io import read_jsonl

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["dev", "test"]); ap.add_argument("--runs", nargs="+", required=True); a = ap.parse_args()
ids = sorted(r["provision_id"] for r in read_jsonl(DATA / f"{a.set}_provisions.jsonl"))
ref = E.by_provision(read_jsonl(DATA / f"{a.set}_reference_obligations.jsonl")); raw = json.loads((DATA / f"{a.set}_annotator_raw_labels.json").read_text())
S_ = lambda o: {"start": o["span_start"], "end": o["span_end"]}

def status(p):
    """Same definition as the pre-specified test analysis: annotators 1 and 2, raw labels, IoU >= 0.5."""
    x, y = raw[p]["a"], raw[p]["b"]; m = E.match_spans([S_(o) for o in x], [S_(o) for o in y])
    if not x and not y: return "both_none"
    if bool(x) != bool(y): return "identification_disagreement"
    if len(x) != len(y) or len(m) != len(x): return "segmentation_disagreement"
    return "agree_on_presence_and_spans"
st = {p: status(p) for p in ids}
group = lambda s: "agree" if s in ("both_none", "agree_on_presence_and_spans") else "disagree"
RESULTS.mkdir(exist_ok=True)
scores = {r: E.RunScore(ids, ref, E.by_provision(final_records(read_jsonl(RUNS / r / "candidates.jsonl")))) for r in a.runs}
with open(RESULTS / f"{a.set}_agreement_status_per_provision.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["provision_id", "annotator_status", "group", "reference_obligations", "annotator1_obligations", "annotator2_obligations"] + [f"{r}_{k}" for r in a.runs for k in ("TP", "FP", "FN")])
    for i, p in enumerate(ids):
        w.writerow([p, st[p], group(st[p]), len(ref.get(p, [])), len(raw[p]["a"]), len(raw[p]["b"])] +
                   [v for r in a.runs for v in (int(scores[r].tp[i]), int(scores[r].npred[i] - scores[r].tp[i]), int(scores[r].nref[i] - scores[r].tp[i]))])
rows = []
for r in a.runs:
    s = scores[r]
    for g, idx in (("all", list(range(len(ids)))), ("agree", [i for i, p in enumerate(ids) if group(st[p]) == "agree"]), ("disagree", [i for i, p in enumerate(ids) if group(st[p]) == "disagree"])):
        tp, n, nr = s.tp[idx].sum(), s.npred[idx].sum(), s.nref[idx].sum(); P, R, F = s.prf(idx)
        rows.append([r, g, len(idx), int(nr), int(tp), int(n - tp), int(nr - tp), round(P, 4), round(R, 4), round(F, 4)])
with open(RESULTS / f"{a.set}_strata_per_run.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "group", "provisions", "reference_obligations", "TP", "FP", "FN", "precision", "recall", "f1"]); w.writerows(rows)
for x in rows: print(*x)
