"""Reference annotation: build the AI-generated reference for a set of provisions. PAID MODEL CALLS.
Usage: python scripts/02_annotate.py --set test --name my_reference --ceiling 5
Writes runs/<name>/raw_labels.json, reference_obligations.jsonl, calls.jsonl. Never overwrites."""
import _path, argparse, json, time
from concurrent.futures import ThreadPoolExecutor
from ofex.annotate import annotate_provision, to_reference
from ofex.config import DATA, RUNS
from ofex.corpus import Corpus
from ofex.io import read_jsonl, write_jsonl
from ofex.llm import Budget

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["dev", "test"]); ap.add_argument("--name", required=True)
ap.add_argument("--ceiling", type=float, required=True); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
out = RUNS / a.name
if (out / "reference_obligations.jsonl").exists(): raise SystemExit("exists; never overwritten")
out.mkdir(parents=True, exist_ok=True); corpus = Corpus(); budget = Budget(a.ceiling)
ids = [r["provision_id"] for r in read_jsonl(DATA / f"{a.set}_provisions.jsonl")]
cache_f = out / "raw_labels.json"; cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
t0 = time.time()
def one(pid):
    if pid in cache: return
    try: cache[pid] = annotate_provision(pid, corpus, out / "calls.jsonl", budget)
    except RuntimeError as e: print("FAILED", pid, e); return
    cache_f.write_text(json.dumps(cache, ensure_ascii=False))
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, ids))
missing = [p for p in ids if p not in cache]
if missing: raise SystemExit(f"{len(missing)} provisions not annotated; rerun to resume")
write_jsonl(out / "reference_obligations.jsonl", [g for pid in ids for g in to_reference(pid, cache[pid])])
print("done", len(ids), "provisions;", sum(len(cache[p]["g"]) for p in ids), "reference obligations;", round(budget.spent, 3), "USD;", round(time.time() - t0), "s")
