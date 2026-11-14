"""Obligation extraction: run OFEX on a set of provisions. PAID MODEL CALLS (about USD 0.17 per run of 240 provisions).
Usage: python scripts/03_extract.py --set test --name my_run --ceiling 0.5
Writes runs/<name>/candidates.jsonl (all records, incl. rejected quotations and no-norm decisions), calls.jsonl, meta.json."""
import _path, argparse, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from ofex.config import DATA, RUNS, EXTRACTION_MODEL, TEMPERATURE, PROMPTS_OFEX
from ofex.corpus import Corpus
from ofex.extract import extract_provision, PROMPT_FILES
from ofex.io import read_jsonl, write_jsonl, sha256_file
from ofex.llm import Budget

ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["dev", "test"]); ap.add_argument("--name", required=True)
ap.add_argument("--ceiling", type=float, required=True); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
d = RUNS / a.name
if (d / "candidates.jsonl").exists(): raise SystemExit("run exists; never overwritten")
d.mkdir(parents=True, exist_ok=True); corpus = Corpus(); budget = Budget(a.ceiling); t0 = time.time()
ids = [r["provision_id"] for r in read_jsonl(DATA / f"{a.set}_provisions.jsonl")]
with ThreadPoolExecutor(a.workers) as ex:
    rows = [r for rs in ex.map(lambda p: extract_provision(p, corpus, d / "calls.jsonl", budget, {"run": a.name}), ids) for r in rs]
write_jsonl(d / "candidates.jsonl", rows); calls = read_jsonl(d / "calls.jsonl")
meta = {"run": a.name, "set": a.set, "model": EXTRACTION_MODEL, "temperature": TEMPERATURE, "provisions": len(ids), "calls": len(calls), "failed_calls": sum(not c["ok"] for c in calls),
        "tokens_in": sum(c.get("tokens_in") or 0 for c in calls), "tokens_out": sum(c.get("tokens_out") or 0 for c in calls), "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4),
        "runtime_seconds": round(time.time() - t0, 1), "prompt_sha256": {f: sha256_file(PROMPTS_OFEX / f) for f in PROMPT_FILES}}
(d / "meta.json").write_text(json.dumps(meta, indent=1))
print(a.name, {k: meta[k] for k in ("calls", "failed_calls", "cost_usd", "runtime_seconds")}, dict(Counter(r.get("decision") or r["status"] for r in rows)))
