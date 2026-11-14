"""Run revised obligation-first extraction (B'). Usage: ./run.sh experiment2/v4_obligation_first_revised/scripts/run_b.py --tag d1 --split old300 --version v4.0
Never overwrites a run. Never reads a reference. One repair retry for non-verbatim quotations (the reference annotators had the same retry)."""
from __future__ import annotations
import argparse, hashlib, json, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from v4lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--split", required=True, choices=["old300", "dev", "eval", "test"])
ap.add_argument("--version", default="v4.0"); ap.add_argument("--no-align", action="store_true"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
d = RUNS4 / a.tag
if (d / "candidates.jsonl").exists(): raise SystemExit("run exists; runs are never overwritten")
d.mkdir(parents=True, exist_ok=True); log = d / "calls.jsonl"; ids = ids_of(a.split); t0 = time.time()
tag = {"tag": a.tag, "version": a.version, "split": a.split}; ALIGN = not a.no_align


def locate_all(text, duties):
    taken, spans, bad = set(), [], []
    for dty in duties:
        sp, method = locate_plus(text, dty["quote"], taken) if ALIGN else (locate(text, dty["quote"], taken), "exact")
        if sp is None: bad.append(dty["quote"])
        else: taken.add(sp)
        spans.append((sp, method) if sp else None)
    return spans, bad


def one(pid: str) -> list[dict]:
    text = corpus[pid]["text"]; ctx, ctx_ids = context(pid)
    msgs = messages("b_step1.txt", a.version, context=ctx, provision_id=pid, text=text)
    r1 = call(msgs, "b_step1", 3000, log, {**tag, "step": 1, "provision_id": pid})
    if r1 is None: return [{"provision_id": pid, "status": "call_failed", "step": 1}]
    duties = r1["duties"] if r1["provision_function"] == "duty_on_regulated_party" else []
    spans, bad = locate_all(text, duties); repaired = False
    if bad:
        m2 = [msgs[0], {"role": "user", "content": msgs[1]["content"] + prompt_file("b_step1_repair.txt", a.version).read_text().replace("{{bad}}", json.dumps(bad, ensure_ascii=False))}]
        r1b = call(m2, "b_step1", 3000, log, {**tag, "step": "1-repair", "provision_id": pid})
        if r1b is not None:
            d2 = r1b["duties"] if r1b["provision_function"] == "duty_on_regulated_party" else []
            s2, bad2 = locate_all(text, d2)
            if len(bad2) < len(bad) and d2: r1, duties, spans, repaired = r1b, d2, s2, True
    rows = []
    for i, (dty, span) in enumerate(zip(duties, spans)):
        src = (corpus.get(dty["actor_source"]) or {}).get("text", "")
        ev = {"actor": found(dty["actor_evidence"], text, src, ctx), "action": found(dty["action_evidence"], text), "force": found(dty["force_evidence"], text, ctx)}
        row = {"cid": f"{pid}#B{i}", "provision_id": pid, "quote": dty["quote"], "step1": dty, "evidence_found": ev, "provision_function": r1["provision_function"], "repaired": repaired}
        if span is None: rows.append({**row, "status": "quote_not_verbatim", "decision": "REJECT"}); continue
        (st, en), method = span
        rows.append({**row, "start": st, "end": en, "quote": text[st:en], "model_quote": dty["quote"], "aligned_by": method, "status": "located",
                     "decision": "ACCEPT" if all(ev.values()) and dty["certainty"] == "certain" else "UNCERTAIN"})
    live = [r for r in rows if r["status"] == "located"]
    if not rows: rows.append({"provision_id": pid, "status": "no_duty", "provision_function": r1["provision_function"], "function_evidence": r1.get("function_evidence")})
    if live:
        duties_txt = json.dumps([{"index": k, "quote": " ".join(r["quote"].split()), "modality": r["step1"]["modality"], "actor_evidence": r["step1"]["actor_evidence"]} for k, r in enumerate(live)], ensure_ascii=False, indent=1)
        r2 = call(messages("b_step2.txt", a.version, context=ctx, provision_id=pid, text=text, duties=duties_txt), "b_step2", 4000, log, {**tag, "step": 2, "provision_id": pid})
        recs = {x["index"]: x for x in (r2 or {}).get("records", [])}
        for k, r in enumerate(live):
            x = recs.get(k) or {}
            r["fields"] = {f: null_norm(x.get(f)) for f in FIELDS}; r["fields"]["modality"] = r["step1"]["modality"]
            r["actor_inferred"] = x.get("actor_inferred"); r["source_provision_ids"] = [pid] + [s for s in x.get("source_provision_ids", []) if s in ctx_ids]; r["step2_missing"] = not x
    return rows


with ThreadPoolExecutor(a.workers) as ex: allrows = [r for rows in ex.map(one, ids) for r in rows]
write_jsonl(d / "candidates.jsonl", allrows)
calls = read_jsonl(log)
files = ["system.txt", "atomic_rules.txt", "b_step1.txt", "b_step1_repair.txt", "b_step2.txt", "schemas.json"]
meta = {"tag": a.tag, "prompt_version": a.version, "model": SUT_MODEL, "temperature": 0.0, "split": a.split, "provisions": len(ids), "calls": len(calls), "failed_calls": sum(not c["ok"] for c in calls),
        "repair_calls": sum(c.get("step") == "1-repair" for c in calls), "tokens_in": sum(c.get("tokens_in") or 0 for c in calls), "tokens_out": sum(c.get("tokens_out") or 0 for c in calls),
        "cost_usd": round(sum(c.get("cost_usd") or 0 for c in calls), 4), "runtime_seconds": round(time.time() - t0, 1), "models_returned": sorted({str(c.get("model_returned")) for c in calls}),
        "alignment": ALIGN, "prompt_sha256": {f: hashlib.sha256(prompt_file(f, a.version).read_bytes()).hexdigest() for f in files}, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0))}
(d / "meta.json").write_text(json.dumps(meta, indent=1))
print(a.tag, {k: meta[k] for k in ("calls", "failed_calls", "repair_calls", "cost_usd", "runtime_seconds")}, dict(Counter(r.get("decision") or r["status"] for r in allrows)), "total spent v4", round(spent(), 4))
