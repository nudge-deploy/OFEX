"""Obligation extraction — OFEX obligation-first extraction.

Step 1 normative interpretation (LLM call 1) → step 2 alignment of quotations to the source (code; one repair call when
needed) → step 3 check of the three evidence strings (code) → step 4 nine-field structuring (LLM call 2) → step 5
deduplication (code). Prompts are read from the frozen files in prompts/ofex/."""
from __future__ import annotations
import functools, json
from collections import defaultdict
from .config import PROMPTS_OFEX, EXTRACTION_MODEL, TEMPERATURE, MAX_TOKENS_STEP1, MAX_TOKENS_STEP2, FIELDS
from .corpus import Corpus
from .dedup import deduplicate
from .llm import chat, schema_format
from .text import found, locate, locate_plus, null_norm

PROMPT_FILES = ["system.txt", "atomic_rules.txt", "b_step1.txt", "b_step1_repair.txt", "b_step2.txt", "schemas.json"]
SCHEMAS = json.loads((PROMPTS_OFEX / "schemas.json").read_text())


def _fill(template: str, **kw) -> str:
    return functools.reduce(lambda s, kv: s.replace("{{" + kv[0] + "}}", kv[1]), kw.items(), template)


def step1_messages(pid: str, text: str, context: str) -> list[dict]:
    user = _fill((PROMPTS_OFEX / "b_step1.txt").read_text(), atomic_rules=(PROMPTS_OFEX / "atomic_rules.txt").read_text(),
                 context=context, provision_id=pid, text=text)
    return [{"role": "system", "content": (PROMPTS_OFEX / "system.txt").read_text()}, {"role": "user", "content": user}]


def repair_messages(msgs: list[dict], bad_quotes: list[str]) -> list[dict]:
    extra = (PROMPTS_OFEX / "b_step1_repair.txt").read_text().replace("{{bad}}", json.dumps(bad_quotes, ensure_ascii=False))
    return [msgs[0], {"role": "user", "content": msgs[1]["content"] + extra}]


def step2_messages(pid: str, text: str, context: str, duties_json: str) -> list[dict]:
    user = _fill((PROMPTS_OFEX / "b_step2.txt").read_text(), atomic_rules=(PROMPTS_OFEX / "atomic_rules.txt").read_text(),
                 context=context, provision_id=pid, text=text, duties=duties_json)
    return [{"role": "system", "content": (PROMPTS_OFEX / "system.txt").read_text()}, {"role": "user", "content": user}]


def _call(msgs, schema_name, max_tokens, log, tag, budget):
    rec = chat(EXTRACTION_MODEL, msgs, max_tokens=max_tokens, response_format=schema_format(schema_name, SCHEMAS[schema_name]),
               temperature=TEMPERATURE, log_path=log, tag=tag, budget=budget)
    if not rec["ok"]: return None
    try: return json.loads(rec["text"])
    except Exception: return None


def _align_all(text: str, duties: list[dict]):
    taken, spans, bad = set(), [], []
    for d in duties:
        sp, method = locate_plus(text, d["quote"], taken)
        if sp is None: bad.append(d["quote"])
        else: taken.add(sp)
        spans.append((sp, method) if sp else None)
    return spans, bad


def extract_provision(pid: str, corpus: Corpus, log, budget=None, tag: dict | None = None) -> list[dict]:
    """All records of one provision, including rejected quotations and the no-norm decision (for the audit trail)."""
    text = corpus.text(pid); ctx, ctx_ids = corpus.context(pid); tag = {**(tag or {}), "provision_id": pid}
    # step 1
    msgs = step1_messages(pid, text, ctx)
    r1 = _call(msgs, "b_step1", MAX_TOKENS_STEP1, log, {**tag, "step": 1}, budget)
    if r1 is None: return [{"provision_id": pid, "status": "call_failed", "step": 1}]
    duties = r1["duties"] if r1["provision_function"] == "duty_on_regulated_party" else []
    # step 2 (alignment; one repair call when some quotation cannot be placed)
    spans, bad = _align_all(text, duties); repaired = False
    if bad:
        r1b = _call(repair_messages(msgs, bad), "b_step1", MAX_TOKENS_STEP1, log, {**tag, "step": "1-repair"}, budget)
        if r1b is not None:
            d2 = r1b["duties"] if r1b["provision_function"] == "duty_on_regulated_party" else []
            s2, bad2 = _align_all(text, d2)
            if len(bad2) < len(bad) and d2: r1, duties, spans, repaired = r1b, d2, s2, True
    # step 3 (evidence check) and record assembly
    rows = []
    for i, (d, span) in enumerate(zip(duties, spans)):
        src = (corpus.by_id.get(d["actor_source"]) or {}).get("text", "")
        ev = {"actor": found(d["actor_evidence"], text, src, ctx), "action": found(d["action_evidence"], text), "force": found(d["force_evidence"], text, ctx)}
        row = {"cid": f"{pid}#B{i}", "provision_id": pid, "quote": d["quote"], "step1": d, "evidence_found": ev, "provision_function": r1["provision_function"], "repaired": repaired}
        if span is None:
            rows.append({**row, "status": "quote_not_verbatim", "decision": "REJECT"}); continue
        (st, en), method = span
        rows.append({**row, "start": st, "end": en, "quote": text[st:en], "model_quote": d["quote"], "aligned_by": method, "status": "located",
                     "decision": "ACCEPT" if all(ev.values()) and d["certainty"] == "certain" else "UNCERTAIN"})
    live = [r for r in rows if r["status"] == "located"]
    if not rows:
        rows.append({"provision_id": pid, "status": "no_duty", "provision_function": r1["provision_function"], "function_evidence": r1.get("function_evidence")})
    # step 4 (nine fields)
    if live:
        duties_json = json.dumps([{"index": k, "quote": " ".join(r["quote"].split()), "modality": r["step1"]["modality"], "actor_evidence": r["step1"]["actor_evidence"]}
                                  for k, r in enumerate(live)], ensure_ascii=False, indent=1)
        r2 = _call(step2_messages(pid, text, ctx, duties_json), "b_step2", MAX_TOKENS_STEP2, log, {**tag, "step": 2}, budget)
        recs = {x["index"]: x for x in (r2 or {}).get("records", [])}
        for k, r in enumerate(live):
            x = recs.get(k) or {}
            r["fields"] = {f: null_norm(x.get(f)) for f in FIELDS}; r["fields"]["modality"] = r["step1"]["modality"]
            r["actor_inferred"] = x.get("actor_inferred"); r["source_provision_ids"] = [pid] + [s for s in x.get("source_provision_ids", []) if s in ctx_ids]
            r["step2_missing"] = not x
    return rows


def final_records(rows: list[dict], output: str = "all_records") -> list[dict]:
    """Step 5 and the output definition. all_records (primary) = every record with a located quotation, ACCEPT or UNCERTAIN;
    accept_only (secondary) = ACCEPT records only."""
    ok = ("ACCEPT", "UNCERTAIN") if output == "all_records" else ("ACCEPT",)
    by = defaultdict(list)
    for r in rows:
        if r.get("status") == "located" and r["decision"] in ok:
            by[r["provision_id"]].append({**r, "pred_id": r["cid"], "fields": r.get("fields") or {}})
    return [x for pid in by for x in deduplicate(by[pid])[0]]
