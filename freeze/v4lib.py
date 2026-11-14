"""Version 4: revised obligation-first extraction (B'). Development data = the 300 provisions of Experiment 1 (both old
splits); test data = fresh provisions never used before. The reference is read only by evaluate.py."""
from __future__ import annotations
import functools, glob, json, os, re, sys, threading
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2_ed_roe" / "scripts"))
from v2_common import *
from p3_filter import stage3_filter

V4 = ROOT / "experiment2" / "v4_obligation_first_revised"; PR = V4 / "prompts"; RUNS4 = V4 / "runs"; DATA4 = V4 / "data"
CEILING_USD = float(os.environ.get("V4_CEILING_USD", "5.0"))
corpus = {p["provision_id"]: p for p in read_jsonl(DATA / "corpus_provisions.jsonl")}
old = provisions()
by_art = defaultdict(list)
for _p in corpus.values(): by_art[(_p["regulation_code"], _p["article"])].append(_p)
REF_ART = re.compile(r"\bPasal\s+(\d+[A-Z]?)\b")
SCHEMAS = json.loads((PR / "schemas.json").read_text())


def ids_of(split: str) -> list[str]:
    if split == "old300": return sorted(old)
    if split in ("dev", "eval"): return sorted(p for p, v in old.items() if v["split"] == split)
    if split == "test": return sorted(r["provision_id"] for r in read_jsonl(DATA4 / "test_provisions.jsonl"))
    raise ValueError(split)


def context(pid: str, cap: int = 6000, cap_ref: int = 2000) -> tuple[str, list[str]]:
    p = corpus[pid]; out, ids, n = [], [], 0
    for q in by_art[(p["regulation_code"], p["article"])]:
        if q["provision_id"] == pid: continue
        s = f"[{q['provision_id']}] {q['text']}"
        if n + len(s) > cap: break
        out.append(s); ids.append(q["provision_id"]); n += len(s)
    m = 0
    for art in dict.fromkeys(REF_ART.findall(p["text"])):
        if art == p["article"]: continue
        for q in by_art.get((p["regulation_code"], art), []):
            s = f"[{q['provision_id']}] {q['text']}"
            if m + len(s) > cap_ref: break
            out.append(s); ids.append(q["provision_id"]); m += len(s)
    return ("\n".join(out) or "(tidak ada)"), ids


def prompt_file(name: str, version: str) -> Path:
    return (PR / version / name) if (PR / version / name).exists() else PR / name


def messages(name: str, version: str, **kw) -> list[dict]:
    kw = {"atomic_rules": prompt_file("atomic_rules.txt", version).read_text(), **kw}
    user = functools.reduce(lambda s, kv: s.replace("{{" + kv[0] + "}}", kv[1]), kw.items(), prompt_file(name, version).read_text())
    return [{"role": "system", "content": prompt_file("system.txt", version).read_text()}, {"role": "user", "content": user}]


_lock = threading.Lock(); _spent = None
def spent() -> float:
    return sum(c.get("cost_usd") or 0 for f in glob.glob(str(V4 / "**" / "calls*.jsonl"), recursive=True) for c in read_jsonl(f))


def guarded_chat(model, msgs, max_tokens, log, tag, response_format=None):
    global _spent
    with _lock:
        if _spent is None: _spent = spent()
        if _spent >= CEILING_USD: raise SystemExit(f"cost ceiling reached: {_spent:.4f} USD")
    rec = chat(model, msgs, max_tokens=max_tokens, response_format=response_format, log_path=log, tag=tag)
    with _lock: _spent += rec.get("cost_usd") or 0
    return rec


def call(msgs, schema_name: str, max_tokens: int, log: Path, tag: dict):
    rf = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": SCHEMAS[schema_name]}}
    rec = guarded_chat(SUT_MODEL, msgs, max_tokens, log, tag, rf)
    if not rec["ok"]: return None
    try: return json.loads(rec["text"])
    except Exception: return None


def found(evidence, *texts) -> bool:
    return bool(evidence) and any(locate(t, evidence) is not None for t in texts if t)


def reference_for(split: str) -> dict:
    ref = reference()
    f = DATA4 / "test_reference_obligations.jsonl"
    if split == "test":
        ref = {}
        for g in read_jsonl(f): ref.setdefault(g["provision_id"], []).append(g)
    return ref


def outputs(rows, which="all_records"):
    """all_records = every record whose quotation is located in the provision (ACCEPT and UNCERTAIN; the flag is kept in the record).
    accept_only = records whose three evidence strings were all found verbatim."""
    keep = [r for r in rows if r.get("status") == "located" and (r["decision"] == "ACCEPT" or (which == "all_records" and r["decision"] == "UNCERTAIN"))]
    by = defaultdict(list)
    for r in keep: by[r["provision_id"]].append({**r, "pred_id": r["cid"], "fields": r.get("fields") or {}})
    return [x for pid in by for x in stage3_filter(by[pid], "contained")[0]]


# ---- v4.1: deterministic alignment of a non-verbatim quotation to the source (no reference involved) -------------------
_FURN = r"(?:\s+|\s*https?://\S+\s*|\s*-\s*\d+\s*-\s*)+"          # whitespace, or page furniture (URL, "- 12 -") between two words
_ITEM = re.compile(r"(?:(?<=\s)|^)(?:[a-z]{1,2}\.|\d{1,2}\.)\s+\S")


def locate_plus(text: str, quote: str, taken: set | None = None):
    """Returns (span, method). 1 exact (whitespace-tolerant); 2 same words with page furniture in between;
    3 the part of the quotation that starts at its last list-item marker (the model joined the list stem to an item)."""
    sp = locate(text, quote, taken)
    if sp: return sp, "exact"
    words = (quote or "").split()
    if len(words) >= 3:
        for m in re.finditer(_FURN.join(re.escape(w) for w in words), text):
            if taken is None or (m.start(), m.end()) not in taken: return (m.start(), m.end()), "page_furniture_tolerant"
    q = " ".join(words); marks = [m.start() for m in _ITEM.finditer(q)]
    for st in reversed(marks):
        if st == 0: continue
        sp = locate(text, q[st:], taken)
        if sp and sp[1] - sp[0] >= 15: return sp, "list_item_part"
    return None, None
