"""Keyword baseline (no model): sentences or list items that contain an Indonesian deontic marker.
Produces spans and modality only."""
import re

MARKERS = (("prohibition", re.compile(r"\b(dilarang|tidak boleh|tidak diperkenankan|tidak dapat)\b", re.I)),
           ("obligation", re.compile(r"\b(wajib|harus|paling lambat|paling lama|paling sedikit|paling banyak)\b", re.I)),
           ("permission", re.compile(r"\b(dapat|berhak|boleh)\b", re.I)))
_LEAD = re.compile(r"^\s*\(\d+[a-z]?\)\s*")
_ITEM = re.compile(r"(?m)^[ \t]*(?:[a-z]{1,2}\.|\d{1,2}\.)[ \t]+")
_SENTENCE_END = re.compile(r"[.;](?=\s|$)")


def _trim(text, s, e):
    while s < e and text[s].isspace(): s += 1
    while e > s and (text[e - 1].isspace() or text[e - 1] in ";"): e -= 1
    return s, e


def units(text: str):
    lead = _LEAD.match(text); base = lead.end() if lead else 0
    items = list(_ITEM.finditer(text, base)); out = []
    if items:
        intro = text[base:items[0].start()]; s, e = _trim(text, base, items[0].start())
        if e > s: out.append((s, e, ""))
        for i, m in enumerate(items):
            end = items[i + 1].start() if i + 1 < len(items) else len(text); s, e = _trim(text, m.end(), end)
            if e > s: out.append((s, e, intro))
        return out
    pos = base
    for m in _SENTENCE_END.finditer(text, base):
        s, e = _trim(text, pos, m.end())
        if e > s: out.append((s, e, ""))
        pos = m.end()
    s, e = _trim(text, pos, len(text))
    if e > s: out.append((s, e, ""))
    return out


def _modality(seg):
    for m, pat in MARKERS:
        if pat.search(seg): return m
    return None


def extract(pid: str, text: str) -> list[dict]:
    us = units(text); has_items = any(c for _, _, c in us); out = []
    for s, e, ctx in us:
        seg = text[s:e]; mod = _modality(seg)
        if mod is None and ctx: mod = _modality(ctx)
        elif mod is not None and not ctx and has_items and seg.rstrip().endswith(":"): continue
        if mod is None: continue
        out.append({"pred_id": f"K:{pid}:{len(out)}", "provision_id": pid, "start": s, "end": e, "quote": seg, "fields": {"modality": mod}})
    return out
