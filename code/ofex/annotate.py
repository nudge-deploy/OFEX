"""Reference annotation by two independent LLM annotators and an LLM adjudicator.

Procedure used to produce the reference in data/: annotator 1 and annotator 2 each annotate one provision with the same
guideline-based system prompt; a quotation not found verbatim gets one repair request; if annotators 1 and 2 agree on the number
of obligations, their spans (IoU ≥ 0.5), modality, and treatment class, the annotation of annotator 2 is the reference; otherwise the
adjudicator, shown both annotations, sets the reference. The annotators never see any system output.
Calls use a strict JSON schema; a schema violation is retried once with the validator message appended (KOREKSI),
and a failed call is repeated up to three times."""
from __future__ import annotations
import json
from .config import PROMPTS_ANNOTATION, ANNOTATOR_A, ANNOTATOR_B, ADJUDICATOR, ANNOTATION_MAX_TOKENS, FIELDS
from .corpus import Corpus
from .evaluate import span_iou
from .llm import chat, schema_format
from .text import locate

AF = ("actor", "modality", "action", "object", "condition", "threshold", "timing", "exception", "evidence", "treatment_class")
_SCHEMA = json.loads((PROMPTS_ANNOTATION / "annotation_schema.json").read_text())
GUIDELINE = (PROMPTS_ANNOTATION / "annotation_guideline.txt").read_text()     # sections 1-5 of the annotation guideline
SYSTEM = ("Anda anotator ahli regulasi jasa keuangan Indonesia. Tandai setiap obligation pada SATU ayat menurut pedoman berikut.\n\n"
          "PEDOMAN\n" + GUIDELINE +
          "\nATURAN KELUARAN\n- `quote` harus salinan PERSIS (kata demi kata, berurutan, tanpa elipsis) dari teks di dalam blok <ayat>; "
          "itulah span yang Anda blok.\n- Ayat tanpa obligation: kembalikan `obligations` kosong.\n"
          "- Field yang tidak disebut di ayat diisi null.\n- `source_provision_ids` hanya diisi ID ayat lain dari blok <konteks> bila memang diperlukan.")
ADJUDICATOR_SYSTEM = SYSTEM + ("\n\nANDA ADJUDIKATOR. Dua anotator bekerja terpisah. Tetapkan anotasi akhir yang paling "
                               "sesuai pedoman; Anda boleh memilih salah satu, menggabungkan, atau memperbaiki. Jangan asal mengambil gabungan keduanya.")
_ENUMS = {"modality": {"obligation", "prohibition", "permission"}, "treatment_class": {"informational", "interpretive", "assessable", "executable"}}


def user_block(pid: str, text: str, ctx: list[dict]) -> str:
    k = "\n".join(f"[{x['provision_id']}] {x['text']}" for x in ctx) or "(tidak ada)"
    return f"ID ayat: {pid}\n<ayat>\n{text}\n</ayat>\n<konteks>\n{k}\n</konteks>"


def _validate(obj) -> str | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("obligations"), list): return "objek harus memuat daftar `obligations`"
    for o in obj["obligations"]:
        if not isinstance(o, dict) or not isinstance(o.get("quote"), str): return "setiap obligation harus memuat `quote` berupa teks"
        for f, allowed in _ENUMS.items():
            if o.get(f) not in allowed: return f"nilai `{f}` tidak sah: {o.get(f)!r}"
    return None


def ask(model, task, system, user, log, budget, tries=3) -> list[dict]:
    for _ in range(tries):
        u = user
        for attempt in range(2):                       # first answer + one KOREKSI retry on a schema violation
            rec = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": u}], max_tokens=ANNOTATION_MAX_TOKENS,
                       response_format=schema_format(_SCHEMA["name"], _SCHEMA["schema"]), log_path=log, tag={"task": task}, budget=budget)
            if not rec["ok"]: break
            err = "jawaban terpotong karena batas token" if rec.get("finish_reason") == "length" else None
            if err is None:
                try: obj = json.loads(rec["text"]); err = _validate(obj)
                except Exception as exc: err = f"jawaban bukan JSON yang sah: {exc}"
            if err is None: return obj["obligations"]
            u = f"{user}\n\n# KOREKSI\nJawaban sebelumnya ditolak validator skema:\n{err}\nPerbaiki dan kembalikan HANYA objek JSON yang sesuai skema."
    raise RuntimeError(f"annotator {model} failed for {task}")


def spans(pid, text, known, items):
    taken, out, bad = set(), [], []
    for o in items:
        sp = locate(text, o["quote"], taken)
        if sp is None: bad.append(o["quote"]); continue
        taken.add(sp)
        out.append({"span_start": sp[0], "span_end": sp[1], **{f: o.get(f) for f in AF},
                    "source_provision_ids": [pid, *[s for s in o.get("source_provision_ids") or [] if s in known]]})
    return out, bad


def annotate_once(model, task, system, pid, text, known, user, log, budget):
    out, bad = spans(pid, text, known, ask(model, task, system, user, log, budget))
    if bad:
        r2 = ask(model, task, system, user + "\n\nPERBAIKI: kutipan berikut tidak ditemukan persis di blok <ayat>: " + json.dumps(bad, ensure_ascii=False)
                 + ". Kembalikan ulang SELURUH daftar dengan kutipan yang persis.", log, budget)
        out2, bad2 = spans(pid, text, known, r2)
        if len(bad2) < len(bad): out, bad = out2, bad2
    return out, bad


def same(x, y) -> bool:
    if len(x) != len(y): return False
    x, y = sorted(x, key=lambda o: o["span_start"]), sorted(y, key=lambda o: o["span_start"])
    for p, q in zip(x, y):
        if span_iou((p["span_start"], p["span_end"]), (q["span_start"], q["span_end"])) < 0.5: return False
        if p["modality"] != q["modality"] or p["treatment_class"] != q["treatment_class"]: return False
    return True


def annotate_provision(pid: str, corpus: Corpus, log, budget=None) -> dict:
    text = corpus.text(pid); ctx = corpus.annotation_context(pid); known = {x["provision_id"] for x in ctx}; u = user_block(pid, text, ctx)
    a, bad_a = annotate_once(ANNOTATOR_A, "anotasi_A", SYSTEM, pid, text, known, u, log, budget)
    b, bad_b = annotate_once(ANNOTATOR_B, "anotasi_B", SYSTEM, pid, text, known, u, log, budget)
    show = lambda L: json.dumps([{"quote": text[o["span_start"]:o["span_end"]], **{f: o[f] for f in AF}, "source_provision_ids": o["source_provision_ids"][1:]} for o in L],
                                ensure_ascii=False, indent=1)
    agreed = same(a, b)
    if agreed: g, bad_g = b, []
    else: g, bad_g = annotate_once(ADJUDICATOR, "adjudikasi", ADJUDICATOR_SYSTEM, pid, text, known, u + f"\n\nANOTATOR A:\n{show(a)}\n\nANOTATOR B:\n{show(b)}", log, budget)
    return {"a": a, "b": b, "g": g, "bad": {"a": bad_a, "b": bad_b, "g": bad_g}, "adjudicated": not agreed}


def to_reference(pid: str, labels: dict) -> list[dict]:
    """Reference obligations of one provision in the evaluation format."""
    return [{"gold_id": f"{pid}#{i+1}", "provision_id": pid, "start": o["span_start"], "end": o["span_end"], "fields": {f: o.get(f) for f in FIELDS},
             "treatment_class": o["treatment_class"], "source_provision_ids": o["source_provision_ids"]}
            for i, o in enumerate(sorted(labels["g"], key=lambda o: o["span_start"]))]
