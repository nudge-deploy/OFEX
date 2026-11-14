"""Corpus of 2,243 provisions (ayat) of 10 regulations and the context given to the models."""
import re
from collections import defaultdict
from .config import DATA, CONTEXT_CHARS, CROSSREF_CHARS, ANNOTATION_CONTEXT_CHARS
from .io import read_jsonl

REF_ARTICLE = re.compile(r"\bPasal\s+(\d+[A-Z]?)\b")


class Corpus:
    def __init__(self, path=DATA / "corpus_provisions.jsonl"):
        self.rows = read_jsonl(path)
        self.by_id = {p["provision_id"]: p for p in self.rows}
        self.by_article = defaultdict(list)
        for p in self.rows:
            self.by_article[(p["regulation_code"], p["article"])].append(p)

    def text(self, pid: str) -> str:
        return self.by_id[pid]["text"]

    def context(self, pid: str) -> tuple[str, list[str]]:
        """Context for OFEX: other paragraphs of the same article (document order, up to 6,000 characters), then the text of
        articles the provision cites (up to 2,000 characters). Returns (text, provision ids)."""
        p = self.by_id[pid]; out, ids, n = [], [], 0
        for q in self.by_article[(p["regulation_code"], p["article"])]:
            if q["provision_id"] == pid: continue
            s = f"[{q['provision_id']}] {q['text']}"
            if n + len(s) > CONTEXT_CHARS: break
            out.append(s); ids.append(q["provision_id"]); n += len(s)
        m = 0
        for art in dict.fromkeys(REF_ARTICLE.findall(p["text"])):
            if art == p["article"]: continue
            for q in self.by_article.get((p["regulation_code"], art), []):
                s = f"[{q['provision_id']}] {q['text']}"
                if m + len(s) > CROSSREF_CHARS: break
                out.append(s); ids.append(q["provision_id"]); m += len(s)
        return ("\n".join(out) or "(tidak ada)"), ids

    def annotation_context(self, pid: str) -> list[dict]:
        """Context for the reference annotators: other paragraphs of the same article in document order, cut when the
        accumulated provision text would exceed 6,000 characters."""
        p = self.by_id[pid]; out, used = [], 0
        for r in sorted(self.by_article[(p["regulation_code"], p["article"])], key=lambda x: x["char_start"]):
            if r["provision_id"] == pid: continue
            if used + len(r["text"]) > ANNOTATION_CONTEXT_CHARS: break
            out.append(r); used += len(r["text"])
        return out
