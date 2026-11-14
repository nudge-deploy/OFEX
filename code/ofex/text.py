"""Text normalisation and verbatim location of quotations."""
import re
import unicodedata

NULL_STRINGS = {"", "null", "none", "nil", "n/a", "na", "-", "–", "—", "tidak ada", "tidak disebutkan",
                "tidak disebut", "tidak tersedia", "kosong"}
_NULL_PHRASE = re.compile(r"^(tidak ada (nilai batas|batas|ambang|pengecualian|bukti|syarat|kondisi|tenggat|frekuensi|waktu|aktor|objek)\b.*"
                          r"|tidak ada .*(disebutkan|ditentukan|dinyatakan|tercantum|diatur)"
                          r"|tidak (disebutkan|ditentukan|dinyatakan|tercantum|diatur|berlaku|relevan)\b.*"
                          r"|belum (disebutkan|ditentukan)\b.*)$")


def norm_text(x) -> str:
    """NFKC, case-folded, whitespace collapsed, trailing punctuation stripped."""
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).casefold()
    return " ".join(s.split()).strip(" .;:,")


def null_norm(x):
    """Null-like strings ("null", "-", "tidak ada", ...) become None."""
    if x is None:
        return None
    s = str(x).strip()
    n = norm_text(s)
    if n in NULL_STRINGS or _NULL_PHRASE.match(n):
        return None
    return s


def locate(text: str, quote: str, taken: set | None = None):
    """Verbatim location of `quote` in `text`; only the kind of whitespace between words may differ.
    Returns the first occurrence not in `taken`, else the first occurrence, else None."""
    words = (quote or "").split()
    if not words:
        return None
    first = None
    for m in re.finditer(r"\s+".join(re.escape(w) for w in words), text):
        span = (m.start(), m.end())
        first = first or span
        if taken is None or span not in taken:
            return span
    return first


_FURNITURE = r"(?:\s+|\s*https?://\S+\s*|\s*-\s*\d+\s*-\s*)+"        # whitespace, URL, or page number "- 12 -"
_ITEM = re.compile(r"(?:(?<=\s)|^)(?:[a-z]{1,2}\.|\d{1,2}\.)\s+\S")


def locate_plus(text: str, quote: str, taken: set | None = None):
    """Step 2 of OFEX. Returns (span, method).
    1 exact; 2 same words with page furniture in between; 3 the part of the quotation that starts at its last
    list-item marker (the model joined the list stem to an item)."""
    sp = locate(text, quote, taken)
    if sp:
        return sp, "exact"
    words = (quote or "").split()
    if len(words) >= 3:
        for m in re.finditer(_FURNITURE.join(re.escape(w) for w in words), text):
            if taken is None or (m.start(), m.end()) not in taken:
                return (m.start(), m.end()), "page_furniture_tolerant"
    q = " ".join(words)
    marks = [m.start() for m in _ITEM.finditer(q)]
    for st in reversed(marks):
        if st == 0:
            continue
        sp = locate(text, q[st:], taken)
        if sp and sp[1] - sp[0] >= 15:
            return sp, "list_item_part"
    return None, None


def found(evidence, *texts) -> bool:
    """True when an evidence string is found verbatim in any of the texts."""
    return bool(evidence) and any(locate(t, evidence) is not None for t in texts if t)
