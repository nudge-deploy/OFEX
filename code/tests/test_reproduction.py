"""Offline reproduction checks (no model call): the code in this folder rebuilds the samples, sends exactly the stored
prompts, and recomputes exactly the reported numbers from the stored outputs."""
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ofex import evaluate as E
from ofex.annotate import user_block, SYSTEM
from ofex.config import DATA, RUNS, FIELDS
from ofex.corpus import Corpus
from ofex.extract import step1_messages, step2_messages, final_records
from ofex.io import read_jsonl
from ofex.sampling import category, stratified_sample
from ofex.text import null_norm, locate_plus

corpus = Corpus()


def test_samples_are_reproduced_from_the_corpus():
    units = [(p["provision_id"], f"{p['regulation_code']}|{category(p['text'])}") for p in corpus.rows]
    dev = stratified_sample(units, 300, 20261112)
    test = stratified_sample([u for u in units if u[0] not in set(dev)], 240, 20261114)
    assert dev == sorted(r["provision_id"] for r in read_jsonl(DATA / "dev_provisions.jsonl"))
    assert test == sorted(r["provision_id"] for r in read_jsonl(DATA / "test_provisions.jsonl"))
    assert not set(dev) & set(test)


@pytest.mark.parametrize("run", ["test1", "dev"])
def test_extraction_prompts_equal_the_stored_calls(run):
    calls = read_jsonl(RUNS / run / "calls.jsonl.gz")
    step1 = [c for c in calls if c.get("step") == 1]
    assert step1
    for c in step1:
        pid = c["provision_id"]; ctx, _ = corpus.context(pid)
        assert step1_messages(pid, corpus.text(pid), ctx) == c["messages"], pid
    rows = {r["cid"]: r for r in read_jsonl(RUNS / run / "candidates.jsonl") if r.get("cid")}
    for c in [c for c in calls if c.get("step") == 2]:
        pid = c["provision_id"]; ctx, _ = corpus.context(pid)
        duties = c["messages"][1]["content"].split("<norma untrusted=\"true\">\n", 1)[1].rsplit("\n</norma>", 1)[0]
        assert step2_messages(pid, corpus.text(pid), ctx, duties) == c["messages"], pid


def test_annotation_prompts_equal_the_stored_calls():
    calls = read_jsonl(RUNS / "reference_annotation" / "test_calls.jsonl.gz")
    first = {}
    for c in calls:
        if c["task"] in ("anotasi_A", "anotasi_B") and "PERBAIKI" not in c["user"] and "KOREKSI" not in c["user"]:
            first.setdefault(c["user"].split("\n", 1)[0].removeprefix("ID ayat: "), c["user"])
    assert len(first) == 240
    for pid, user in first.items():
        assert user_block(pid, corpus.text(pid), corpus.annotation_context(pid)) == user, pid
    assert "1. TUGAS" in SYSTEM and SYSTEM.startswith("Anda anotator ahli")


def test_freeze_record_matches_prompt_and_code_files():
    import hashlib
    root = Path(__file__).resolve().parents[2]; rec = json.loads((root / "freeze" / "freeze_record.json").read_text())
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(h(root / "prompts" / "ofex" / f) == v for f, v in rec["prompt_files_sha256"].items())
    assert all(h(root / "freeze" / f) == v for f, v in rec["code_sha256"].items())
    for r in ("dev", "test1", "test2", "test3"):
        assert json.loads((RUNS / r / "meta.json").read_text())["prompt_sha256"] == rec["prompt_files_sha256"]


def test_quotation_alignment_is_reproduced():
    for r in read_jsonl(RUNS / "test1" / "candidates.jsonl"):
        if r.get("status") == "located" and r["aligned_by"] != "exact":
            sp, how = locate_plus(corpus.text(r["provision_id"]), r["model_quote"])
            assert how == r["aligned_by"] and corpus.text(r["provision_id"])[sp[0]:sp[1]] == r["quote"]


def test_reported_test_numbers_are_reproduced():
    ids = sorted(r["provision_id"] for r in read_jsonl(DATA / "test_provisions.jsonl"))
    ref = E.by_provision(read_jsonl(DATA / "test_reference_obligations.jsonl"))
    runs = [E.RunScore(ids, ref, E.by_provision(final_records(read_jsonl(RUNS / r / "candidates.jsonl")))) for r in ("test1", "test2", "test3")]
    m = E.prf_with_ci(runs)
    assert sum(len(v) for v in ref.values()) == 194
    assert round(m["precision"]["value"], 4) == 0.7184 and round(m["recall"]["value"], 4) == 0.7320 and round(m["f1"]["value"], 4) == 0.7251
    assert [round(m[k][x], 4) for k in ("precision", "recall", "f1") for x in ("ci_lo", "ci_hi")] == [0.6199, 0.8091, 0.651, 0.7986, 0.6447, 0.7917]
    assert [int(r.tp.sum()) for r in runs] == [141, 143, 142]


def test_reported_development_numbers_are_reproduced():
    ids = sorted(r["provision_id"] for r in read_jsonl(DATA / "dev_provisions.jsonl"))
    ref = E.by_provision(read_jsonl(DATA / "dev_reference_obligations.jsonl"))
    s = E.RunScore(ids, ref, E.by_provision(final_records(read_jsonl(RUNS / "dev" / "candidates.jsonl"))))
    assert (int(s.tp.sum()), int(s.npred.sum()), int(s.nref.sum())) == (187, 222, 233)


def test_every_final_quotation_is_source_text():
    for r in ("test1", "test2", "test3", "dev"):
        for x in final_records(read_jsonl(RUNS / r / "candidates.jsonl")):
            assert corpus.text(x["provision_id"])[x["start"]:x["end"]] == x["quote"]


def test_null_normalisation():
    assert null_norm("null") is None and null_norm(" - ") is None and null_norm("Tidak ada pengecualian yang disebutkan") is None
    assert null_norm("tidak ada kesesuaian identitas") == "tidak ada kesesuaian identitas"


def test_matching_rule():
    g = [{"start": 0, "end": 100}]; assert E.match_spans(g, [{"start": 0, "end": 60}]) == [(0, 0, 0.6)]
    assert E.match_spans(g, [{"start": 0, "end": 40}]) == []
