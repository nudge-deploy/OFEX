"""Rebuild the two samples from the corpus with the fixed seeds and check them against the stored sample files.
No model call. Output: results/sample_check.json (the data files are never overwritten)."""
import _path, json
from collections import Counter
from ofex.config import DATA, RESULTS, DEV_SIZE, DEV_SEED, TEST_SIZE, TEST_SEED
from ofex.corpus import Corpus
from ofex.io import read_jsonl
from ofex.sampling import category, stratified_sample

corpus = Corpus()
units = [(p["provision_id"], f"{p['regulation_code']}|{category(p['text'])}") for p in corpus.rows]
dev = stratified_sample(units, DEV_SIZE, DEV_SEED)
rest = [u for u in units if u[0] not in set(dev)]
test = stratified_sample(rest, TEST_SIZE, TEST_SEED)
stored_dev = sorted(r["provision_id"] for r in read_jsonl(DATA / "dev_provisions.jsonl"))
stored_test = sorted(r["provision_id"] for r in read_jsonl(DATA / "test_provisions.jsonl"))
res = {"corpus_provisions": len(units), "regulations": len({p["regulation_code"] for p in corpus.rows}), "articles": len({(p["regulation_code"], p["article"]) for p in corpus.rows}),
       "dev_reproduced": dev == stored_dev, "test_reproduced": test == stored_test, "overlap_dev_test": len(set(stored_dev) & set(stored_test)), "unused_provisions": len(units) - len(stored_dev) - len(stored_test),
       "categories_corpus": dict(Counter(s.split("|")[1] for _, s in units)), "test_by_regulation": dict(Counter(p.split("/")[0] for p in stored_test))}
RESULTS.mkdir(exist_ok=True); (RESULTS / "sample_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
