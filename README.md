# OFEX: Obligation-First Evidence-grounded eXtraction — research artefacts

Data, prompts, model outputs, and code for the article *Traceable Obligation Extraction from Indonesian Financial Regulations: An Obligation-First LLM Pipeline*.

All numbers reported in the article can be recomputed from the files in this repository without any model call (Section 4).

## 1. Contents

| Path | Content |
|---|---|
| `data/corpus_provisions.jsonl` | 2,243 provisions (paragraphs, or articles without paragraphs) of 10 regulations of OJK and Bank Indonesia |
| `data/dev_provisions.jsonl` | 300 development provisions |
| `data/test_provisions.jsonl` | 240 held-out test provisions, sampled after the freeze; no overlap with the development set |
| `data/dev_reference_obligations.jsonl`, `data/test_reference_obligations.jsonl` | reference obligations (233 and 194) |
| `data/dev_annotator_raw_labels.json`, `data/test_annotator_raw_labels.json` | labels of annotator 1 (`a`), annotator 2 (`b`), and the final reference (`g`) per provision; quotations that could not be located (`bad`) |
| `prompts/ofex/` | the frozen OFEX prompts and JSON schemas |
| `prompts/annotation/` | annotation guideline (as included in the annotators' prompt) and annotation output schema |
| `freeze/` | freeze record (time, SHA-256 of prompt and code files) and the pipeline code that was hashed |
| `runs/dev`, `runs/test1`–`runs/test3` | OFEX outputs: every record (`candidates.jsonl`), every model call with messages and raw answers (`calls.jsonl.gz`), and run metadata (`meta.json`) |
| `runs/reference_annotation/` | every model call of the reference annotation (development and test) |
| `results/` | tables produced by `code/scripts/` |
| `figures/` | sources (Mermaid) of Fig. 1 (research workflow) and Fig. 2 (OFEX architecture) |
| `code/` | Python package `ofex`, scripts, and reproduction tests |

## 2. Data

**Regulations.** POJK 8/2023, 39/2024, 1/2026, 6/2026, 7/2026, 8/2026, 9/2026 and PBI 3/2026, 5/2026, 6/2026. Texts were extracted from the official PDF documents published on the JDIH portals of OJK and Bank Indonesia and split into 744 articles and 2,243 provisions. Some provisions retain page numbers or URLs from the PDF layout.

**Provision record** (`*_provisions.jsonl`): `provision_id` (e.g. `POJK-39-2024/Pasal-12/Ayat-2`), `regulation_code`, `article`, `text`, and for the samples `set` and `stratum` (regulation × marker category used for stratified sampling).

**Reference obligation** (`*_reference_obligations.jsonl`): `gold_id`, `provision_id`, character span `start`/`end` in the provision text, nine fields (`actor`, `modality`, `action`, `object`, `condition`, `threshold`, `timing`, `exception`, `evidence`), `treatment_class`, and `source_provision_ids`.

**Reference annotation.** The reference labels were produced by two LLM annotators (Qwen 3.7 Plus, Claude Haiku 5.5) working independently with the same guideline, and an LLM adjudicator (Claude Sonnet 5.5) for provisions on which they differed. They have not been validated by legal or compliance experts; all scores measure agreement with this reference (article, Sections III-C and IV-J).

## 3. Method in brief

OFEX processes one provision at a time: (1) normative interpretation by an LLM: provision function and a list of norms with quoted actor, action, and normative-force evidence; (2) deterministic alignment of each quotation to the source text, with one repair call; (3) deterministic check of the three evidence quotations (ACCEPT or UNCERTAIN); (4) nine-field structuring by an LLM; (5) normalisation and deduplication. Model: `openai/gpt-4o-mini` via OpenRouter, temperature 0, strict JSON schema. The prompts and code were frozen on 14 November 2026, 11:55 WIB, before the test provisions were sampled.

## 4. Reproducing the results

Requirements: Python 3.11 or later and the packages in `requirements.txt`.

```
pip install -r requirements.txt
cd code
python scripts/01_samples.py                                     # rebuild both samples from the corpus and seeds
python scripts/04_evaluate.py --set test --runs test1 test2 test3
python scripts/04_evaluate.py --set dev --runs dev
python scripts/05_strata_per_run.py --set test --runs test1 test2 test3
python -m pytest -q tests
```

None of these steps calls a model. The tests check that the samples are rebuilt exactly, that the code builds exactly the prompts stored in the call logs, that the freeze hashes match the prompt and code files, and that the reported numbers are reproduced.

Expected main result (test set, 240 provisions, 194 reference obligations, mean of three runs):

| Condition | Precision | Recall | F1 [95% CI] |
|---|---|---|---|
| Keyword baseline (1 run) | 0.44 | 0.55 | 0.49 [0.39, 0.58] |
| OFEX, all records | 0.72 | 0.73 | 0.73 [0.64, 0.79] |
| OFEX, ACCEPT only | 0.77 | 0.65 | 0.70 [0.62, 0.77] |

Repeating the experiment from scratch requires an OpenRouter API key in the environment variable `OPENROUTER_API_KEY` and incurs cost (about USD 3.8 for annotating 240 provisions and USD 0.17 per OFEX run):

```
python scripts/02_annotate.py --set test --name new_reference --ceiling 5
python scripts/03_extract.py  --set test --name new_run1 --ceiling 0.3
```

New runs never overwrite stored ones. Model outputs vary slightly between runs at temperature 0, so a new run gives close but not identical numbers.

## 5. Notes

- `freeze/v4lib.py` and `freeze/run_b.py` are the exact files whose hashes were recorded at the freeze; they depend on the original project environment and are kept for verification. The package in `code/ofex` is a standalone implementation of the same pipeline; the tests confirm that it builds identical prompts and reproduces all reported numbers from the stored outputs.
- Run labels in `runs/*/meta.json` and `calls.jsonl.gz` were normalised to `dev`, `test1`, `test2`, `test3`; messages, model answers, and records are unchanged.
- Prompts, the annotation guideline, and the provision texts are in Indonesian.
