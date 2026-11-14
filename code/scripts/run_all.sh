#!/bin/bash
# End-to-end order of the experiment. Steps 2 and 3 make paid model calls and need OPENROUTER_API_KEY in the environment.
# The stored outputs in ../data and ../runs are the ones the reported results are computed from. Steps 2 and 3 are
# only needed to repeat the experiment from scratch (model outputs vary slightly between runs).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
$PY scripts/01_samples.py                                              # samples from the corpus (no cost)
# $PY scripts/02_annotate.py --set test --name ref_test_new --ceiling 5   # Reference annotation, about USD 3.8 for 240 provisions
# for n in 1 2 3; do $PY scripts/03_extract.py --set test --name test_new$n --ceiling 0.3; done   # Obligation extraction, about USD 0.17 per run
$PY scripts/04_evaluate.py --set dev  --runs dev                        # Evaluation on the development set (stored run)
$PY scripts/04_evaluate.py --set test --runs test1 test2 test3          # Evaluation on the test set (stored runs)
$PY scripts/05_strata_per_run.py --set test --runs test1 test2 test3     # per-run results by annotator agreement
$PY -m pytest -q tests                                                  # reproduction checks
