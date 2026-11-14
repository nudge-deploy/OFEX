"""OFEX (Obligation-First Evidence-grounded eXtraction) of regulatory obligations: research code.

Modules
  config            paths, model names, seeds, prices
  io                JSONL helpers (runs are never overwritten)
  text              text normalisation, null handling, verbatim location of quotations
  corpus            corpus loading and article context
  sampling          marker categories and stratified sampling
  llm               OpenRouter client with call log and cost ceiling
  annotate          Reference annotation (annotator 1, annotator 2, adjudicator)
  extract           Obligation extraction: OFEX extraction (five steps)
  dedup             normalisation and deduplication (step 5)
  evaluate          Evaluation: matching, metrics, confidence intervals, field scores
  keyword_baseline  keyword baseline (no model)
"""
