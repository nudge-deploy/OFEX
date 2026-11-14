"""Fixed settings of the experiment. Values are those used for the reported results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repository root
DATA = ROOT / "data"
PROMPTS_OFEX = ROOT / "prompts" / "ofex"           # frozen OFEX prompts (see freeze/freeze_record.json)
PROMPTS_ANNOTATION = ROOT / "prompts" / "annotation"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# --- model under test (Obligation extraction) -------------------------------------------------------------------
EXTRACTION_MODEL = "openai/gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS_STEP1 = 3000
MAX_TOKENS_STEP2 = 4000
CONTEXT_CHARS = 6000          # other paragraphs of the same article
CROSSREF_CHARS = 2000         # text of articles the provision cites

# --- reference annotation (Reference annotation) -----------------------------------------------------------------
ANNOTATOR_A = "qwen/qwen3.7-plus"
ANNOTATOR_B = "anthropic/claude-haiku-5.5"
ADJUDICATOR = "anthropic/claude-sonnet-5.5"
ANNOTATION_MAX_TOKENS = 12000
ANNOTATION_CONTEXT_CHARS = 6000

# --- prices, USD per million tokens (OpenRouter list prices on 14 Nov 2026; used only to report cost) ---
PRICES = {EXTRACTION_MODEL: (0.15, 0.60), ANNOTATOR_A: (0.32, 1.28), ANNOTATOR_B: (0.10, 0.50), ADJUDICATOR: (2.00, 10.00)}

# --- sampling -----------------------------------------------------------------------------------
DEV_SIZE, DEV_SEED = 300, 20261112
TEST_SIZE, TEST_SEED = 240, 20261114

# --- evaluation ---------------------------------------------------------------------------------
IOU_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261109
CONFIDENCE_LEVEL = 0.95
CONTAINMENT = 0.9             # deduplication: a span this much inside a longer one is removed

FIELDS = ["actor", "modality", "action", "object", "condition", "threshold", "timing", "exception", "evidence"]
