"""OpenRouter chat client. Every call is appended to a JSONL log with messages, raw answer, tokens, latency, and cost.
The API key is read from the environment variable OPENROUTER_API_KEY and is never written anywhere."""
import hashlib, json, os, threading, time
from pathlib import Path
from .config import PRICES

URL = "https://openrouter.ai/api/v1/chat/completions"
_lock = threading.Lock()


class Budget:
    """Stops every further call once the logged cost reaches the ceiling."""
    def __init__(self, ceiling_usd: float, already_spent: float = 0.0):
        self.ceiling, self.spent = ceiling_usd, already_spent
    def check(self):
        if self.spent >= self.ceiling:
            raise SystemExit(f"cost ceiling reached: {self.spent:.4f} USD")
    def add(self, usd: float):
        with _lock: self.spent += usd


def chat(model: str, messages: list[dict], *, max_tokens: int, response_format: dict | None = None, temperature: float = 0.0,
         log_path: Path | None = None, tag: dict | None = None, budget: Budget | None = None, tries: int = 3) -> dict:
    """One chat call (with up to `tries` attempts on transport errors). Returns {ok, text, tokens_in, tokens_out, cost_usd, ...}."""
    import httpx
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise SystemExit("set OPENROUTER_API_KEY in the environment")
    if budget: budget.check()
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_format: payload["response_format"] = response_format
    rec = {"model_requested": model, "temperature": temperature, "max_tokens": max_tokens, **(tag or {})}
    for attempt in range(1, tries + 1):
        t0 = time.time()
        try:
            r = httpx.post(URL, json=payload, timeout=240, headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
            body = r.json()
            if r.status_code != 200 or "choices" not in body:
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(body)[:300]}")
            u = body.get("usage") or {}; pin, pout = PRICES.get(model, (0, 0)); ch = body["choices"][0]
            rec.update(ok=True, attempts=attempt, latency_ms=int((time.time() - t0) * 1000), model_returned=body.get("model"), provider=body.get("provider"),
                       tokens_in=u.get("prompt_tokens", 0), tokens_out=u.get("completion_tokens", 0), finish_reason=ch.get("finish_reason"),
                       text=ch["message"].get("content") or "")
            rec["cost_usd"] = round((rec["tokens_in"] * pin + rec["tokens_out"] * pout) / 1e6, 6)
            break
        except Exception as exc:  # noqa: BLE001 - logged, retried
            rec.update(ok=False, attempts=attempt, error=str(exc)[:400], latency_ms=int((time.time() - t0) * 1000), text="", cost_usd=0.0)
            time.sleep(2 * attempt)
    rec["prompt_sha256"] = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
    if budget: budget.add(rec.get("cost_usd") or 0)
    if log_path is not None:
        with _lock:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f: f.write(json.dumps({**rec, "messages": messages}, ensure_ascii=False) + "\n")
    return rec


def schema_format(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
