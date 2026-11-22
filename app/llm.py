import time

import numpy as np
from openai import OpenAI

from .config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # A stuck call must fail, not hang a run: the SDK default waits up to 10 minutes per try.
        _client = OpenAI(api_key=OPENAI_API_KEY, timeout=120, max_retries=2)
    return _client


def embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 100):
        resp = client().embeddings.create(model=EMBED_MODEL, input=texts[i : i + 100])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def chat(system: str, user: str, json_mode: bool = False) -> tuple[str, dict]:
    """Returns the reply and {tokens_in, tokens_out, latency_ms} for the log."""
    kwargs = {}
    # Reasoning models (o-series, gpt-5) reject a temperature setting.
    if not CHAT_MODEL.startswith(("o", "gpt-5")):
        kwargs["temperature"] = 0
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # Caps runaway output (JSON mode can stream whitespace until the limit). Far above any real reply.
    kwargs["max_completion_tokens"] = 8000
    started = time.monotonic()
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    usage = getattr(resp, "usage", None)
    meta = {
        "tokens_in": getattr(usage, "prompt_tokens", None),
        "tokens_out": getattr(usage, "completion_tokens", None),
        "latency_ms": int((time.monotonic() - started) * 1000),
    }
    return resp.choices[0].message.content or "", meta
