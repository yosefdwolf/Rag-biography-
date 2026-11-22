import numpy as np
from openai import OpenAI

from .config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), 100):
        resp = client().embeddings.create(model=EMBED_MODEL, input=texts[i : i + 100])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def chat(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {}
    # Reasoning models (o-series, gpt-5) reject a temperature setting.
    if not CHAT_MODEL.startswith(("o", "gpt-5")):
        kwargs["temperature"] = 0
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return resp.choices[0].message.content or ""
