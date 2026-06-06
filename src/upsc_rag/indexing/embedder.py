"""OpenAI embedding client with batched requests and retry logic."""
from __future__ import annotations

import os
import time
from typing import Sequence

from openai import OpenAI


def embed_texts(
    texts: Sequence[str],
    *,
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI API, returning vectors in the same order."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    vectors: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        for attempt in range(3):
            try:
                response = client.embeddings.create(input=batch, model=model)
                batch_vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                vectors.extend(batch_vectors)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"  Embedding attempt {attempt + 1} failed ({exc}), retrying in {wait}s…")
                time.sleep(wait)

    return vectors
