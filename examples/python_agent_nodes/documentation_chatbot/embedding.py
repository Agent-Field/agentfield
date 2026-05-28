"""Shared embedding helper built on FastEmbed for the doc chatbot."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

from fastembed import TextEmbedding


@lru_cache(maxsize=1)
def _load_model() -> TextEmbedding:
    model_name = os.getenv("DOC_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return TextEmbedding(model_name=model_name)


def embed_texts(texts: Iterable[str]) -> List[List[float]]:
    """Embed an iterable of strings and return Python lists."""

    model = _load_model()
    embeddings = list(model.embed(list(texts)))
    return [vector.tolist() for vector in embeddings]


def embed_query(text: str) -> List[float]:
    """Shortcut for single-question embeddings."""

    return embed_texts([text])[0]
