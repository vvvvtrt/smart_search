import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("search.embedding")

_MODEL = None
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
MODEL_PATH = os.environ.get("ST_MODEL_PATH", DEFAULT_MODEL)
OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
_QUERY_PREFIX = "query: "


def _maybe_prefix_query(text: str) -> str:
    if text.lower().startswith("query:"):
        return text
    return f"{_QUERY_PREFIX}{text}"


def load_model():
    """Load sentence-transformers model once.

    Args:
        None.

    Returns:
        Model instance or None.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence_transformers_not_installed")
        return None
    try:
        if os.path.isdir(MODEL_PATH):
            _MODEL = SentenceTransformer(MODEL_PATH, device="cpu")
        else:
            _MODEL = SentenceTransformer(DEFAULT_MODEL)
    except Exception as exc:
        logger.warning("embedding_model_load_failed: %s (model=%s offline=%s)", exc, MODEL_PATH, OFFLINE)
        return None
    return _MODEL


def encode_query(text: str) -> Optional[np.ndarray]:
    """Encode query to normalized vector.

    Args:
        text: Input text.

    Returns:
        Numpy array or None.
    """
    model = load_model()
    if model is None:
        return None
    vec = model.encode([_maybe_prefix_query(text)], normalize_embeddings=True)
    return vec
