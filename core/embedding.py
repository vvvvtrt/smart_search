import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("search.embedding")

_MODEL = None
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
MODEL_PATH = os.environ.get("ST_MODEL_PATH", DEFAULT_MODEL)
OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


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
        _MODEL = SentenceTransformer(MODEL_PATH)
    except Exception as exc:
        logger.warning("embedding_model_load_failed", extra={"error": str(exc), "model": MODEL_PATH, "offline": OFFLINE})
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
    vec = model.encode([text], normalize_embeddings=True)
    return vec
