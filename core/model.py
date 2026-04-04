import os
from typing import List, Dict, Any

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "model.cbm"))

_model = None
_catboost_ok = True


def load_model():
    global _model, _catboost_ok
    if _model is not None or not _catboost_ok:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        _model = model
        return _model
    except Exception:
        _catboost_ok = False
        return None


def score_items(items: List[Dict[str, Any]], features: List[Dict[str, Any]]):
    model = load_model()
    if not model:
        return None
    # CatBoost can handle dict features directly if we pass as list of dicts
    preds = model.predict_proba(features)[:, 1]
    scored = []
    for item, score in zip(items, preds):
        it = dict(item)
        it["score"] = float(score)
        scored.append(it)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
