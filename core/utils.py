import json
import os
from typing import Any, Dict, Optional


def read_json(path: str) -> Dict[str, Any]:
    """Read JSON payload.

    Args:
        path: JSON path or '-'.

    Returns:
        Parsed JSON dict.
    """
    if not path or path == "-":
        return json.load(os.sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Dict[str, Any]) -> None:
    """Write JSON to stdout.

    Args:
        data: JSON serializable dict.

    Returns:
        None.
    """
    print(json.dumps(data, ensure_ascii=False, indent=2))


def file_mtime(path: str) -> Optional[float]:
    """Return file mtime.

    Args:
        path: File path.

    Returns:
        mtime or None.
    """
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


def load_cache(path: str) -> Optional[Dict[str, Any]]:
    """Load cache payload from disk.

    Args:
        path: Cache file path.

    Returns:
        Cache dict or None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(path: str, payload: Dict[str, Any]) -> None:
    """Save cache payload to disk.

    Args:
        path: Cache file path.
        payload: Cache dict.

    Returns:
        None.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
