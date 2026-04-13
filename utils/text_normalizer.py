from __future__ import annotations

import re
from typing import Any, List


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_steps(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if "\n" in text:
        parts = [part.strip(" -\t") for part in text.splitlines()]
    else:
        parts = [part.strip(" -\t") for part in re.split(r"[.;]", text)]
    return [part for part in parts if part]


def safe_int(value: Any, default: int = 0) -> int:
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else default


def safe_float(value: Any, default: float = 0.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    return float(match.group(1)) if match else default
