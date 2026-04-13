from __future__ import annotations

from datetime import datetime


def build_download_name(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{prefix}_{stamp}.json"
