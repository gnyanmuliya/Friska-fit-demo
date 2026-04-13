from __future__ import annotations

from typing import Any, Dict

from core.soap_engine import SoapEngine


class SoapParserService:
    def __init__(self) -> None:
        self.engine = SoapEngine()

    def parse_and_build_plan(self, source_text: str) -> Dict[str, Any]:
        return self.engine.generate_plan_from_text(source_text)
