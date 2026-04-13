from __future__ import annotations

from typing import Any, Dict

from core.fitness_engine import FitnessEngine


class WorkoutService:
    def __init__(self) -> None:
        self.engine = FitnessEngine()

    def build_plan(self, profile: Dict[str, Any]) -> Dict[str, dict]:
        return self.engine.generate_plan(profile)
