from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExerciseItem:
    name: str
    exercise_name: str
    guidid: str = ""
    unique_id: str = ""
    category: str = "main"
    equipment: str = ""
    sets: str = ""
    reps: str = ""
    intensity_rpe: str = ""
    rest: str = ""
    benefit: str = ""
    safety_cue: str = ""
    steps: List[str] = field(default_factory=list)
    body_region: str = ""
    primary_category: str = ""
    tags: str = ""
    video_url: str = ""
    thumbnail_url: str = ""
    video_path: str = ""
    image_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DayPlan:
    day_name: str
    main_workout_category: str
    warmup: List[Dict[str, Any]] = field(default_factory=list)
    main_workout: List[Dict[str, Any]] = field(default_factory=list)
    cooldown: List[Dict[str, Any]] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_name": self.day_name,
            "main_workout_category": self.main_workout_category,
            "warmup": self.warmup,
            "main_workout": self.main_workout,
            "cooldown": self.cooldown,
            "safety_notes": self.safety_notes,
        }


@dataclass
class SoapParseResult:
    source_text: str
    history: str = ""
    findings: str = ""
    plan_of_action: str = ""
    exercise_session: str = ""
    structured_vitals: Dict[str, str] = field(default_factory=dict)
    inferred_profile: Dict[str, Any] = field(default_factory=dict)
    prescribed_exercises: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    frequency_per_week: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
