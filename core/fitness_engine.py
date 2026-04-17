from __future__ import annotations

import asyncio
import copy
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core import fitness as legacy_fitness
from core.video_mapper import VideoMapper


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "fitness.csv"
SOAP_DATASET_PATH = PROJECT_ROOT / "dataset" / "soap_data.csv"


legacy_fitness.BASE_DIR = str(PROJECT_ROOT)
legacy_fitness.DATASETS_DIR = str(PROJECT_ROOT / "dataset")
legacy_fitness.FitnessDataset.POSSIBLE_PATHS = [str(DATASET_PATH)]


def _sample_rotation_candidates_compat(frame: pd.DataFrame) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None

    weights = 1.0 / (1.0 + frame["_score"].clip(lower=0))
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    positive_count = int((weights > 0).sum())
    if positive_count <= 0:
        return None

    candidate_count = min(15, len(frame), positive_count)
    available = list(range(len(frame)))
    weight_list = weights.astype(float).tolist()
    chosen: list[int] = []
    for _ in range(candidate_count):
        active_weights = [weight_list[idx] for idx in available]
        total = sum(active_weights)
        if total <= 0:
            break
        selected_pos = random.choices(available, weights=active_weights, k=1)[0]
        chosen.append(selected_pos)
        available.remove(selected_pos)

    if not chosen:
        return None

    candidates = frame.iloc[chosen]
    if candidates.empty:
        return None
    return candidates.iloc[random.randrange(len(candidates))]


legacy_fitness._sample_rotation_candidates = _sample_rotation_candidates_compat


ToolResult = legacy_fitness.ToolResult
ExerciseFilter = legacy_fitness.ExerciseFilter
_hard_medical_exclusion = legacy_fitness._hard_medical_exclusion
_classify_exercise_categories = legacy_fitness._classify_exercise_categories
_classify_warmup_bucket = legacy_fitness._classify_warmup_bucket
_classify_cooldown_bucket = legacy_fitness._classify_cooldown_bucket
_current_day_focus = legacy_fitness._current_day_focus
_apply_day_based_main_variation = legacy_fitness._apply_day_based_main_variation
_prepare_rotation_frame = legacy_fitness._prepare_rotation_frame
_select_rotated_dataset_row = legacy_fitness._select_rotated_dataset_row


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(profile or {})
    if not normalized.get("primary_goal") and normalized.get("goal"):
        normalized["primary_goal"] = normalized["goal"]
    if not normalized.get("goal") and normalized.get("primary_goal"):
        normalized["goal"] = normalized["primary_goal"]

    explicit_days = normalized.get("days")
    requested_days = normalized.get("days_per_week")
    if isinstance(explicit_days, list) and explicit_days:
        normalized["days_per_week"] = explicit_days
    elif isinstance(requested_days, list) and requested_days:
        normalized["days_per_week"] = requested_days
    else:
        weekly_days = int(normalized.get("weekly_days") or 3)
        normalized["days_per_week"] = legacy_fitness._DAY_ORDER[: max(1, min(weekly_days, len(legacy_fitness._DAY_ORDER)))]

    if normalized.get("body_region") == "Upper Body":
        normalized["body_region"] = "Upper"
    elif normalized.get("body_region") == "Lower Body":
        normalized["body_region"] = "Lower"

    if normalized.get("blood_pressure") and not normalized.get("bp"):
        normalized["bp"] = normalized["blood_pressure"]

    normalized.setdefault("fitness_level", "Beginner")
    normalized.setdefault("session_duration", "30 min")
    normalized.setdefault("weight_kg", 70)
    return normalized


def _seed_for_profile(profile: Dict[str, Any]) -> int:
    payload = repr(sorted(_normalize_profile(profile).items())).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest()[:8], 16)


def _execute_legacy_plan(profile: Dict[str, Any]) -> ToolResult:
    planner = legacy_fitness.FitnessPlanGeneratorTool()
    normalized = _normalize_profile(profile)
    state = random.getstate()
    random.seed(_seed_for_profile(normalized))
    try:
        return _run_async(planner.execute(constraints=normalized))
    finally:
        random.setstate(state)


def run_old_engine(profile: Dict[str, Any]) -> Dict[str, Any]:
    result = _execute_legacy_plan(profile)
    if not result.success:
        raise RuntimeError(result.error or "Legacy fitness engine failed.")
    return result.data.get("plans_json") or result.data.get("json_plan") or {}


def generate_plan_local_from_dataset(profile: Dict[str, Any], dataset_path: str, *, enrich_video: bool = True) -> Dict[str, Any]:
    old_paths = list(legacy_fitness.FitnessDataset.POSSIBLE_PATHS)
    legacy_fitness.FitnessDataset.POSSIBLE_PATHS = [dataset_path] + old_paths
    try:
        plan = run_old_engine(profile)
    finally:
        legacy_fitness.FitnessDataset.POSSIBLE_PATHS = old_paths
    if enrich_video:
        return VideoMapper("dataset/Exercise videos.csv").enrich_plan(copy.deepcopy(plan))
    return plan


def generate_plan_local(profile: Dict[str, Any], *, enrich_video: bool = True) -> Dict[str, Any]:
    plan = run_old_engine(profile)
    if enrich_video:
        return VideoMapper("dataset/Exercise videos.csv").enrich_plan(copy.deepcopy(plan))
    return plan


def run_new_engine(profile: Dict[str, Any]) -> Dict[str, Any]:
    return generate_plan_local(profile, enrich_video=False)


def validate_equivalence(profile: Dict[str, Any]) -> None:
    old = run_old_engine(profile)
    new = run_new_engine(profile)
    assert old == new


class FitnessEngine:
    def __init__(self) -> None:
        self.video_mapper = VideoMapper(csv_path="dataset/Exercise videos.csv")

    def generate_plan(self, profile: Dict[str, Any]) -> Dict[str, dict]:
        return generate_plan_local(profile, enrich_video=True)

    def run_ground_truth(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return run_old_engine(profile)

    def validate_equivalence(self, profile: Dict[str, Any]) -> None:
        validate_equivalence(profile)

    def apply_filters(self, df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        return ExerciseFilter.apply_filters(df, _normalize_profile(profile))

    def apply_medical_guardrails(
        self,
        df: pd.DataFrame,
        context: Optional[Dict[str, Any]],
        slot_type: str = "main",
    ) -> pd.DataFrame:
        return _hard_medical_exclusion(df, context, slot_type)
