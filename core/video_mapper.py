from __future__ import annotations

from typing import Dict, List

from services.dataset_service import DatasetService
from utils.text_normalizer import normalize_text


class VideoMapper:
    def __init__(self) -> None:
        self._video_df = DatasetService.load_video_dataset()
        self._by_guid = self._build_guid_lookup()
        self._by_name = self._build_name_lookup()

    def _build_guid_lookup(self) -> Dict[str, dict]:
        lookup: Dict[str, dict] = {}
        for _, row in self._video_df.iterrows():
            guid = str(row.get("guidid", "")).strip().lower()
            if guid:
                lookup[guid] = row.to_dict()
        return lookup

    def _build_name_lookup(self) -> Dict[str, dict]:
        lookup: Dict[str, dict] = {}
        for _, row in self._video_df.iterrows():
            names = [
                row.get("folder_name", ""),
                row.get("video_name", ""),
                row.get("exercise_name", ""),
            ]
            for value in names:
                key = normalize_text(value)
                if key and key not in lookup:
                    lookup[key] = row.to_dict()
        return lookup

    def enrich_exercise(self, exercise: dict) -> dict:
        if self._video_df.empty:
            return exercise

        guid_key = str(exercise.get("guidid", "")).strip().lower()
        row = self._by_guid.get(guid_key)
        if row is None:
            name_key = normalize_text(exercise.get("name", ""))
            row = self._by_name.get(name_key)

        if row is None:
            return exercise

        exercise["video_url"] = str(row.get("sas_video_path") or row.get("video_path") or "")
        exercise["thumbnail_url"] = str(row.get("sas_image_path") or row.get("image_path") or "")
        exercise["video_path"] = str(row.get("video_path") or "")
        exercise["image_path"] = str(row.get("image_path") or "")
        if not exercise.get("benefit"):
            exercise["benefit"] = str(row.get("health_benefit") or "")
        if not exercise.get("safety_cue"):
            exercise["safety_cue"] = str(row.get("safety_cue") or "")
        return exercise

    def enrich_plan(self, plan: Dict[str, dict]) -> Dict[str, dict]:
        for day_data in plan.values():
            for section in ("warmup", "main_workout", "cooldown"):
                items: List[dict] = day_data.get(section, [])
                day_data[section] = [self.enrich_exercise(dict(item)) for item in items]
        return plan
