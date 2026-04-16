from typing import Any, Dict, Iterable, Optional

import pandas as pd


class VideoMapper:
    def __init__(self, csv_path: str = "dataset/Exercise videos.csv"):
        self.df = pd.read_csv(csv_path)
        self.df.columns = [str(col).strip() for col in self.df.columns]
        self.df = self.df.rename(
            columns={
                "GuidId": "guidid",
                "Video Link": "video_url",
                "sas_video_path": "video_url",
                "sas_image_path": "thumbnail_url",
                "Notification": "notification",
                "Exercise Name": "exercise_name",
                "Health benefit": "health_benefit",
                "Safety cue": "safety_cue",
            }
        ).fillna("")

    def _lookup_row(self, guid: Any) -> Optional[Dict[str, Any]]:
        guid_text = str(guid or "").strip()
        if not guid_text or "guidid" not in self.df.columns:
            return None

        row = self.df[self.df["guidid"].astype(str).str.strip() == guid_text]
        if row.empty:
            return None

        return row.iloc[0].to_dict()

    def enrich_exercise(self, exercise: Dict[str, Any]) -> Dict[str, Any]:
        guid = (
            exercise.get("guidid")
            or exercise.get("guid")
            or exercise.get("guid_id")
            or exercise.get("gui_id")
            or exercise.get("GUIID")
        )
        video_row = self._lookup_row(guid)
        if not video_row:
            return exercise

        exercise["video_url"] = str(video_row.get("video_url", "")).strip()
        exercise["thumbnail_url"] = str(video_row.get("thumbnail_url", "")).strip()
        exercise["video_path"] = str(video_row.get("video_path", "")).strip()
        exercise["image_path"] = str(video_row.get("image_path", "")).strip()
        if not exercise.get("benefit"):
            exercise["benefit"] = str(video_row.get("health_benefit", "")).strip()
        if not exercise.get("safety_cue"):
            exercise["safety_cue"] = str(video_row.get("safety_cue", "")).strip()
        return exercise

    def enrich_plan(self, plan: Dict[str, Dict[str, Iterable[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Adds video and thumbnail fields to each exercise using the exercise guid.
        Supports the weekly plan structure returned by the workout engine.
        """
        if not isinstance(plan, dict):
            return plan

        for day_data in plan.values():
            if not isinstance(day_data, dict):
                continue

            for section_key in ("warmup", "main_workout", "cooldown", "exercises"):
                exercises = day_data.get(section_key, [])
                if not isinstance(exercises, list):
                    continue

                for exercise in exercises:
                    if isinstance(exercise, dict):
                        self.enrich_exercise(exercise)
        return plan
