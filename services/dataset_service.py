from __future__ import annotations

from functools import lru_cache
from typing import Dict

import pandas as pd

from utils.constants import DATASET_DIR
from utils.text_normalizer import normalize_text


class DatasetService:
    FITNESS_FILE = DATASET_DIR / "fitness.csv"
    VIDEO_FILE = DATASET_DIR / "Exercise videos.csv"

    @classmethod
    @lru_cache(maxsize=1)
    def load_fitness_dataset(cls) -> pd.DataFrame:
        if not cls.FITNESS_FILE.exists():
            return pd.DataFrame()

        df = pd.read_csv(cls.FITNESS_FILE)
        df.columns = [str(col).strip() for col in df.columns]
        aliases: Dict[str, str] = {
            "Unique ID": "unique_id",
            "GuidId": "guidid",
            "Exercise Name": "exercise_name",
            "Age Suitability": "age_suitability",
            "Goal": "goal",
            "Primary Category": "primary_category",
            "Body Region": "body_region",
            "Equipments": "equipments",
            "Fitness Level": "fitness_level",
            "Physical limitation": "physical_limitations",
            "Physical limitations": "physical_limitations",
            "Sets": "sets",
            "Reps": "reps",
            "RPE": "rpe",
            "Rest": "rest",
            "Rest intervals": "rest_intervals",
            "Health benefit": "health_benefit",
            "Steps to perform": "steps_to_perform",
            "Safety cue": "safety_cue",
            "MET value": "met_value",
            "is_not_suitable_for": "is_not_suitable_for",
            "Tags": "tags",
            "Video name": "video_name",
        }
        df = df.rename(columns=aliases)

        required_columns = [
            "unique_id",
            "guidid",
            "exercise_name",
            "age_suitability",
            "goal",
            "primary_category",
            "body_region",
            "equipments",
            "fitness_level",
            "physical_limitations",
            "sets",
            "reps",
            "rpe",
            "rest_intervals",
            "health_benefit",
            "steps_to_perform",
            "safety_cue",
            "met_value",
            "is_not_suitable_for",
            "tags",
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = ""

        df = df.fillna("")
        df["tags"] = df["tags"].apply(cls._normalize_tags)
        df["goal"] = df["goal"].replace("", "All")
        df["fitness_level"] = df["fitness_level"].replace("", "Beginner")
        return df

    @classmethod
    @lru_cache(maxsize=1)
    def load_video_dataset(cls) -> pd.DataFrame:
        if not cls.VIDEO_FILE.exists():
            return pd.DataFrame()

        df = pd.read_csv(cls.VIDEO_FILE)
        df.columns = [str(col).strip() for col in df.columns]
        aliases = {
            "GuidId": "guidid",
            "folder_name": "folder_name",
            "video_name": "video_name",
            "video_path": "video_path",
            "sas_video_path": "sas_video_path",
            "sas_image_path": "sas_image_path",
            "image_path": "image_path",
            "health_benefit": "health_benefit",
            "safety_cue": "safety_cue",
        }
        df = df.rename(columns=aliases)
        df = df.fillna("")
        if "exercise_name" not in df.columns:
            df["exercise_name"] = df["folder_name"].apply(lambda value: str(value).replace("_", " "))
        return df

    @staticmethod
    def _normalize_tags(value: str) -> str:
        text = normalize_text(value)
        replacements = {
            "warmup": "warm up",
            "cool down": "cooldown",
            "main workout": "main workout",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
