from __future__ import annotations

from functools import lru_cache
import re
from typing import Dict

import pandas as pd

from utils.constants import DATASET_DIR
from utils.text_normalizer import normalize_text


class DatasetService:
    FITNESS_FILE = DATASET_DIR / "fitness.csv"
    VIDEO_FILE = DATASET_DIR / "Exercise videos.csv"
    SOAP_FILE = DATASET_DIR / "soap_data.csv"

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
            "Video Link": "video_url",
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
        df["reps"] = df["reps"].apply(cls._sanitize_reps)
        df["tags"] = df["tags"].apply(cls._normalize_tags)
        df["goal"] = df["goal"].replace("", "All")
        df["fitness_level"] = df["fitness_level"].replace("", "Beginner")
        return df

    @classmethod
    @lru_cache(maxsize=1)
    def load_soap_dataset(cls) -> pd.DataFrame:
        if not cls.SOAP_FILE.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(cls.SOAP_FILE)
        except Exception:
            return pd.DataFrame()

        df.columns = [str(col).strip() for col in df.columns]
        aliases = {
            "Unique ID": "unique_id",
            "GuidId": "guidid",
            "Exercise Name": "exercise_name",
            "Video name": "video_name",
            "Video Link": "video_url",
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
            "rest",
            "rest_intervals",
            "health_benefit",
            "steps_to_perform",
            "safety_cue",
            "met_value",
            "is_not_suitable_for",
            "tags",
            "video_name",
            "video_url",
        ]
        for column in required_columns:
            if column not in df.columns:
                df[column] = ""

        df = df.fillna("")
        df["reps"] = df["reps"].apply(cls._sanitize_reps)
        df["tags"] = df["tags"].apply(cls._normalize_tags)
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
            "Video name": "video_name",
            "Video Link": "video_url",
            "Notification": "notification",
            "Exercise Name": "exercise_name",
            "video_name": "video_name",
            "video_path": "video_path",
            "sas_video_path": "sas_video_path",
            "sas_image_path": "sas_image_path",
            "image_path": "image_path",
            "health_benefit": "health_benefit",
            "Health benefit": "health_benefit",
            "safety_cue": "safety_cue",
            "Safety cue": "safety_cue",
        }
        df = df.rename(columns=aliases)
        df = df.fillna("")
        if "exercise_name" not in df.columns:
            if "folder_name" in df.columns:
                df["exercise_name"] = df["folder_name"].apply(lambda value: str(value).replace("_", " "))
            else:
                df["exercise_name"] = ""
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

    @staticmethod
    def _sanitize_reps(value: object) -> str:
        if isinstance(value, int) and value > 100:
            return "10-15"
        text = str(value or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return "10-12"
        numbers = [int(part) for part in re.findall(r"\d+", text)]
        if re.fullmatch(r"\d{4,}", text) or any(number > 300 for number in numbers):
            return "10-15"
        if any(char.isdigit() for char in text):
            return text
        return "10-12"
