from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

PRIMARY_GOALS = ["Weight Loss", "Muscle Gain", "Weight Maintenance", "Mobility"]
SECONDARY_GOALS = [
    "Improve Posture",
    "Increase Stamina",
    "Core Strength",
    "Better Sleep",
    "Mobility",
    "Balance",
    "Toning",
    "None",
]
BODY_PARTS = ["Full Body", "Core", "Upper Body", "Lower Body"]
FITNESS_LEVELS = ["Beginner", "Intermediate", "Advanced"]
EQUIPMENT_LIST = [
    "No Equipment",
    "Dumbbells",
    "Resistance Bands",
    "Kettlebell",
    "Bench",
    "Foam Roller",
    "Medicine Ball",
]
DURATIONS = ["2 min", "5 min"]
LOCATIONS = ["Home", "Gym", "Outdoors"]
UNIT_SYSTEMS = ["Metric (kg/cm)", "Imperial (lbs/in)"]
GENDERS = ["Male", "Female", "Non-binary", "Prefer not to say"]

DEFAULT_DAY_SPLITS = {
    "full body": ["Full Body Strength", "Cardio Conditioning", "Mobility Recovery"],
    "upper body": ["Upper Strength", "Cardio Conditioning", "Mobility Recovery"],
    "lower body": ["Lower Strength", "Cardio Conditioning", "Mobility Recovery"],
    "core": ["Core Strength", "Cardio Conditioning", "Mobility Recovery"],
}
