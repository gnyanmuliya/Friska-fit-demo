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

PRIMARY_GOALS = ["Weight Loss", "Weight Gain", "Weight Maintenance"]
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
MEDICAL_CONDITIONS = [
    "Hypertension",
    "High Cholesterol / Dyslipidemia",
    "Obesity / Overweight",
    "Type 2 Diabetes",
    "Prediabetes",
    "Metabolic Syndrome",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Osteoarthritis",
    "Osteoporosis",
    "Depression / Anxiety",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Sleep Apnea",
    "Fatty Liver Disease (NAFLD)",
    "COPD",
    "Heart Failure (Stable)",
    "Sarcopenia",
    "Chronic Low Back Pain",
    "Cognitive Decline / Dementia",
    "Cancer (Lifestyle-related / Post-treatment)",
]
PHYSICAL_LIMITATIONS = [
    "Acute Cervical Injury",
    "Neck Pain",
    "Thoracic Stiffness (Upper Back Rigidity)",
    "Poor Posture (Rounded Shoulders / Forward Head)",
    "Acute Shoulder Pain",
    "Shoulder Impingement",
    "Shoulder Pain (General)",
    "Limited Shoulder Range of Motion",
    "Limited Overhead Reach",
    "Rotator Cuff Weakness / Instability",
    "Acute Elbow Pain",
    "Elbow Pain",
    "Acute Wrist Pain",
    "Wrist Pain / Limited Wrist Mobility",
    "Acute Low Back Pain",
    "Acute Lumbar Disc Injury",
    "Lumbar Stiffness / Restricted Spinal Mobility",
    "Lower Back Pain (Non-Specific)",
    "Acute Hip Pain",
    "Hip Pain",
    "Hip Stiffness / Limited Hip Mobility",
    "Acute Knee Injury",
    "Acute Knee Pain",
    "Knee Injury (ACL / Meniscus / Post-Surgery)",
    "Knee Pain (General / Chronic)",
    "Ankle Pain",
    "Ankle Instability / Recurrent Sprains",
    "Foot Pain (Plantar Fasciitis / Heel Pain)",
    "Acute Hamstring Tear",
    "Limited Single-Leg Balance",
    "Reduced Core Stability",
    "General Joint Stiffness (Multi-Joint / Morning Stiffness)",
]
LOCATIONS = ["Home", "Gym", "Outdoors"]
UNIT_SYSTEMS = ["Metric (kg/cm)", "Imperial (lbs/in)"]
GENDERS = ["Male", "Female", "Non-binary", "Prefer not to say"]

DEFAULT_DAY_SPLITS = {
    "full body": ["Full Body Strength", "Cardio Conditioning", "Mobility Recovery"],
    "upper body": ["Upper Strength", "Cardio Conditioning", "Mobility Recovery"],
    "lower body": ["Lower Strength", "Cardio Conditioning", "Mobility Recovery"],
    "core": ["Core Strength", "Cardio Conditioning", "Mobility Recovery"],
}
