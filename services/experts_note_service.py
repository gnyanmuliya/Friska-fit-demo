from __future__ import annotations

import logging
import math
import random
import re
from typing import Any, Dict, List, Set

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None

import pandas as pd

from core.azure_ai_parser import AzureAIContentFilterError, AzureAIPrescriptionParser
from core.fitness import (
    ExerciseFilter,
    FitnessDataset,
    _hard_medical_exclusion,
)
from core.fitness_engine import FitnessEngine
from utils.constants import DAY_ORDER, DATASET_DIR

logger = logging.getLogger(__name__)


# ============ CLINICAL CONTEXT PARSING ============

def _extract_medical_conditions(notes_text: str) -> List[str]:
    """Extract medical conditions from notes text."""
    conditions_keywords = {
        "diabetes": ["diabetes", "diabetic", "blood sugar"],
        "hypertension": ["hypertension", "high blood pressure", "elevated bp"],
        "obesity": ["obese", "obesity", "overweight"],
        "arthritis": ["arthritis", "arthritic"],
        "osteoporosis": ["osteoporosis", "bone density"],
        "asthma": ["asthma", "respiratory"],
        "copd": ["copd", "lung"],
        "heart disease": ["cardiac", "heart disease", "cardiovascular"],
    }
    
    text_lower = notes_text.lower()
    found_conditions = []
    
    for condition, keywords in conditions_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_conditions.append(condition)
    
    return found_conditions


def _extract_physical_limitations(notes_text: str) -> List[str]:
    """Extract physical limitations and injuries from notes text."""
    limitations_keywords = {
        "knee pain": ["knee pain", "knee injury", "acl", "meniscus", "patella"],
        "low back pain": ["low back", "lumbar", "back pain", "sciatica", "herniated disc"],
        "shoulder pain": ["shoulder pain", "rotator cuff", "impingement"],
        "neck pain": ["neck pain", "cervical", "stiff neck"],
        "hip pain": ["hip pain", "hip injury"],
        "ankle pain": ["ankle pain", "ankle injury", "sprain"],
        "wrist pain": ["wrist pain", "carpal tunnel"],
        "elbow pain": ["elbow pain", "tennis elbow"],
        "tendinitis": ["tendinitis", "tendon"],
        "bursitis": ["bursitis", "bursa"],
    }
    
    text_lower = notes_text.lower()
    found_limitations = []
    
    for limitation, keywords in limitations_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_limitations.append(limitation)
    
    return found_limitations


def _extract_activity_restrictions(notes_text: str) -> List[str]:
    """Extract activity restrictions and avoidances from notes text."""
    restriction_keywords = {
        "avoid jumping": ["avoid jumping", "no jumping", "no jump", "no plyometric", "no plyo"],
        "avoid high impact": ["avoid high impact", "no high impact", "low impact only"],
        "avoid running": ["avoid running", "no running", "no cardio"],
        "avoid floor work": ["avoid floor", "cannot lie down", "no floor work", "no prone"],
        "avoid overhead": ["avoid overhead", "no overhead", "overhead restricted"],
        "avoid spinal loading": ["avoid spinal loading", "spinal load", "heavy spine"],
        "avoid bending": ["avoid bending", "no bending", "avoid forward bend"],
        "avoid twisting": ["avoid twisting", "no rotation", "avoid rotation"],
        "avoid heavy weights": ["avoid heavy", "no heavy", "light weight only"],
    }
    
    text_lower = notes_text.lower()
    found_restrictions = []
    
    for restriction, keywords in restriction_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_restrictions.append(restriction)
    
    return found_restrictions


def _extract_prescribed_exercises(notes_text: str) -> List[Dict[str, Any]]:
    """Extract explicitly prescribed exercises from notes."""
    # Pattern to match exercise prescriptions like "Bird Dog 3x10" or "Wall Push-ups: 2 sets"
    pattern = r"([a-z\s\-\(\)]+?)(?:\s*(?:x|sets?|reps?|times?)\s*(\d+))?(?:[\s,;]|$)"
    
    exercise_keywords = ["exercise", "perform", "do", "practice", "prescribed"]
    
    prescribed = []
    text_lower = notes_text.lower()
    
    # Look for sections that mention exercises
    for keyword in exercise_keywords:
        if keyword in text_lower:
            # Extract sentences or phrases around exercise mentions
            sentences = re.split(r'[.!?]', notes_text)
            for sent in sentences:
                if keyword in sent.lower():
                    # Try to find exercise names (capitalized words)
                    words = sent.split()
                    for i, word in enumerate(words):
                        if word[0].isupper() and len(word) > 2:
                            name = word.strip("(),:")
                            if name and name not in ["The", "A", "An", "For", "If", "And", "Or"]:
                                # Try to extract reps if present
                                reps_match = re.search(r'(\d+)\s*(?:x|reps?|times?)', sent)
                                reps = reps_match.group(1) if reps_match else None
                                prescribed.append({
                                    "name": name,
                                    "reps": reps or "10",
                                })
    
    return prescribed


def _create_clinical_flags(notes_text: str, limitations: List[str], restrictions: List[str]) -> Dict[str, bool]:
    """Create clinical flags for safety guardrails."""
    text_lower = notes_text.lower()
    flags = {}
    
    # Knee-related flags
    if any(term in " ".join(limitations).lower() for term in ["knee", "acl", "meniscus"]):
        flags["knee_sensitive"] = True
    elif "avoid jumping" in " ".join(restrictions).lower():
        flags["knee_sensitive"] = True
    
    # Floor work restrictions
    if "avoid floor work" in " ".join(restrictions).lower():
        flags["avoid_floor_work"] = True
    elif any(term in text_lower for term in ["cannot lie down", "difficulty getting up", "floor contraindicated"]):
        flags["avoid_floor_work"] = True
    
    # High impact restrictions
    if any(term in " ".join(restrictions).lower() for term in ["avoid high impact", "high impact restricted"]):
        flags["high_impact_restricted"] = True
    
    # Spinal loading restrictions
    if "avoid spinal loading" in " ".join(restrictions).lower():
        flags["avoid_spinal_loading"] = True
    
    # Overhead restrictions
    if "avoid overhead" in " ".join(restrictions).lower():
        flags["overhead_restricted"] = True
    
    return flags


# ============ EXPERTS NOTE SERVICE ============

class ExpertsNoteService:
    """
    Generates workout plans from expert/doctor notes without user intake form.
    Uses default health parameters and actively parses notes to influence plan generation.
    """

    def __init__(self) -> None:
        # Initialize fitness engine for plan generation
        self.engine = FitnessEngine()
        # Initialize AI parser for doctor's notes
        self.ai_parser = AzureAIPrescriptionParser()
        # Load dataset for filtering verification
        fitness_csv = DATASET_DIR / "fitness.csv"
        old_paths = list(FitnessDataset.POSSIBLE_PATHS)
        FitnessDataset.POSSIBLE_PATHS = [str(fitness_csv)] + old_paths
        try:
            self.dataset = FitnessDataset.load()
        finally:
            FitnessDataset.POSSIBLE_PATHS = old_paths

    @staticmethod
    def extract_pdf_text(pdf_file: Any) -> str:
        """Extract text from PDF file."""
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Use pasted text or install requirements.txt.")
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text

    @staticmethod
    def _get_default_profile() -> Dict[str, Any]:
        """Create default user profile for plan generation."""
        return {
            "age": 30,
            "gender": "Male",
            "weight_kg": 70,
            "height_cm": 170,
            "fitness_level": "Beginner",
            "primary_goal": "General Fitness",
            "secondary_goal": "None",
            "days_per_week": ["Monday", "Wednesday", "Friday"],
            "weekly_days": 3,
            "days": ["Monday", "Wednesday", "Friday"],
            "session_duration": "30-45 minutes",
            "available_equipment": ["Bodyweight Only"],
            "workout_location": "Home",
            "target_body_parts": ["Full Body"],
            "medical_conditions": [],
            "physical_limitation": "None",
            "specific_avoidance": "None",
            "blood_pressure": "N/A",
            "flags": {},
        }

    def _create_profile_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, Any]:
        """Create profile dictionary from AI prescription."""
        # Extract days from weekly schedule
        weekly_schedule = ai_prescription.get("weekly_schedule", {})
        active_days = [day.capitalize() for day, activity in weekly_schedule.items() 
                      if activity and activity.lower() not in ["rest", "rest or bike ride/brisk walk"]]
        
        return {
            "age": 30,
            "gender": "Male",
            "weight_kg": 70,
            "height_cm": 170,
            "fitness_level": "Beginner",
            "primary_goal": ai_prescription.get("goal", "General Fitness"),
            "secondary_goal": "None",
            "days_per_week": active_days,
            "weekly_days": len(active_days),
            "days": active_days,
            "session_duration": ai_prescription.get("session_duration", "45-60 minutes"),
            "available_equipment": self._display_equipment_list(self._extract_available_equipment(ai_prescription)),
            "workout_location": "Home",
            "target_body_parts": ai_prescription.get("target_body_parts", ["Full Body"]),
            "medical_conditions": ai_prescription.get("medical_conditions", []),
            "physical_limitation": ", ".join(ai_prescription.get("physical_limitations", [])) if ai_prescription.get("physical_limitations") else "None",
            "specific_avoidance": ", ".join(ai_prescription.get("activity_restrictions", [])) if ai_prescription.get("activity_restrictions") else "None",
            "blood_pressure": "N/A",
            "flags": self._create_flags_from_ai(ai_prescription),
        }

    def _create_clinical_context_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, Any]:
        """Create clinical context from AI prescription."""
        return {
            "conditions": ai_prescription.get("medical_conditions", []),
            "limitations": ai_prescription.get("physical_limitations", []),
            "restrictions": ai_prescription.get("activity_restrictions", []),
            "flags": self._create_flags_from_ai(ai_prescription),
        }

    def _create_flags_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, bool]:
        """Create clinical flags from AI prescription."""
        flags = {}
        limitations = ai_prescription.get("physical_limitations", [])
        restrictions = ai_prescription.get("activity_restrictions", [])
        
        limitations_text = " ".join(limitations).lower()
        restrictions_text = " ".join(restrictions).lower()
        
        # Knee safety
        if any(term in limitations_text for term in ["knee", "acl", "meniscus"]):
            flags["knee_sensitive"] = True
        elif "avoid jumping" in restrictions_text:
            flags["knee_sensitive"] = True
        
        # Floor work restrictions
        if "avoid floor work" in restrictions_text:
            flags["avoid_floor_work"] = True
        elif any(term in limitations_text for term in ["difficulty getting up", "floor contraindicated"]):
            flags["avoid_floor_work"] = True
        
        # High impact restrictions
        if any(term in restrictions_text for term in ["avoid high impact", "high impact restricted"]):
            flags["high_impact_restricted"] = True
        
        # Spinal loading restrictions
        if "avoid spinal loading" in restrictions_text:
            flags["avoid_spinal_loading"] = True
        
        # Overhead restrictions
        if "avoid overhead" in restrictions_text:
            flags["overhead_restricted"] = True
        
        return flags

    def _parse_clinical_context(self, notes_text: str) -> Dict[str, Any]:
        """
        Parse expert notes to extract clinical context.
        
        Returns:
            Dictionary with extracted medical conditions, limitations, restrictions, prescribed exercises, and flags
        """
        conditions = _extract_medical_conditions(notes_text)
        limitations = _extract_physical_limitations(notes_text)
        restrictions = _extract_activity_restrictions(notes_text)
        prescribed = _extract_prescribed_exercises(notes_text)
        flags = _create_clinical_flags(notes_text, limitations, restrictions)
        
        logger.info(f"[ExpertsNoteService] Parsed Clinical Context:")
        logger.info(f"  Conditions: {conditions}")
        logger.info(f"  Limitations: {limitations}")
        logger.info(f"  Restrictions: {restrictions}")
        logger.info(f"  Prescribed exercises: {len(prescribed)}")
        logger.info(f"  Flags: {flags}")
        
        return {
            "medical_conditions": conditions,
            "physical_limitations": limitations,
            "restrictions": restrictions,
            "prescribed_exercises": prescribed,
            "flags": flags,
            "raw_context": {
                "conditions": conditions,
                "limitations": limitations,
                "restrictions": restrictions,
            }
        }

    def _build_local_prescription(self, notes_text: str) -> Dict[str, Any]:
        """Create a safe structured prescription without calling Azure AI."""
        clinical = self._parse_clinical_context(notes_text)
        schedule = self._extract_weekly_schedule_from_text(notes_text)
        modalities = self._infer_modalities_from_text(notes_text)
        equipment = self._infer_equipment_from_text(notes_text)
        prescribed = clinical.get("prescribed_exercises", [])

        schedule = self._expand_weekly_schedule_from_frequency(notes_text, schedule, modalities)

        active_days = [
            day for day, activity in schedule.items()
            if not self._is_rest_day_text(activity)
        ]

        return {
            "days_per_week": len(active_days) or 3,
            "session_duration": self._extract_duration_label(notes_text),
            "goal": self._infer_goal_from_text(notes_text),
            "focus": modalities or ["mobility", "strength"],
            "medical_conditions": clinical.get("medical_conditions", []),
            "physical_limitations": clinical.get("physical_limitations", []),
            "activity_restrictions": clinical.get("restrictions", []),
            "intensity": self._infer_intensity_from_text(notes_text),
            "prescribed_exercises": prescribed,
            "avoid_exercises": [],
            "equipment_required": equipment,
            "target_body_parts": self._infer_target_body_parts(notes_text),
            "weekly_schedule": schedule,
            "session_directives": [],
            "session_types": [],
            "day_type_rules": [
                {"type": "hiit", "mapped_to": "full_body_cardio"},
                {"type": "interval", "mapped_to": "full_body_cardio"},
                {"type": "circuit", "mapped_to": "full_body_cardio"},
            ],
            "cardio_requirements": {
                "frequency": "2-3x per week",
                "duration": self._extract_duration_label(notes_text),
                "activities": ["Walking"] if "cardio" not in modalities else ["Walking", "Treadmill"],
            },
            "resistance_training": {
                "frequency": "2-3x per week",
                "equipment": ", ".join(equipment),
                "sets": 2,
                "reps": "10-15",
                "exercises": [item.get("name", "") for item in prescribed if isinstance(item, dict)],
            },
            "hiit_training": {
                "frequency": "1-2x per week",
                "duration": "20-30 mins",
                "structure": "5-6 exercises at controlled intervals",
            },
            "additional_requirements": {},
            "_source": "local_fallback",
        }

    def _ensure_usable_weekly_schedule(self, ai_prescription: Dict[str, Any], notes_text: str) -> Dict[str, Any]:
        """Expand sparse AI schedules when notes provide frequency rules instead of a sample week."""
        prescription = dict(ai_prescription or {})
        raw_schedule = prescription.get("weekly_schedule", {})
        schedule = raw_schedule if isinstance(raw_schedule, dict) else {}
        active_days = [
            day for day, activity in schedule.items()
            if activity and not self._is_rest_day_text(activity)
        ]
        has_frequency_rules = self._notes_have_frequency_rules(notes_text)

        if len(active_days) < 3 and has_frequency_rules:
            modalities = prescription.get("focus") if isinstance(prescription.get("focus"), list) else []
            modalities = modalities or self._infer_modalities_from_text(notes_text)
            schedule = self._expand_weekly_schedule_from_frequency(notes_text, schedule, modalities)
            prescription["weekly_schedule"] = self._enforce_week_balance(schedule)
            prescription["days_per_week"] = len([
                day for day, activity in prescription["weekly_schedule"].items()
                if activity and not self._is_rest_day_text(activity)
            ])
        elif schedule:
            prescription["weekly_schedule"] = self._enforce_week_balance(self._normalize_weekly_schedule(schedule))
        else:
            modalities = self._infer_modalities_from_text(notes_text)
            prescription["weekly_schedule"] = self._enforce_week_balance(
                self._expand_weekly_schedule_from_frequency(notes_text, {}, modalities)
            )
            prescription["days_per_week"] = len([
                day for day, activity in prescription["weekly_schedule"].items()
                if activity and not self._is_rest_day_text(activity)
            ])

        return prescription

    def _notes_have_frequency_rules(self, notes_text: str) -> bool:
        text = str(notes_text or "").lower()
        return bool(re.search(r"\b\d+\s*(?:-\s*\d+)?\s*x?\s*per\s+week\b", text)) or "daily" in text

    def _normalize_weekly_schedule(self, schedule: Dict[str, Any]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        day_aliases = {day.lower(): day.lower() for day in DAY_ORDER}
        
        # First, add all days from the input schedule
        for raw_day, activity in schedule.items():
            day = str(raw_day or "").strip().lower()
            if day in day_aliases:
                normalized[day] = str(activity or "").strip() or "Rest"
        
        # Ensure all 7 days are present, fill missing days with "Rest"
        for day in [d.lower() for d in DAY_ORDER]:
            if day not in normalized:
                normalized[day] = "Rest"
        
        return normalized

    def _enforce_week_balance(self, weekly_schedule: Dict[str, Any]) -> Dict[str, str]:
        normalized = self._normalize_weekly_schedule(weekly_schedule or {})
        active_days = [
            day for day, activity in normalized.items()
            if activity and not self._is_rest_day_text(activity)
        ]
        if not active_days:
            return {
                "monday": "Strength",
                "tuesday": "Hybrid",
                "wednesday": "Strength",
                "thursday": "Hybrid",
                "friday": "Strength",
                "saturday": "Hybrid",
                "sunday": "Rest",
            }

        strength_days = [
            day for day in active_days
            if any(term in str(normalized.get(day, "")).lower() for term in ["strength", "resistance", "weights", "hybrid"])
        ]
        cardio_days = [
            day for day in active_days
            if any(term in str(normalized.get(day, "")).lower() for term in ["cardio", "hiit", "interval", "hybrid"])
        ]

        if not strength_days:
            for day in active_days[: min(3, len(active_days))]:
                if "hybrid" not in str(normalized.get(day, "")).lower():
                    normalized[day] = "Strength" if day in {"monday", "wednesday", "friday"} else "Hybrid"

        if not cardio_days:
            for day in active_days[: min(3, len(active_days))]:
                current = str(normalized.get(day, "")).strip()
                normalized[day] = "Hybrid" if "strength" in current.lower() else "Cardio"

        return normalized

    def _expand_weekly_schedule_from_frequency(
        self,
        notes_text: str,
        current_schedule: Dict[str, Any] | None,
        modalities: List[str],
    ) -> Dict[str, str]:
        text = str(notes_text or "")
        schedule = {day.lower(): [] for day in DAY_ORDER}

        for day, activity in self._normalize_weekly_schedule(current_schedule or {}).items():
            if activity and not self._is_rest_day_text(activity):
                schedule.setdefault(day, []).extend(self._split_activity_label(activity))

        yoga_days = self._extract_named_class_days(text, ["yoga", "hatha"])
        for day in yoga_days:
            schedule[day].append("Yoga")

        cardio_count = self._extract_modality_frequency(text, ["cardio", "cardiovascular", "dance", "brisk walking", "hiking", "bike", "elliptical"], default=0)
        mobility_count = self._extract_modality_frequency(text, ["stretching", "physioball", "abs"], default=0)
        resistance_count = self._extract_modality_frequency(text, ["resistance", "weights", "bands", "body weight", "strength"], default=0)

        if not any([cardio_count, mobility_count, resistance_count, yoga_days]):
            fallback_labels = [
                "Strength",
                "Hybrid",
                "Strength",
                "Hybrid",
                "Strength",
                "Hybrid",
                "Rest",
            ]
            for day, label in zip([item.lower() for item in DAY_ORDER], fallback_labels):
                if label != "Rest":
                    schedule[day].append(label)
        else:
            self._assign_activity_to_days(schedule, "Cardio", cardio_count, ["monday", "wednesday", "friday", "saturday", "tuesday", "thursday"])
            self._assign_activity_to_days(schedule, "Resistance Training", resistance_count, ["monday", "wednesday", "saturday", "friday", "tuesday", "thursday"])
            self._assign_activity_to_days(schedule, "Mobility/Core", mobility_count, ["thursday", "saturday", "tuesday", "friday", "wednesday"])

        normalized: Dict[str, str] = {}
        for day in [item.lower() for item in DAY_ORDER]:
            activities = list(dict.fromkeys(item for item in schedule.get(day, []) if item and not self._is_rest_day_text(item)))
            normalized[day] = " and ".join(activities) if activities else "Rest"

        active_count = sum(1 for activity in normalized.values() if not self._is_rest_day_text(activity))
        if active_count == 0:
            fallback = {
                "monday": "Strength",
                "tuesday": "Hybrid",
                "wednesday": "Strength",
                "thursday": "Hybrid",
                "friday": "Strength",
                "saturday": "Hybrid",
                "sunday": "Rest",
            }
            normalized.update(fallback)
        return self._enforce_week_balance(normalized)

    def _split_activity_label(self, activity: Any) -> List[str]:
        return [
            part.strip()
            for part in re.split(r"\s*(?:/|,|\band\b|\+)\s*", str(activity or ""), flags=re.IGNORECASE)
            if part.strip()
        ]

    def _extract_modality_frequency(self, notes_text: str, keywords: List[str], default: int = 0) -> int:
        text = re.sub(r"\s+", " ", str(notes_text or ""))
        sentences = re.split(r"[\n.;]+", text)
        best = default
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if not any(keyword in sentence_lower for keyword in keywords):
                continue
            match = re.search(r"(\d+)\s*(?:-\s*(\d+))?\s*x?\s*per\s+week", sentence_lower)
            if match:
                low = int(match.group(1))
                high = int(match.group(2) or low)
                best = max(best, high)
        return min(best, 6)

    def _extract_named_class_days(self, notes_text: str, keywords: List[str]) -> List[str]:
        days: List[str] = []
        day_names = [day.lower() for day in DAY_ORDER]
        for sentence in re.split(r"[\n.;]+", str(notes_text or "")):
            sentence_lower = sentence.lower()
            if not any(keyword in sentence_lower for keyword in keywords):
                continue
            for day in day_names:
                if re.search(rf"\b{day}\b", sentence_lower) and day not in days:
                    days.append(day)
        return days

    def _assign_activity_to_days(
        self,
        schedule: Dict[str, List[str]],
        activity: str,
        count: int,
        preferred_days: List[str],
    ) -> None:
        if count <= 0:
            return
        assigned = 0
        for day in preferred_days:
            if assigned >= count:
                break
            day_activities = schedule.setdefault(day, [])
            if activity in day_activities:
                assigned += 1
                continue
            day_activities.append(activity)
            assigned += 1

    def _extract_weekly_schedule_from_text(self, notes_text: str) -> Dict[str, str]:
        schedule: Dict[str, str] = {}
        day_pattern = re.compile(
            r"^\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b\s*[:\-]?\s*(.*)",
            re.IGNORECASE,
        )
        for raw_line in str(notes_text or "").splitlines():
            line = raw_line.strip()
            match = day_pattern.search(line)
            if not match:
                continue
            day = match.group(1).lower()
            activity = match.group(2).strip() or self._activity_label_from_modalities(self._infer_modalities_from_text(notes_text))
            schedule[day] = activity
        return schedule

    def _infer_modalities_from_text(self, notes_text: str) -> List[str]:
        text = str(notes_text or "").lower()
        modalities: List[str] = []
        checks = [
            ("full_body_cardio", ["hiit", "interval", "tabata", "circuit"]),
            ("cardio", ["cardio", "walk", "treadmill", "bike", "elliptical"]),
            ("resistance", ["strength", "resistance", "weights", "dumbbell", "band", "push", "squat", "lunge"]),
            ("yoga", ["yoga", "child pose", "cobra", "downward"]),
            ("mobility", ["mobility", "stretch", "warm-up", "warm up"]),
        ]
        for modality, keywords in checks:
            if any(keyword in text for keyword in keywords):
                modalities.append(modality)
        return modalities

    def _activity_label_from_modalities(self, modalities: List[str]) -> str:
        if "full_body_cardio" in modalities:
            return "HIIT/Interval"
        if "resistance" in modalities and "cardio" in modalities:
            return "Cardio and Resistance Training"
        if "resistance" in modalities:
            return "Resistance Training"
        if "cardio" in modalities:
            return "Cardio"
        if "yoga" in modalities:
            return "Yoga and Mobility"
        return "Mobility and Strength"

    def _infer_equipment_from_text(self, notes_text: str) -> List[str]:
        text = str(notes_text or "").lower()
        equipment: List[str] = []
        equipment_checks = [
            ("dumbbell", ["dumbbell", "db ", " lbs", " lb "]),
            ("resistance band", ["resistance band", "banded", " band "]),
            ("treadmill", ["treadmill"]),
            ("cycling", ["bike", "cycle", "cycling"]),
            ("elliptical", ["elliptical"]),
        ]
        for label, keywords in equipment_checks:
            if any(keyword in text for keyword in keywords):
                equipment.append(label)
        return equipment or ["bodyweight"]

    def _extract_duration_label(self, notes_text: str) -> str:
        match = re.search(r"(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*(?:min|mins|minutes)\b", str(notes_text or ""), re.IGNORECASE)
        if not match:
            return "30-45 minutes"
        if match.group(2):
            return f"{match.group(1)}-{match.group(2)} minutes"
        return f"{match.group(1)} minutes"

    def _infer_goal_from_text(self, notes_text: str) -> str:
        text = str(notes_text or "").lower()
        if any(term in text for term in ["weight loss", "fat loss", "calorie"]):
            return "Weight Loss"
        if any(term in text for term in ["rehab", "pain", "injury", "therapy"]):
            return "Rehabilitation"
        if any(term in text for term in ["strength", "resistance"]):
            return "Strength"
        return "General Fitness"

    def _infer_intensity_from_text(self, notes_text: str) -> str:
        text = str(notes_text or "").lower()
        if any(term in text for term in ["vigorous", "hard", "high intensity", "hiit"]):
            return "high"
        if any(term in text for term in ["gentle", "easy", "light", "mobility", "stretch"]):
            return "low"
        return "moderate"

    def _infer_target_body_parts(self, notes_text: str) -> List[str]:
        text = str(notes_text or "").lower()
        targets: List[str] = []
        if any(term in text for term in ["upper", "shoulder", "arm", "chest", "back"]):
            targets.append("Upper")
        if any(term in text for term in ["lower", "hip", "knee", "leg", "glute", "ankle"]):
            targets.append("Lower")
        if any(term in text for term in ["core", "spine", "ab"]):
            targets.append("Core")
        return targets or ["Full Body"]

    def generate_plan_from_notes(self, notes_text: str) -> Dict[str, Any]:
        """
        Generate a complete workout plan from expert notes using AI parsing and dataset-driven generation.

        Steps:
        1. Parse notes using Azure AI to extract structured prescription
        2. Load and filter dataset based on AI-derived parameters
        3. Apply medical exclusions and restrictions
        4. Generate plan following the exact weekly schedule from AI
        5. Inject prescribed exercises and follow Plan of Action specifications

        Args:
            notes_text: Raw text from expert/doctor notes
            
        Returns:
            Dictionary with 'plan', 'profile', 'ai_prescription', and 'clinical_context'
        """
        if not notes_text or not notes_text.strip():
            raise ValueError("No notes provided")

        # STEP 1: Parse clinical context using Azure AI. If Azure content
        # filtering blocks benign rehab wording, fall back locally.
        used_local_fallback = False
        try:
            ai_prescription = self.ai_parser.parse_notes(notes_text)
        except AzureAIContentFilterError:
            logger.warning("[ExpertsNoteService] Falling back to local note parser after Azure content filter block.")
            ai_prescription = self._build_local_prescription(notes_text)
            used_local_fallback = True
        ai_prescription = self._ensure_usable_weekly_schedule(ai_prescription, notes_text)

        # STEP 2: Load dataset and apply AI-derived filters
        df = FitnessDataset.load(str(DATASET_DIR / "fitness.csv"))
        
        # Apply basic profile filters
        profile = self._create_profile_from_ai(ai_prescription)
        df = ExerciseFilter.apply_filters(df, profile)
        
        # Apply medical exclusions
        clinical_context = self._create_clinical_context_from_ai(ai_prescription)
        df = _hard_medical_exclusion(df, clinical_context)
        
        # Remove explicitly avoided exercises
        df = self._exclude_avoided_exercises(df, ai_prescription.get("avoid_exercises", []))

        # STEP 3: Generate plan following the exact weekly schedule
        plan = self._generate_plan_from_weekly_schedule(df, ai_prescription, clinical_context, profile)

        logger.info(f"[ExpertsNoteService] Generated AI-driven plan with {len(plan)} days")
        
        return {
            "plan": plan,
            "profile": profile,
            "notes_text": notes_text,
            "ai_prescription": ai_prescription,
            "clinical_context": clinical_context,
            "used_local_fallback": used_local_fallback,
        }

    def extract_high_level_goals(self, notes_text: str) -> Dict[str, Any]:
        """
        Extract high-level program structure from expert notes.
        Focus: Numbers, days, and general goals.
        Avoids triggering body-related filters.
        
        Args:
            notes_text: Raw text from expert/doctor notes
            
        Returns:
            Dictionary with days_per_week, goal, weekly_schedule, and equipment_required
        """
        if not notes_text or not notes_text.strip():
            raise ValueError("No notes provided")
        
        try:
            result = self.ai_parser.extract_high_level_goals(notes_text)
            logger.info("[ExpertsNoteService] Successfully extracted high-level goals")
            return result
        except Exception as e:
            logger.error(f"[ExpertsNoteService] Error extracting high-level goals: {str(e)}")
            raise

    def extract_clinical_safety(self, notes_text: str) -> Dict[str, Any]:
        """
        Identify medical flags and exclusions from expert notes.
        Focus: Medical terms and "Avoid" instructions.
        
        Args:
            notes_text: Raw text from expert/doctor notes
            
        Returns:
            Dictionary with medical_conditions, physical_limitations, activity_restrictions, and avoid_exercises
        """
        if not notes_text or not notes_text.strip():
            raise ValueError("No notes provided")
        
        try:
            result = self.ai_parser.extract_clinical_safety(notes_text)
            logger.info("[ExpertsNoteService] Successfully extracted clinical safety information")
            return result
        except Exception as e:
            logger.error(f"[ExpertsNoteService] Error extracting clinical safety: {str(e)}")
            raise

    def extract_exercise_repertoire(self, notes_text: str) -> Dict[str, Any]:
        """
        Extract specific exercises from expert notes using clinical names only.
        Focus: Formal exercise names without vivid descriptions to avoid filter triggers.
        
        Args:
            notes_text: Raw text from expert/doctor notes
            
        Returns:
            Dictionary with prescribed_exercises list (name, sets, reps, type)
        """
        if not notes_text or not notes_text.strip():
            raise ValueError("No notes provided")
        
        try:
            result = self.ai_parser.extract_exercise_repertoire(notes_text)
            logger.info("[ExpertsNoteService] Successfully extracted exercise repertoire")
            return result
        except Exception as e:
            logger.error(f"[ExpertsNoteService] Error extracting exercise repertoire: {str(e)}")
            raise

    def _verify_and_filter_plan(self, plan: Dict[str, dict], clinical_context: Dict[str, Any]) -> Dict[str, dict]:
        """
        Verify plan respects all safety constraints from clinical context.
        Remove any exercises that violate the constraints.
        """
        flags = clinical_context.get("flags", {})
        
        for day_name, day_plan in plan.items():
            for section in ["warmup", "main_workout", "cooldown"]:
                exercises = day_plan.get(section, [])
                safe_exercises = []
                
                for exercise in exercises:
                    if self._exercise_is_safe(exercise, flags):
                        safe_exercises.append(exercise)
                    else:
                        ex_name = exercise.get("name", exercise.get("exercise_name", "Unknown"))
                        logger.warning(f"[ExpertsNoteService] Removed unsafe exercise: {ex_name}")
                
                day_plan[section] = safe_exercises
        
        return plan

    @staticmethod
    def _exercise_is_safe(exercise: Dict[str, Any], flags: Dict[str, bool]) -> bool:
        """
        Check if an exercise is safe given clinical flags.
        """
        ex_text = " ".join([
            str(exercise.get("name", "")).lower(),
            str(exercise.get("exercise_name", "")).lower(),
            str(exercise.get("description", "")).lower(),
        ])
        
        # Check knee sensitivity
        if flags.get("knee_sensitive", False):
            if any(term in ex_text for term in ["jump", "squat", "lunge", "plyometric", "burpee"]):
                return False
        
        # Check floor work restrictions
        if flags.get("avoid_floor_work", False):
            if any(term in ex_text for term in ["floor", "prone", "supine", "plank", "push-up", "crunch"]):
                return False
        
        # Check high impact restrictions
        if flags.get("high_impact_restricted", False):
            if any(term in ex_text for term in ["jump", "run", "burpee", "box jump", "high impact"]):
                return False
        
        # Check spinal loading restrictions
        if flags.get("avoid_spinal_loading", False):
            if any(term in ex_text for term in ["deadlift", "squat", "overhead press", "heavy"]):
                return False
        
        # Check overhead restrictions
        if flags.get("overhead_restricted", False):
            if any(term in ex_text for term in ["overhead", "press", "raise"]):
                return False
        
        return True

    def _create_profile_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a fitness profile from AI prescription data.
        """
        profile = {
            "goal": ai_prescription.get("goal", "General Fitness"),
            "primary_goal": ai_prescription.get("goal", "General Fitness"),
            "fitness_level": "intermediate",  # Default
            "days_per_week": ai_prescription.get("days_per_week", 3),
            "session_duration": ai_prescription.get("session_duration", 45),
            "focus": ai_prescription.get("focus", []),
            "intensity": ai_prescription.get("intensity", "moderate"),
            "target_body_parts": ai_prescription.get("target_body_parts", []),
            "available_equipment": self._display_equipment_list(self._extract_available_equipment(ai_prescription)),
            "medical_conditions": ai_prescription.get("medical_conditions", []),
            "physical_limitation": ", ".join(ai_prescription.get("physical_limitations", [])) if ai_prescription.get("physical_limitations") else "None",
        }
        return profile

    def _create_clinical_context_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create clinical context from AI prescription data.
        """
        clinical_context = {
            "medical_conditions": ai_prescription.get("medical_conditions", []),
            "physical_limitations": ai_prescription.get("physical_limitations", []),
            "activity_restrictions": ai_prescription.get("activity_restrictions", []),
            "flags": self._create_clinical_flags_from_ai(ai_prescription),
        }
        return clinical_context

    def _create_clinical_flags_from_ai(self, ai_prescription: Dict[str, Any]) -> Dict[str, bool]:
        """
        Create clinical flags from AI prescription.
        """
        flags = {}
        limitations = ai_prescription.get("physical_limitations", [])
        restrictions = ai_prescription.get("activity_restrictions", [])
        
        limitations_text = " ".join(limitations).lower()
        restrictions_text = " ".join(restrictions).lower()
        
        # Knee-related flags
        if any(term in limitations_text for term in ["knee", "acl", "meniscus", "patella"]):
            flags["knee_sensitive"] = True
        elif "avoid jumping" in restrictions_text:
            flags["knee_sensitive"] = True
        
        # Floor work restrictions
        if "avoid floor work" in restrictions_text:
            flags["avoid_floor_work"] = True
        
        # High impact restrictions
        if "avoid high impact" in restrictions_text:
            flags["high_impact_restricted"] = True
        
        # Spinal loading restrictions
        if "avoid spinal loading" in restrictions_text:
            flags["avoid_spinal_loading"] = True
        
        # Overhead restrictions
        if "avoid overhead" in restrictions_text:
            flags["overhead_restricted"] = True
        
        return flags

    def _exclude_avoided_exercises(self, df: pd.DataFrame, avoid_exercises: List[str]) -> pd.DataFrame:
        """
        Remove exercises that are explicitly avoided.
        """
        if not avoid_exercises:
            return df
        
        # Filter out exercises whose names contain avoided terms
        avoid_lower = [term.lower() for term in avoid_exercises]
        
        def should_exclude(row):
            exercise_name = str(row.get("name", "")).lower()
            exercise_desc = str(row.get("description", "")).lower()
            exercise_text = exercise_name + " " + exercise_desc
            
            return not any(avoid_term in exercise_text for avoid_term in avoid_lower)
        
        return df[df.apply(should_exclude, axis=1)]

    def _generate_plan_from_weekly_schedule(
        self,
        df: pd.DataFrame,
        ai_prescription: Dict[str, Any],
        clinical_context: Dict[str, Any],
        profile: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate workout plan following the exact weekly schedule from AI prescription.
        """
        weekly_schedule = ai_prescription.get("weekly_schedule", {})
        plan = {}
        weekly_used_families: Set[str] = set()
        weekly_used_exercises: Set[str] = set()
        weekly_used_equipment: Set[str] = set()
        used_cardio_modes: Set[str] = set()
        profile = profile or self._create_profile_from_ai(ai_prescription)
        user_seed = self._get_user_seed(profile)
        required_strength_days = self._extract_required_strength_days(ai_prescription)
        weekly_strength_days_generated = 0
        non_rest_days_total = sum(
            1
            for activity in weekly_schedule.values()
            if not re.search(r"\brest\b", str(activity or "").lower())
        )
        non_rest_days_seen = 0
        
        for day_index, (day_name, activities_text) in enumerate(weekly_schedule.items()):
            random.seed(user_seed + day_index)
            normalized_day_name = str(day_name).capitalize()
            if re.search(r"\brest\b", str(activities_text or "").lower()):
                plan[normalized_day_name] = self._build_structured_day_plan(
                    day_name=normalized_day_name,
                    day_plan={"warmup": [], "main_workout": [], "cooldown": []},
                    activities_text=activities_text,
                    activities=[],
                    ai_prescription=ai_prescription,
                    is_rest_day=True,
                )
                continue
            non_rest_days_seen += 1
            
            # Parse activities from the text
            activities = self._parse_day_activities(activities_text, ai_prescription)
            remaining_active_days = max(0, non_rest_days_total - non_rest_days_seen)
            activities, added_strength = self._enforce_frequency(
                activities=activities,
                required_strength_days=required_strength_days,
                weekly_strength_days_generated=weekly_strength_days_generated,
                remaining_active_days=remaining_active_days,
            )
            activities = self._apply_session_directives(normalized_day_name, activities, ai_prescription)
            activities = self._apply_activity_priority(activities)
            if added_strength or any(str(item.get("type", "")).lower() == "resistance" for item in activities):
                weekly_strength_days_generated += 1
            
            day_plan = {"warmup": [], "main_workout": [], "cooldown": []}
            desired_main_count = min(6, max(5, sum(int(activity.get("target_count", 0) or 0) for activity in activities) or 5))
            
            # Generate main workout exercises
            for activity in activities:
                exercises = self._generate_activity_exercises(
                    df,
                    activity,
                    ai_prescription,
                    clinical_context,
                    profile=profile,
                    day_index=day_index,
                    excluded_families=weekly_used_families,
                    weekly_used_exercises=weekly_used_exercises,
                    weekly_used_equipment=weekly_used_equipment,
                    used_cardio_modes=used_cardio_modes,
                )
                day_plan["main_workout"].extend(exercises)
            
            day_plan["main_workout"] = self._dedupe_exercise_variations(day_plan["main_workout"])
            day_plan["main_workout"] = self._normalize_intensity(day_plan["main_workout"])

            # Generate workout-specific warmup exercises with upper/lower variety.
            day_plan["warmup"] = self._generate_warmup_exercises(
                df,
                activities_text,
                clinical_context,
                day_index=day_index,
                main_workout=day_plan["main_workout"],
            )
            
            # Generate workout-specific cooldown exercises with upper/lower variety.
            day_plan["cooldown"] = self._generate_cooldown_exercises(
                df,
                activities_text,
                clinical_context,
                day_index=day_index,
                main_workout=day_plan["main_workout"],
            )
            day_plan["main_workout"] = self._ensure_uniform_main_workout(
                df=df,
                day_plan=day_plan,
                activities_text=activities_text,
                ai_prescription=ai_prescription,
                clinical_context=clinical_context,
                target_count=desired_main_count,
                day_index=day_index,
                excluded_families=weekly_used_families,
                weekly_used_exercises=weekly_used_exercises,
                profile=profile,
            )
            day_plan["main_workout"] = self._validate_day_plan(day_plan["main_workout"], activities)
            day_plan["main_workout"] = self._dedupe_exercise_variations(day_plan["main_workout"])
            day_plan["main_workout"] = self._normalize_intensity(day_plan["main_workout"])
            weekly_used_families.update(
                self._exercise_family(ex.get("name", ""))
                for ex in day_plan["main_workout"]
                if not ex.get("_is_session_payload")
            )
            weekly_used_exercises.update(
                self._exercise_key(ex.get("name", ""))
                for ex in day_plan["main_workout"]
                if not ex.get("_is_session_payload")
            )
            weekly_used_equipment.update(
                self._normalize_equipment_label(ex.get("equipment", ""))
                for ex in day_plan["main_workout"]
                if self._normalize_equipment_label(ex.get("equipment", ""))
            )
            used_cardio_modes.update(
                str(ex.get("equipment", "")).strip().lower()
                for ex in day_plan["main_workout"]
                if ex.get("_session_type") in {"cardio", "full_body_cardio"} and str(ex.get("equipment", "")).strip()
            )
            
            plan[normalized_day_name] = self._build_structured_day_plan(
                day_name=normalized_day_name,
                day_plan=day_plan,
                activities_text=activities_text,
                activities=activities,
                ai_prescription=ai_prescription,
            )
        
        return plan

    def _apply_activity_priority(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority = ["resistance", "cardio", "full_body_cardio", "yoga", "mobility"]
        priority_index = {name: idx for idx, name in enumerate(priority)}
        unique_activities: List[Dict[str, Any]] = []
        seen_types: Set[str] = set()

        for activity in activities or []:
            activity_type = str(activity.get("type", "")).strip().lower()
            if not activity_type or activity_type in seen_types:
                continue
            seen_types.add(activity_type)
            unique_activities.append(activity)

        sorted_activities = sorted(
            unique_activities,
            key=lambda item: priority_index.get(str(item.get("type", "")).strip().lower(), len(priority)),
        )
        return sorted_activities[:2]

    def _parse_day_activities(self, activities_text: str, ai_prescription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse activity descriptions from daily schedule text.
        """
        text_lower = str(activities_text or "").lower()
        if re.search(r"\brest\b", text_lower):
            return []
        has_yoga = "yoga" in text_lower
        has_mobility = any(term in text_lower for term in ["mobility", "stretch", "physioball"])
        has_cardio = any(term in text_lower for term in ["cardio", "walk", "walking", "cycling", "elliptical", "rowing", "hybrid"])
        has_hiit = any(term in text_lower for term in ["hiit", "interval", "tabata"])
        has_circuit_resistance = "circuit resistance" in text_lower
        has_strength = any(term in text_lower for term in ["weight", "weights", "resistance", "strength", "dumbbell", "band", "hybrid"]) or has_circuit_resistance

        activities: List[Dict[str, Any]] = []

        def _add(activity_type: str, **kwargs: Any) -> None:
            payload = {"type": activity_type}
            payload.update(kwargs)
            activities.append(payload)

        if has_hiit:
            _add("full_body_cardio", target_count=5)
        if has_strength:
            _add(
                "resistance",
                target_count=4 if (has_cardio or has_hiit) else 5,
                focus="full",
                paired_with="cardio" if (has_cardio or has_hiit) else "",
            )
        if has_cardio and not has_hiit:
            _add("cardio", target_count=1 if has_strength else 5)
        if has_yoga:
            _add("yoga", target_count=5 if not (has_cardio or has_strength or has_hiit) else 1)
        if has_mobility and not has_yoga:
            _add("mobility", target_count=5 if not (has_cardio or has_strength or has_hiit) else 1)

        if has_cardio and has_yoga and not has_strength:
            activities = [item for item in activities if str(item.get("type", "")).lower() != "resistance"]

        if has_circuit_resistance:
            activities = [item for item in activities if str(item.get("type", "")).lower() != "full_body_cardio"]

        if not activities:
            activities = [{"type": "cardio", "target_count": 5}]

        return self._apply_activity_priority(activities)

    def _generate_activity_exercises(
        self,
        df: pd.DataFrame,
        activity: Dict[str, Any],
        ai_prescription: Dict[str, Any],
        clinical_context: Dict[str, Any],
        profile: Dict[str, Any] | None = None,
        day_index: int = 0,
        excluded_families: Set[str] | None = None,
        weekly_used_exercises: Set[str] | None = None,
        weekly_used_equipment: Set[str] | None = None,
        used_cardio_modes: Set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate specific exercises for an activity type from dataset with complete information.
        """
        activity_type = activity.get("type")
        if activity_type == "pilates":
            return []
        exercises: List[Dict[str, Any]] = []
        flags = clinical_context.get("flags", {})
        target_count = int(activity.get("target_count", 6) or 6)
        equipment_preferences = self._extract_available_equipment(ai_prescription)
        profile = profile or self._create_profile_from_ai(ai_prescription)
        allowed_cardio_types = self._get_allowed_cardio_types(ai_prescription, profile)
        weekly_used_exercises = set(weekly_used_exercises or set())
        weekly_used_equipment = set(weekly_used_equipment or set())
        used_cardio_modes = set(used_cardio_modes or set())
        goal_multipliers = self._diversify_by_goal(ai_prescription.get("goal"))

        if activity_type == "resistance":
            target_count = max(3, min(target_count, 4)) if "cardio" in str(activity.get("paired_with", "")).lower() else target_count
            resistance_df = df[df["Primary Category"].str.lower().str.contains("strength|weight|resistance", na=False)].copy()
            resistance_df = resistance_df[~resistance_df["Tags"].str.contains("warm up|cooldown", na=False)]
            resistance_df = self._apply_strict_equipment_filter(resistance_df, profile)
            resistance_df = self._filter_rows_by_required_equipment(resistance_df, ai_prescription)
            resistance_df = self._balance_equipment(resistance_df)
            safe_df = resistance_df[resistance_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
            if safe_df.empty:
                safe_df = resistance_df

            focus = str(activity.get("focus") or "full").lower()
            safe_df = self._filter_rows_by_intensity_band(safe_df, ai_prescription)
            safe_df = self._apply_goal_weighting(safe_df, activity_type, goal_multipliers, day_index)
            safe_df = self._prioritize_weekly_equipment_variety(safe_df, weekly_used_equipment)
            safe_df = self._exclude_used_rows(safe_df, weekly_used_exercises, excluded_families)
            selected_rows = self._select_resistance_rows(
                safe_df,
                target_count,
                focus,
                equipment_preferences,
                day_index=day_index,
                excluded_families=excluded_families,
            )
            selected_rows = self._sample_rows(selected_rows, target_count)
            for _, row in selected_rows.iterrows():
                exercises.append(self._format_exercise_from_dataset(row, "resistance"))

        elif activity_type == "cardio":
            return [
                self._build_session_payload(
                    {"session_type": "cardio"},
                    ai_prescription,
                    profile=profile,
                    used_cardio_modes=used_cardio_modes,
                )
            ]

        elif activity_type == "full_body_cardio":
            target_count = max(4, min(target_count, 5))
            exercises.append(self._build_session_payload({"session_type": "full_body_cardio"}, ai_prescription, profile=profile, used_cardio_modes=used_cardio_modes))
            cardio_df = df[df["Primary Category"].str.lower().str.contains("cardio", na=False)].copy()
            cardio_df = cardio_df[~cardio_df["Primary Category"].str.contains("strength|resistance", case=False, na=False)]
            cardio_df = cardio_df[~cardio_df["Tags"].str.contains("warm up|cooldown", na=False)]
            cardio_df = self._apply_strict_equipment_filter(cardio_df, profile)
            cardio_df = self._filter_cardio_rows_by_allowed_types(cardio_df, allowed_cardio_types)
            safe_df = cardio_df[cardio_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
            if safe_df.empty:
                safe_df = cardio_df

            safe_df = self._filter_rows_by_intensity_band(safe_df, ai_prescription)
            safe_df = self._apply_goal_weighting(safe_df, activity_type, goal_multipliers, day_index)
            safe_df = self._exclude_used_rows(safe_df, weekly_used_exercises, excluded_families)
            selected_rows = self._pick_distinct_rows(
                self._rank_rows_by_equipment(safe_df, equipment_preferences),
                max(0, target_count - 1),
                excluded_families=excluded_families,
            )
            selected_rows = self._sample_rows(selected_rows, max(0, target_count - 1))
            for _, row in selected_rows.iterrows():
                exercises.append(self._format_exercise_from_dataset(row, "cardio"))

        elif activity_type == "yoga":
            return [
                self._build_session_payload(
                    {"session_type": "yoga"},
                    ai_prescription,
                    profile=profile,
                    used_cardio_modes=used_cardio_modes,
                )
            ]

        elif activity_type == "mobility":
            mobility_df = df[
                df["Primary Category"].str.lower().str.contains("mobility|stretch|flexibility", na=False)
            ].copy()
            mobility_df = mobility_df[~mobility_df["Tags"].str.contains("warm up|cooldown", na=False)]
            mobility_df = self._apply_strict_equipment_filter(mobility_df, profile)
            safe_df = mobility_df[mobility_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
            if safe_df.empty:
                safe_df = mobility_df
            safe_df = self._filter_rows_by_intensity_band(safe_df, ai_prescription)
            safe_df = self._apply_goal_weighting(safe_df, activity_type, goal_multipliers, day_index)
            safe_df = self._exclude_used_rows(safe_df, weekly_used_exercises, excluded_families)
            selected_rows = self._pick_distinct_rows(self._rank_rows_by_equipment(safe_df, equipment_preferences), target_count, excluded_families=excluded_families)
            selected_rows = self._sample_rows(selected_rows, target_count)
            for _, row in selected_rows.iterrows():
                exercises.append(self._format_exercise_from_dataset(row, "mobility"))

        return exercises
        """
        Check if an exercise is safe given clinical flags.
        """
        ex_text = " ".join([
            str(exercise.get("name", "")).lower(),
            str(exercise.get("exercise_name", "")).lower(),
            str(exercise.get("benefit", "")).lower(),
            str(exercise.get("body_region", "")).lower(),
        ])
        
        # Knee safety
        if flags.get("knee_sensitive"):
            unsafe_knee = ["jump", "hop", "plyo", "burpee", "box jump", "lunge", "squat", "box step"]
            if any(term in ex_text for term in unsafe_knee):
                return False
        
        # Floor work restrictions
        if flags.get("avoid_floor_work"):
            unsafe_floor = ["prone", "supine", "floor", "lying", "mat work", "plank", "push-up", "kneeling"]
            if any(term in ex_text for term in unsafe_floor):
                return False
        
        # High impact restrictions
        if flags.get("high_impact_restricted"):
            unsafe_impact = ["running", "sprinting", "jumping", "hopping", "high impact", "plyometric"]
            if any(term in ex_text for term in unsafe_impact):
                return False
        
        # Spinal loading restrictions
        if flags.get("avoid_spinal_loading"):
            unsafe_spine = ["deadlift", "squat", "heavy", "spinal", "back extension", "loaded"]
            if any(term in ex_text for term in unsafe_spine):
                return False
        
        # Overhead restrictions
        if flags.get("overhead_restricted"):
            unsafe_overhead = ["overhead", "press", "shoulder press", "snatch", "clean"]
            if any(term in ex_text for term in unsafe_overhead):
                return False
        
        return True

    def _generate_warmup_exercises(
        self,
        df: pd.DataFrame,
        activities_text: str,
        clinical_context: Dict[str, Any],
        day_index: int = 0,
        main_workout: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate 3 fixed warmup exercises: 1 cardio, 1 upper body mobility, 1 lower body mobility.
        All exercises must be tagged as "Warm up" in the dataset.
        """
        warmup_exercises: List[Dict[str, Any]] = []
        flags = clinical_context.get("flags", {})
        
        # Filter warmup exercises from dataset
        warmup_df = df[df["Tags"].str.contains("warm up", na=False)].copy()
        
        # Apply safety filters
        safe_warmup_df = warmup_df[warmup_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
        
        if safe_warmup_df.empty:
            safe_warmup_df = warmup_df  # Fallback if no safe exercises
        
        focus_regions = self._infer_main_body_regions(main_workout or [], activities_text)
        cardio_warmup = safe_warmup_df[safe_warmup_df["Primary Category"].str.lower().str.contains("cardio", na=False)]
        upper_mobility = safe_warmup_df[
            safe_warmup_df["Primary Category"].str.lower().str.contains("mobility|stretch|flexibility", na=False) &
            safe_warmup_df["Body Region"].str.lower().str.contains("upper|shoulder|arm|chest|back|neck", na=False)
        ]
        lower_mobility = safe_warmup_df[
            safe_warmup_df["Primary Category"].str.lower().str.contains("mobility|stretch|flexibility", na=False) &
            safe_warmup_df["Body Region"].str.lower().str.contains("lower|leg|hip|glute|calf", na=False)
        ]
        cardio_warmup = self._rotate_rows(cardio_warmup, day_index)
        upper_mobility = self._rotate_rows(upper_mobility, day_index)
        lower_mobility = self._rotate_rows(lower_mobility, day_index)

        seen_names: Set[str] = set()
        warmup_exercises.extend(self._build_support_exercises(cardio_warmup, "warmup", count=1, sets="1", reps="1-2 minutes", min_seconds=120, seen_names=seen_names))
        if "upper" in focus_regions:
            warmup_exercises.extend(self._build_support_exercises(upper_mobility, "warmup", count=1, sets="1", seen_names=seen_names))
        if "lower" in focus_regions:
            warmup_exercises.extend(self._build_support_exercises(lower_mobility, "warmup", count=1, sets="1", seen_names=seen_names))
        if not any("upper" in self._body_region_bucket(ex.get("body_region")) for ex in warmup_exercises):
            warmup_exercises.extend(self._build_support_exercises(upper_mobility, "warmup", count=1, sets="1", seen_names=seen_names))
        if not any("lower" in self._body_region_bucket(ex.get("body_region")) for ex in warmup_exercises):
            warmup_exercises.extend(self._build_support_exercises(lower_mobility, "warmup", count=1, sets="1", seen_names=seen_names))
        if len(warmup_exercises) < 3:
            remaining_needed = 3 - len(warmup_exercises)
            warmup_exercises.extend(self._build_support_exercises(self._rotate_rows(safe_warmup_df, day_index), "warmup", count=remaining_needed, sets="1", seen_names=seen_names))

        return self._dedupe_exercise_variations(warmup_exercises)[:3]

    def _generate_cooldown_exercises(
        self,
        df: pd.DataFrame,
        activities_text: str,
        clinical_context: Dict[str, Any],
        day_index: int = 0,
        main_workout: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate 2 fixed cooldown exercises: 1 upper body stretch and 1 lower body stretch.
        All exercises must be tagged as "Cooldown" in the dataset.
        """
        cooldown_exercises: List[Dict[str, Any]] = []
        flags = clinical_context.get("flags", {})
        
        # Filter cooldown exercises from dataset
        cooldown_df = df[df["Tags"].str.contains("cooldown", na=False)].copy()
        
        # Apply safety filters
        safe_cooldown_df = cooldown_df[cooldown_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
        
        if safe_cooldown_df.empty:
            safe_cooldown_df = cooldown_df  # Fallback if no safe exercises
        
        upper_stretch = safe_cooldown_df[
            safe_cooldown_df["Primary Category"].str.lower().str.contains("stretch|flexibility|mobility", na=False) &
            safe_cooldown_df["Body Region"].str.lower().str.contains("upper|shoulder|arm|chest|back|neck|core|full", na=False)
        ]
        lower_stretch = safe_cooldown_df[
            safe_cooldown_df["Primary Category"].str.lower().str.contains("stretch|flexibility|mobility", na=False) &
            safe_cooldown_df["Body Region"].str.lower().str.contains("lower|leg|hip|glute|calf|full", na=False)
        ]
        lower_specific = safe_cooldown_df[
            safe_cooldown_df["Primary Category"].str.lower().str.contains("stretch|flexibility|mobility", na=False) &
            safe_cooldown_df["Body Region"].str.lower().str.contains("lower|leg|hip|glute|calf", na=False)
        ]
        upper_stretch = self._rotate_rows(upper_stretch, day_index)
        lower_stretch = self._rotate_rows(lower_specific if not lower_specific.empty else lower_stretch, day_index)

        seen_names: Set[str] = set()
        cooldown_exercises.extend(self._build_support_exercises(upper_stretch, "cooldown", count=1, sets="1", seen_names=seen_names))
        cooldown_exercises.extend(self._build_support_exercises(lower_stretch, "cooldown", count=1, sets="1", seen_names=seen_names))
        if len(cooldown_exercises) < 2:
            remaining_needed = 2 - len(cooldown_exercises)
            cooldown_exercises.extend(self._build_support_exercises(self._rotate_rows(safe_cooldown_df, day_index), "cooldown", count=remaining_needed, sets="1", seen_names=seen_names))

        return self._dedupe_exercise_variations(cooldown_exercises)[:2]

    def _generate_day_title(self, activities: List[Dict[str, Any]]) -> str:
        names = [str(activity.get("type", "")).replace("_", " ").title() for activity in activities if activity.get("type")]
        if not names:
            return "Rest Day"
        if any(str(activity.get("type", "")).lower() == "full_body_cardio" for activity in activities):
            return "Full Body Hybrid"
        return " + ".join(names)

    def _generate_title_from_plan(self, main_workout: List[Dict[str, Any]]) -> str:
        types: Set[str] = set()
        for exercise in main_workout or []:
            if exercise.get("_is_session_payload"):
                session_type = str(exercise.get("_session_type") or exercise.get("session_type") or "").strip().lower()
                if session_type:
                    types.add(session_type)
            else:
                category = str(exercise.get("category", "")).strip().lower()
                if category:
                    types.add(category)

        if not types:
            return "Workout Session"

        # HIIT/Interval and cardio should collapse into a single title family.
        if "full_body_cardio" in types and "cardio" in types:
            types.discard("cardio")

        mapping = {
            "cardio": "Cardio",
            "full_body_cardio": "Full Body Hybrid",
            "resistance": "Strength",
            "strength": "Strength",
            "yoga": "Yoga",
            "mobility": "Mobility",
        }
        names = sorted({mapping.get(item, item.replace("_", " ").title()) for item in types if item})
        return " + ".join(names)

    def _validate_day_plan(
        self,
        main_workout: List[Dict[str, Any]],
        expected_activities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        expected_types = {
            str(activity.get("type", "")).strip().lower()
            for activity in expected_activities or []
            if str(activity.get("type", "")).strip()
        }
        if not main_workout or not expected_types:
            return list(main_workout or [])

        validated: List[Dict[str, Any]] = []
        for exercise in main_workout:
            actual_type = ""
            if exercise.get("_is_session_payload"):
                actual_type = str(exercise.get("_session_type") or exercise.get("session_type") or "").strip().lower()
            else:
                category = str(exercise.get("category", "")).strip().lower()
                primary_category = str(exercise.get("primary_category", "")).strip().lower()
                if any(term in f"{category} {primary_category}" for term in ["strength", "resistance", "weight"]):
                    actual_type = "resistance"
                elif "cardio" in category or "cardio" in primary_category:
                    actual_type = "cardio"
                elif "yoga" in category or "yoga" in primary_category:
                    actual_type = "yoga"
                elif any(term in f"{category} {primary_category}" for term in ["mobility", "stretch", "flexibility"]):
                    actual_type = "mobility"

            if actual_type in expected_types or (actual_type == "cardio" and "full_body_cardio" in expected_types):
                validated.append(exercise)

        if expected_types == {"cardio"}:
            cardio_sessions = [
                exercise for exercise in validated
                if exercise.get("_is_session_payload") and str(exercise.get("_session_type") or "").strip().lower() == "cardio"
            ]
            return cardio_sessions[:1] if cardio_sessions else []

        if expected_types == {"yoga"}:
            yoga_sessions = [
                exercise for exercise in validated
                if exercise.get("_is_session_payload") and str(exercise.get("_session_type") or "").strip().lower() == "yoga"
            ]
            return yoga_sessions[:1] if yoga_sessions else []

        if "cardio" in expected_types and "resistance" not in expected_types:
            validated = [
                exercise for exercise in validated
                if not (
                    exercise.get("_is_session_payload")
                    and str(exercise.get("_session_type") or "").strip().lower() == "resistance"
                )
            ]

        if expected_types == {"cardio", "yoga"}:
            filtered_sessions: List[Dict[str, Any]] = []
            seen_session_types: Set[str] = set()
            for exercise in validated:
                if not exercise.get("_is_session_payload"):
                    continue
                session_type = str(exercise.get("_session_type") or "").strip().lower()
                if session_type in {"cardio", "yoga"} and session_type not in seen_session_types:
                    filtered_sessions.append(exercise)
                    seen_session_types.add(session_type)
            return filtered_sessions

        return validated

    def _classify_day_focus(self, day_plan: Dict[str, Any], activities_text: str) -> Dict[str, str]:
        """Infer a user-friendly workout category/title from exercises and schedule text."""
        main_exercises = day_plan.get("main_workout", [])
        if not main_exercises and self._is_rest_day_text(activities_text):
            return {
                "main_workout_category": "Rest Day",
                "workout_title": "Rest Day",
            }

        activity_types: List[str] = []
        body_regions: Set[str] = set()

        for exercise in main_exercises:
            category = str(exercise.get("category", "")).lower()
            primary_category = str(exercise.get("primary_category", "")).lower()
            body_region = str(exercise.get("body_region", "")).lower()
            name = str(exercise.get("name", "")).lower()

            if body_region:
                if any(term in body_region for term in ["upper", "shoulder", "chest", "arm", "back"]):
                    body_regions.add("upper")
                if any(term in body_region for term in ["lower", "leg", "glute", "hip", "calf"]):
                    body_regions.add("lower")
                if "core" in body_region:
                    body_regions.add("core")

            if category == "hiit" or "hiit" in primary_category:
                activity_types.append("hiit")
            elif category == "cardio" or "cardio" in primary_category:
                activity_types.append("cardio")
            elif category in {"resistance", "strength"} or any(term in primary_category for term in ["strength", "weight", "resistance"]):
                activity_types.append("strength")

            if any(term in name for term in ["circuit", "combo"]):
                activity_types.append("circuit")

        normalized_activity_text = str(activities_text or "").lower()
        if "hiit" in normalized_activity_text or "interval" in normalized_activity_text:
            activity_types.append("full_body_cardio")
        if "cardio" in normalized_activity_text:
            activity_types.append("cardio")
        if any(term in normalized_activity_text for term in ["strength", "resistance", "weights"]):
            activity_types.append("strength")
        if "circuit" in normalized_activity_text:
            activity_types.append("full_body_cardio")
        if "yoga" in normalized_activity_text:
            activity_types.append("yoga")

        unique_activities = list(dict.fromkeys(activity_types))

        if "full_body_cardio" in unique_activities:
            category = "Full Body Cardio"
        elif "strength" in unique_activities and "cardio" in unique_activities:
            category = "Strength + Cardio"
        elif "yoga" in unique_activities and "cardio" in unique_activities:
            category = "Yoga + Cardio"
        elif "cardio" in unique_activities and "strength" not in unique_activities:
            category = "Cardio Day"
        elif "strength" in unique_activities:
            if "upper" in body_regions and "lower" in body_regions:
                category = "Full Body Strength"
            elif "upper" in body_regions:
                category = "Upper Body Strength"
            elif "lower" in body_regions:
                category = "Lower Body Strength"
            elif "core" in body_regions:
                category = "Core Strength"
            else:
                category = "Strength Day"
        elif "yoga" in unique_activities:
            category = "Yoga"
        elif normalized_activity_text.strip():
            category = normalized_activity_text.strip().title()
        else:
            category = "Workout Session"

        workout_title = category if category == "Rest Day" else f"{category} Workout"
        return {
            "main_workout_category": category,
            "workout_title": workout_title,
        }

    def _build_structured_day_plan(
        self,
        day_name: str,
        day_plan: Dict[str, Any],
        activities_text: str,
        activities: List[Dict[str, Any]],
        ai_prescription: Dict[str, Any],
        is_rest_day: bool = False,
    ) -> Dict[str, Any]:
        """Convert an internal day plan into the frontend response structure."""
        if not is_rest_day:
            day_plan["main_workout"] = self._validate_day_plan(day_plan.get("main_workout", []), activities)
        generated_title = self._generate_title_from_plan(day_plan.get("main_workout", [])) if not is_rest_day else "Rest Day"
        day_focus = {
            "main_workout_category": generated_title,
            "workout_title": generated_title,
        } if not is_rest_day else {
            "main_workout_category": "Rest Day",
            "workout_title": "Rest Day",
        }
        warmup = day_plan.get("warmup", []) if not is_rest_day else []
        main_workout = day_plan.get("main_workout", []) if not is_rest_day else []
        cooldown = day_plan.get("cooldown", []) if not is_rest_day else []

        all_exercises = warmup + main_workout + cooldown
        total_calories = sum(int(ex.get("estimated_calories", 0) or 0) for ex in all_exercises)
        total_duration_sec = sum(int(ex.get("est_time_sec", 0) or 0) for ex in all_exercises)

        difficulty_level = str(
            ai_prescription.get("intensity")
            or ai_prescription.get("fitness_level")
            or "Beginner"
        ).strip().title()

        return {
            "day_name": str(day_name).capitalize(),
            "workout_title": day_focus["workout_title"],
            "difficulty_level": difficulty_level,
            "warmup_duration": self._section_duration_label(warmup),
            "main_workout_category": day_focus["main_workout_category"],
            "cooldown_duration": self._section_duration_label(cooldown),
            "warmup": warmup,
            "main_workout": main_workout,
            "cooldown": cooldown,
            "safety_notes": ["Stay hydrated", "Monitor RPE"],
            "est_total_calories": total_calories,
            "est_total_duration_min": round(total_duration_sec / 60) if total_duration_sec else 0,
        }

    def _ensure_uniform_main_workout(
        self,
        df: pd.DataFrame,
        day_plan: Dict[str, Any],
        activities_text: str,
        ai_prescription: Dict[str, Any],
        clinical_context: Dict[str, Any],
        target_count: int = 6,
        day_index: int = 0,
        excluded_families: Set[str] | None = None,
        weekly_used_exercises: Set[str] | None = None,
        profile: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Ensure every active day has a stable, non-empty main workout size."""
        existing = self._dedupe_exercise_variations(list(day_plan.get("main_workout", []) or []))
        normalized_text = str(activities_text or "").lower()
        is_cardio_only_day = (
            any(term in normalized_text for term in ["cardio", "walk", "walking", "cycling", "elliptical", "rowing"])
            and not any(term in normalized_text for term in ["strength", "resistance", "weights", "dumbbell", "band", "yoga", "mobility", "stretch", "hiit", "interval", "tabata"])
        )
        is_yoga_only_day = "yoga" in normalized_text and not any(
            term in normalized_text for term in ["cardio", "strength", "resistance", "weights", "mobility", "stretch", "hiit", "interval", "tabata"]
        )
        if is_cardio_only_day or is_yoga_only_day:
            session_blocks = [ex for ex in existing if ex.get("_is_session_payload")]
            return session_blocks[:1] if session_blocks else existing[:1]
        if len(existing) >= target_count:
            return existing[:target_count]

        flags = clinical_context.get("flags", {})
        equipment_preferences = self._extract_available_equipment(ai_prescription)
        existing_names = {str(ex.get("name", "")).strip().lower() for ex in existing if str(ex.get("name", "")).strip()}
        session_blocks = [ex for ex in existing if ex.get("_is_session_payload")]
        existing_non_session_families = {
            self._exercise_family(ex.get("name", ""))
            for ex in existing
            if str(ex.get("name", "")).strip() and not ex.get("_is_session_payload")
        }
        excluded_families = set(excluded_families or set())
        weekly_used_exercises = set(weekly_used_exercises or set())

        if any(term in normalized_text for term in ["strength", "resistance", "weights", "dumbbell", "band"]):
            fallback_section = "resistance"
            fallback_rows = self._fallback_main_rows(df, flags, equipment_preferences, preferred_mode="resistance", focus_text=normalized_text)
        elif "yoga" in normalized_text:
            fallback_section = "yoga"
            fallback_rows = self._fallback_main_rows(df, flags, equipment_preferences, preferred_mode="yoga", focus_text=normalized_text)
        else:
            fallback_section = "cardio"
            fallback_rows = self._fallback_main_rows(df, flags, equipment_preferences, preferred_mode="cardio", focus_text=normalized_text)
        if profile is not None:
            fallback_rows = self._apply_strict_equipment_filter(fallback_rows, profile)
        fallback_rows = self._exclude_used_rows(fallback_rows, weekly_used_exercises, excluded_families)
        fallback_rows = self._sample_rows(fallback_rows, len(fallback_rows))

        for _, row in fallback_rows.iterrows():
            row_name = str(row.get("Exercise Name", "")).strip()
            row_family = self._exercise_family(row_name)
            if (
                not row_name
                or row_family in existing_non_session_families
                or row_family in excluded_families
            ):
                continue
            existing.append(self._format_exercise_from_dataset(row, fallback_section))
            existing_non_session_families.add(row_family)
            if len(existing) >= target_count:
                break

        if len(existing) < target_count:
            fallback_rows = self._sample_rows(fallback_rows, len(fallback_rows))
            for _, row in fallback_rows.iterrows():
                row_name = str(row.get("Exercise Name", "")).strip()
                row_family = self._exercise_family(row_name)
                if not row_name or row_family in existing_non_session_families:
                    continue
                existing.append(self._format_exercise_from_dataset(row, fallback_section))
                existing_non_session_families.add(row_family)
                if len(existing) >= target_count:
                    break

        non_session = [ex for ex in self._dedupe_exercise_variations(existing) if not ex.get("_is_session_payload")]
        return session_blocks + non_session[:max(0, target_count - len(session_blocks))]

    def _section_duration_label(self, exercises: List[Dict[str, Any]]) -> str:
        total_sec = sum(int(ex.get("est_time_sec", 0) or 0) for ex in exercises)
        if total_sec <= 0:
            return "None"
        total_min = max(1, round(total_sec / 60))
        return f"{total_min} min" if total_min == 1 else f"{total_min} mins"

    def _is_rest_day_text(self, activities_text: Any) -> bool:
        normalized = str(activities_text or "").strip().lower()
        return bool(re.search(r"\brest\b", normalized))

    def _extract_required_strength_days(self, ai_prescription: Dict[str, Any]) -> int:
        resistance = ai_prescription.get("resistance_training", {})
        frequency_text = ""
        if isinstance(resistance, dict):
            frequency_text = str(resistance.get("frequency") or "")
        numbers = [int(num) for num in re.findall(r"\d+", frequency_text)]
        return max(0, min(numbers[0], 7)) if numbers else 0

    def _enforce_frequency(
        self,
        activities: List[Dict[str, Any]],
        required_strength_days: int,
        weekly_strength_days_generated: int,
        remaining_active_days: int,
    ) -> tuple[List[Dict[str, Any]], bool]:
        if required_strength_days <= 0:
            return activities, False

        current = list(activities or [])
        has_strength = any(str(item.get("type", "")).lower() == "resistance" for item in current)
        if has_strength:
            return current, False
        has_yoga = any(str(item.get("type", "")).lower() == "yoga" for item in current)
        has_cardio_only = any(str(item.get("type", "")).lower() == "cardio" for item in current) and not has_yoga
        if has_yoga:
            return current, False

        strength_days_left = required_strength_days - weekly_strength_days_generated
        if strength_days_left <= 0:
            return current, False
        # Force resistance onto remaining days when needed to satisfy weekly frequency.
        if strength_days_left > remaining_active_days and has_cardio_only:
            current.insert(0, {"type": "resistance", "target_count": 4, "focus": "full", "forced_by_frequency": True})
            return current, True
        return current, False

    def _diversify_by_goal(self, goal: Any) -> Dict[str, float]:
        normalized = str(goal or "").strip().lower()
        if "weight" in normalized or "fat" in normalized:
            return {"cardio": 1.35, "resistance": 0.9, "mobility": 1.0}
        if "strength" in normalized or "muscle" in normalized:
            return {"cardio": 0.9, "resistance": 1.35, "mobility": 1.0}
        if any(term in normalized for term in ["rehab", "recovery", "therapy", "injury"]):
            return {"cardio": 0.85, "resistance": 1.0, "mobility": 1.4}
        return {"cardio": 1.0, "resistance": 1.0, "mobility": 1.0}

    def _apply_goal_weighting(
        self,
        rows: pd.DataFrame,
        activity_type: str,
        multipliers: Dict[str, float],
        day_index: int,
    ) -> pd.DataFrame:
        if rows.empty:
            return rows
        key = "cardio" if activity_type in {"cardio", "full_body_cardio"} else activity_type
        weight = float(multipliers.get(key, 1.0))
        if abs(weight - 1.0) < 0.01:
            return self._rotate_rows(rows, day_index)
        scored = rows.copy()
        scored["_goal_weight"] = scored.index.to_series().apply(lambda _: random.random() * weight)
        scored = scored.sort_values(by="_goal_weight", ascending=False).drop(columns=["_goal_weight"])
        return self._rotate_rows(scored, day_index)

    def _balance_equipment(self, rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty or "Equipments" not in rows.columns:
            return rows
        equipment_series = rows["Equipments"].astype(str)
        db_rows = rows[equipment_series.str.contains("dumbbell", case=False, na=False)]
        band_rows = rows[equipment_series.str.contains("band", case=False, na=False)]
        if db_rows.empty or band_rows.empty:
            return rows
        db_n = max(1, min(len(db_rows), int(round(len(rows) * 0.5))))
        band_n = max(1, min(len(band_rows), len(rows) - db_n))
        balanced = pd.concat([
            db_rows.sample(n=db_n, replace=(len(db_rows) < db_n), random_state=None),
            band_rows.sample(n=band_n, replace=(len(band_rows) < band_n), random_state=None),
        ])
        if len(balanced) < len(rows):
            remaining = rows.loc[~rows.index.isin(balanced.index)]
            needed = len(rows) - len(balanced)
            if not remaining.empty:
                balanced = pd.concat([balanced, remaining.sample(n=min(needed, len(remaining)), random_state=None)])
        return balanced.drop_duplicates(subset=["Exercise Name"], keep="first")

    def _extract_duration_minutes(self, text: Any, default: int = 30) -> int:
        normalized = str(text or "").lower()
        duration_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*min', normalized)
        return int(duration_match.group(1)) if duration_match else default

    def _apply_session_directives(
        self,
        day_name: str,
        activities: List[Dict[str, Any]],
        ai_prescription: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        directives = ai_prescription.get("session_directives", [])
        if not isinstance(directives, list):
            return activities

        updated = list(activities)
        normalized_day = str(day_name).strip().lower()
        for directive in directives:
            if not isinstance(directive, dict):
                continue
            directive_day = str(directive.get("day", "")).strip().lower()
            if directive_day != normalized_day:
                continue

            session_type = str(directive.get("session_type", "")).strip().lower()
            if not session_type:
                continue

            mapped_type = "full_body_cardio" if session_type in {"hiit", "interval", "circuit"} else session_type
            matched = False
            for activity in updated:
                if str(activity.get("type", "")).strip().lower() == mapped_type:
                    activity["label"] = str(directive.get("label") or directive.get("instruction") or activity.get("label") or "").strip()
                    activity["instruction"] = str(directive.get("instruction") or activity.get("instruction") or "").strip()
                    activity["duration"] = int(directive.get("duration_min") or activity.get("duration") or 30)
                    matched = True
                    break

            if not matched:
                updated.append({
                    "type": mapped_type,
                    "duration": int(directive.get("duration_min") or 30),
                    "label": str(directive.get("label") or directive.get("instruction") or "Workout Session").strip(),
                    "instruction": str(directive.get("instruction") or "").strip(),
                })

        return updated

    def _display_equipment_list(self, equipment: List[str]) -> List[str]:
        labels = {
            "bodyweight": "Bodyweight Only",
            "dumbbell": "Dumbbells",
            "resistance band": "Resistance Band",
            "treadmill": "Treadmill",
            "elliptical": "Elliptical",
            "cycling": "Bike",
            "brisk walking": "Walking",
        }
        return [labels.get(item, str(item).title()) for item in equipment]

    def _get_user_seed(self, profile: Dict[str, Any]) -> int:
        primary_goal = str(profile.get("primary_goal") or profile.get("goal") or "")
        intensity = str(profile.get("intensity") or profile.get("fitness_level") or "")
        equipment = ",".join(sorted(str(item).lower() for item in (profile.get("equipment_required") or [])))
        days_per_week = str(profile.get("days_per_week") or profile.get("days") or "")
        return hash(primary_goal + intensity + equipment + days_per_week) % 10000

    def _sample_rows(self, rows: pd.DataFrame, target_count: int) -> pd.DataFrame:
        if rows.empty or target_count <= 0:
            return rows.head(0)
        return rows.sample(n=min(target_count, len(rows)), replace=False, random_state=None)

    def _exercise_key(self, name: Any) -> str:
        return re.sub(r"\s+", " ", str(name or "").strip().lower())

    def _exclude_used_rows(
        self,
        rows: pd.DataFrame,
        weekly_used_exercises: Set[str] | None = None,
        excluded_families: Set[str] | None = None,
    ) -> pd.DataFrame:
        if rows.empty:
            return rows
        weekly_used_exercises = set(weekly_used_exercises or set())
        excluded_families = set(excluded_families or set())

        def is_unused(row: pd.Series) -> bool:
            name = row.get("Exercise Name", "")
            return (
                self._exercise_key(name) not in weekly_used_exercises
                and self._exercise_family(name) not in excluded_families
            )

        filtered = rows[rows.apply(is_unused, axis=1)]
        return filtered if not filtered.empty else rows

    def _profile_bodyweight_only(self, profile: Dict[str, Any]) -> bool:
        equipment = profile.get("available_equipment") or profile.get("equipment") or []
        if isinstance(equipment, str):
            equipment = [equipment]
        normalized = {str(item or "").strip().lower() for item in equipment}
        return bool(normalized) and normalized.issubset({"bodyweight only", "bodyweight", "no equipment", "none"})

    def _apply_strict_equipment_filter(self, rows: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        if rows.empty:
            return rows
        if self._profile_bodyweight_only(profile):
            bodyweight_rows = rows[rows["Equipments"].astype(str).str.contains("bodyweight", case=False, na=False)]
            return bodyweight_rows if not bodyweight_rows.empty else rows.head(0)
        return rows

    def _rotate_rows(self, rows: pd.DataFrame, offset: int) -> pd.DataFrame:
        if rows.empty:
            return rows
        offset = int(offset or 0) % len(rows)
        if offset == 0:
            return rows
        return pd.concat([rows.iloc[offset:], rows.iloc[:offset]])

    def _exercise_family(self, name: Any) -> str:
        text = str(name or "").lower()
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = re.sub(
            r"\b(incline|decline|knee|modified|assisted|wall|seated|standing|lying|"
            r"level|level-?\d+|beginner|advanced|alternate|alternating|single|double)\b",
            " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        family_patterns = {
            "push up": r"\b(push[\s-]*ups?|pushup|press[\s-]*ups?)\b",
            "mountain climber": r"\bmountain\s+climber",
            "lunge": r"\blunges?\b",
            "squat": r"\bsquats?\b",
            "plank": r"\bplank",
            "bridge": r"\bbridge",
            "row": r"\brow",
            "press": r"\bpress",
            "curl": r"\bcurl",
            "raise": r"\braise",
            "twist": r"\btwist",
            "crunch": r"\bcrunch",
            "leg raise": r"\bleg\s+raise",
            "windshield wiper": r"\bwindshield\s+wiper",
        }
        for family, pattern in family_patterns.items():
            if re.search(pattern, text):
                return family
        return text

    def _dedupe_exercise_variations(self, exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen_families: Set[str] = set()
        for exercise in exercises:
            if exercise.get("_is_session_payload"):
                deduped.append(exercise)
                continue
            family = self._exercise_family(exercise.get("name", ""))
            if not family or family in seen_families:
                continue
            seen_families.add(family)
            deduped.append(exercise)
        return deduped

    def _body_region_bucket(self, value: Any) -> str:
        text = str(value or "").lower()
        if any(term in text for term in ["upper", "shoulder", "arm", "chest", "back", "neck"]):
            return "upper"
        if any(term in text for term in ["lower", "leg", "hip", "glute", "calf", "ankle"]):
            return "lower"
        if "core" in text:
            return "core"
        return "full" if "full" in text else ""

    def _infer_main_body_regions(self, main_workout: List[Dict[str, Any]], activities_text: Any) -> Set[str]:
        regions: Set[str] = set()
        for exercise in main_workout:
            bucket = self._body_region_bucket(exercise.get("body_region", ""))
            if bucket:
                regions.add(bucket)
        text = str(activities_text or "").lower()
        if "upper" in text:
            regions.add("upper")
        if "lower" in text:
            regions.add("lower")
        if "core" in text:
            regions.add("core")
        if not regions or "full" in regions:
            regions.update({"upper", "lower"})
        return regions

    def _sanitize_sets(self, sets: Any, section: str) -> str:
        text = str(sets or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return "1" if section in {"warmup", "cooldown", "cardio", "session"} else "2"
        numbers = [int(num) for num in re.findall(r"\d+", text)]
        if numbers and max(numbers) > 12:
            return "1" if section in {"warmup", "cooldown", "cardio", "session"} else "2"
        return text

    def _sanitize_reps(self, reps: Any, section: str, category: Any = "") -> str:
        if isinstance(reps, int) and reps > 100:
            return self._default_reps(section, category)
        text = str(reps or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return self._default_reps(section, category)
        numbers = [int(num) for num in re.findall(r"\d+", text)]
        if re.fullmatch(r"\d{4,}", text) or any(num > 300 for num in numbers):
            return self._default_reps(section, category)
        return text

    def _default_reps(self, section: str, category: Any = "") -> str:
        category_text = str(category or "").lower()
        if section == "warmup":
            return "8-12 reps"
        if section == "cooldown":
            return "Hold 20-30 sec"
        if section in {"cardio", "session"} or "cardio" in category_text:
            return "30-45 sec"
        return "10-12 reps"

    def _sets_count(self, sets: Any) -> int:
        numbers = [int(num) for num in re.findall(r"\d+", str(sets or ""))]
        if not numbers:
            return 1
        return max(1, min(numbers[0], 12))

    def _format_rpe_label(self, rpe: Any) -> str:
        text = str(rpe or "").strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        return text if text.upper().startswith("RPE") else f"RPE {text}"

    def _rpe_midpoint(self, value: Any) -> float | None:
        numbers = [float(num) for num in re.findall(r"\d+(?:\.\d+)?", str(value or ""))]
        if not numbers:
            return None
        return sum(numbers[:2]) / min(len(numbers), 2)

    def _target_rpe_band(self, ai_prescription: Dict[str, Any]) -> tuple[float, float]:
        intensity = str(ai_prescription.get("intensity") or "").lower()
        if any(term in intensity for term in ["high", "vigorous", "hard"]):
            return (6.0, 8.0)
        if any(term in intensity for term in ["low", "light", "easy"]):
            return (2.0, 5.0)
        return (4.0, 7.0)

    def _filter_rows_by_intensity_band(self, rows: pd.DataFrame, ai_prescription: Dict[str, Any]) -> pd.DataFrame:
        if rows.empty:
            return rows
        low, high = self._target_rpe_band(ai_prescription)

        def in_band(row: pd.Series) -> bool:
            midpoint = self._rpe_midpoint(row.get("RPE", ""))
            return midpoint is None or low <= midpoint <= high

        filtered = rows[rows.apply(in_band, axis=1)]
        return filtered if not filtered.empty else rows

    def _normalize_intensity(self, day_exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        main_with_rpe = [
            (exercise, self._rpe_midpoint(exercise.get("intensity_rpe") or exercise.get("rpe")))
            for exercise in day_exercises
            if not exercise.get("_is_session_payload")
        ]
        numeric_rpes = sorted(rpe for _, rpe in main_with_rpe if rpe is not None)
        if len(numeric_rpes) < 2:
            return day_exercises

        midpoint = numeric_rpes[len(numeric_rpes) // 2]
        normalized: List[Dict[str, Any]] = []
        removed = 0
        for exercise in day_exercises:
            if exercise.get("_is_session_payload"):
                normalized.append(exercise)
                continue
            rpe = self._rpe_midpoint(exercise.get("intensity_rpe") or exercise.get("rpe"))
            if rpe is None or abs(rpe - midpoint) <= 2:
                normalized.append(exercise)
            else:
                removed += 1

        return normalized if normalized and removed < len(day_exercises) else day_exercises

    def _extract_available_equipment(self, ai_prescription: Dict[str, Any]) -> List[str]:
        equipment_items: List[str] = []

        for item in ai_prescription.get("equipment_required", []) or []:
            equipment_items.append(str(item or ""))

        resistance_training = ai_prescription.get("resistance_training", {})
        if isinstance(resistance_training, dict):
            equipment_items.append(str(resistance_training.get("equipment", "") or ""))

        cardio_requirements = ai_prescription.get("cardio_requirements", {})
        if isinstance(cardio_requirements, dict):
            for item in cardio_requirements.get("activities", []) or []:
                equipment_items.append(str(item or ""))

        for exercise in ai_prescription.get("prescribed_exercises", []) or []:
            if isinstance(exercise, dict):
                equipment_items.append(str(exercise.get("equipment", "") or ""))

        normalized: List[str] = []
        for raw_item in equipment_items:
            normalized_item = self._normalize_equipment_label(raw_item)
            if normalized_item and normalized_item not in normalized:
                normalized.append(normalized_item)
        return normalized or ["bodyweight"]

    def _extract_session_defaults(self, ai_prescription: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {}
        for session in ai_prescription.get("session_types", []) or []:
            if not isinstance(session, dict):
                continue
            session_type = str(session.get("type", "")).strip().lower()
            if session_type:
                defaults[session_type] = session
        return defaults

    def _extract_day_type_rules(self, ai_prescription: Dict[str, Any]) -> Dict[str, str]:
        rules: Dict[str, str] = {}
        for rule in ai_prescription.get("day_type_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            src = str(rule.get("type", "")).strip().lower()
            mapped = str(rule.get("mapped_to", "")).strip().lower()
            if src and mapped:
                rules[src] = mapped
        return rules

    def _resolve_session_label(self, session_type: str, session_defaults: Dict[str, Dict[str, Any]], fallback: str) -> str:
        session_info = session_defaults.get(str(session_type).lower(), {})
        return str(session_info.get("label") or fallback or "Workout Session").strip()

    def _resolve_session_duration(self, session_type: str, session_defaults: Dict[str, Dict[str, Any]], fallback: int) -> int:
        session_info = session_defaults.get(str(session_type).lower(), {})
        try:
            return int(session_info.get("duration") or fallback)
        except (TypeError, ValueError):
            return fallback

    def _normalize_equipment_label(self, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        mapping = {
            "db": "dumbbell",
            "dumbbells": "dumbbell",
            "dumbbell": "dumbbell",
            "rb": "resistance band",
            "band": "resistance band",
            "bands": "resistance band",
            "resistance band": "resistance band",
            "treadmill": "treadmill",
            "bike": "cycling",
            "cycle": "cycling",
            "cycling": "cycling",
            "spin bike": "cycling",
            "spinning": "cycling",
            "elliptical": "elliptical",
            "brisk walking": "brisk walking",
            "walking": "brisk walking",
            "bodyweight": "bodyweight",
            "no equipment": "bodyweight",
        }
        for key, mapped in mapping.items():
            if key in text:
                return mapped
        return text

    def _filter_rows_by_required_equipment(
        self,
        rows: pd.DataFrame,
        ai_prescription: Dict[str, Any],
        cardio_mode: bool = False,
    ) -> pd.DataFrame:
        if rows.empty:
            return rows

        required_equipment = self._extract_available_equipment(ai_prescription)
        strict_equipment = [item for item in required_equipment if item != "bodyweight"]
        if not strict_equipment:
            return rows

        def matches(row: pd.Series) -> bool:
            row_equipment = self._normalize_equipment_label(row.get("Equipments", ""))
            if cardio_mode:
                cardio_equipment = [item for item in strict_equipment if item in {"treadmill", "elliptical", "cycling"}]
                return not cardio_equipment or any(item in row_equipment for item in cardio_equipment)
            return any(item in row_equipment for item in strict_equipment)

        filtered = rows[rows.apply(matches, axis=1)]
        if not filtered.empty:
            return filtered

        bodyweight_rows = rows[rows["Equipments"].astype(str).str.lower().str.contains("bodyweight", na=False)]
        return bodyweight_rows if not bodyweight_rows.empty else rows

    def _prioritize_weekly_equipment_variety(self, rows: pd.DataFrame, weekly_used_equipment: Set[str]) -> pd.DataFrame:
        if rows.empty:
            return rows

        variety_targets = ["dumbbell", "resistance band", "bodyweight"]
        missing_targets = [item for item in variety_targets if item not in weekly_used_equipment]
        if not missing_targets:
            return rows

        prioritized_frames = []
        for target in missing_targets:
            target_rows = rows[rows["Equipments"].astype(str).str.lower().str.contains(target.replace(" ", "|"), na=False)]
            if not target_rows.empty:
                prioritized_frames.append(target_rows)

        if not prioritized_frames:
            return rows

        prioritized = pd.concat(prioritized_frames + [rows]).drop_duplicates(subset=["Exercise Name"], keep="first")
        return prioritized

    def _filter_cardio_rows_by_allowed_types(
        self,
        rows: pd.DataFrame,
        allowed_cardio_types: List[str],
    ) -> pd.DataFrame:
        if rows.empty:
            return rows

        normalized_allowed = {self._normalize_equipment_label(item) for item in allowed_cardio_types if item}
        if not normalized_allowed:
            return rows

        def is_allowed(row: pd.Series) -> bool:
            row_equipment = self._normalize_equipment_label(row.get("Equipments", ""))
            if "brisk walking" in normalized_allowed and row_equipment in {"brisk walking", "bodyweight", ""}:
                return True
            return row_equipment in normalized_allowed

        filtered = rows[rows.apply(is_allowed, axis=1)]
        return filtered if not filtered.empty else rows.head(0)

    def _get_allowed_cardio_types(self, ai_prescription: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
        allowed: List[str] = ["Brisk Walking"]
        available_equipment = {
            str(item or "").strip().lower()
            for item in (profile.get("available_equipment") or [])
            if str(item or "").strip()
        }

        if "treadmill" in available_equipment:
            allowed.append("Treadmill")
        if "cycling" in available_equipment or "bike" in available_equipment:
            allowed.append("Cycling")
        if "elliptical" in available_equipment:
            allowed.append("Elliptical")
        return list(dict.fromkeys(allowed))

    def _choose_cardio_mode(self, ai_prescription: Dict[str, Any], profile: Dict[str, Any], used_cardio_modes: Set[str] | None = None) -> str:
        used_cardio_modes = {str(item or "").strip().lower() for item in (used_cardio_modes or set()) if str(item or "").strip()}
        allowed = self._get_allowed_cardio_types(ai_prescription, profile)
        available = [mode for mode in allowed if mode.lower() not in used_cardio_modes]
        selection_pool = available or allowed or ["Brisk Walking"]
        return selection_pool[0]

    def _build_session_payload(
        self,
        activity: Dict[str, Any],
        ai_prescription: Dict[str, Any],
        profile: Dict[str, Any] | None = None,
        used_cardio_modes: Set[str] | None = None,
    ) -> Dict[str, Any]:
        session_type = str(activity.get("session_type") or activity.get("type") or "session").lower()
        duration = int(activity.get("duration") or 30)
        session_defaults = self._extract_session_defaults(ai_prescription)
        profile = profile or self._create_profile_from_ai(ai_prescription)
        default_label_map = {
            "cardio": "Cardio",
            "full_body_cardio": "Cardio",
            "yoga": "Yoga Session",
            "mobility": "Mobility Session",
        }
        label = str(activity.get("label") or "").strip() or self._resolve_session_label(
            session_type,
            session_defaults,
            default_label_map.get(session_type, "Workout Session"),
        )
        duration = int(activity.get("duration") or self._resolve_session_duration(session_type, session_defaults, duration) or 30)

        if session_type in {"cardio", "full_body_cardio"}:
            mode_name = self._choose_cardio_mode(ai_prescription, profile, used_cardio_modes=used_cardio_modes)
            name = f"Cardio ({mode_name})"
            benefit = "Cardiorespiratory conditioning and calorie burn"
            instruction = str(activity.get("instruction") or f"Perform {mode_name} continuously for {duration} minutes as prescribed in the note.").strip()
            equipment = mode_name
            met_value = 4.0
        elif session_type == "yoga":
            name = label
            benefit = "Improves flexibility, mobility, and recovery"
            instruction = str(activity.get("instruction") or f"{label} for {duration} minutes as prescribed in the note.").strip()
            equipment = ""
            met_value = 3.0
        else:
            name = label
            benefit = "Structured guided session"
            instruction = str(activity.get("instruction") or f"Complete {label} for {duration} minutes.").strip()
            equipment = ""
            met_value = 3.0

        estimated_calories = max(1, round(duration * 5))
        return {
            "name": name,
            "exercise_name": name,
            "sets": "1",
            "reps": f"{duration} min",
            "category": "session",
            "equipment": equipment,
            "benefit": benefit,
            "health_benefits": benefit,
            "safety_cue": "Follow coach guidance and maintain comfortable effort",
            "safety_notes": "Follow coach guidance and maintain comfortable effort",
            "steps": [instruction],
            "steps_to_perform": instruction,
            "rest": "As needed",
            "rest_time": "As needed",
            "intensity_rpe": "RPE 4-7",
            "estimated_calories": estimated_calories,
            "est_calories": f"Est: {estimated_calories} Cal",
            "est_time_sec": duration * 60,
            "est_time_human": self._humanize_seconds(duration * 60),
            "planned_sets_count": 1,
            "planned_total_cal": estimated_calories,
            "met_value": met_value,
            "_is_session_payload": True,
            "_session_type": session_type,
        }

    def _rank_rows_by_equipment(self, rows: pd.DataFrame, equipment_preferences: List[str]) -> pd.DataFrame:
        if rows.empty:
            return rows

        preferred = [str(item).lower() for item in equipment_preferences if item]

        def score(row: pd.Series) -> int:
            row_equipment = self._normalize_equipment_label(row.get("Equipments", ""))
            points = 0
            if preferred:
                if any(pref in row_equipment for pref in preferred if pref != "bodyweight"):
                    points += 10
                if "bodyweight" in preferred and "bodyweight" in row_equipment:
                    points += 2
            return points

        ranked = rows.copy()
        ranked["_equipment_score"] = ranked.apply(score, axis=1)
        ranked = ranked.sort_values(by=["_equipment_score"], ascending=False)
        return ranked.drop(columns=["_equipment_score"])

    def _pick_distinct_rows(
        self,
        rows: pd.DataFrame,
        count: int,
        excluded_families: Set[str] | None = None,
    ) -> pd.DataFrame:
        if rows.empty:
            return rows
        excluded_families = set(excluded_families or set())
        selected_indexes: List[Any] = []
        seen_families: Set[str] = set()

        for idx, row in rows.iterrows():
            family = self._exercise_family(row.get("Exercise Name", ""))
            if not family or family in seen_families or family in excluded_families:
                continue
            selected_indexes.append(idx)
            seen_families.add(family)
            if len(selected_indexes) >= count:
                break

        if len(selected_indexes) < count:
            for idx, row in rows.iterrows():
                family = self._exercise_family(row.get("Exercise Name", ""))
                if not family or family in seen_families:
                    continue
                selected_indexes.append(idx)
                seen_families.add(family)
                if len(selected_indexes) >= count:
                    break

        return rows.loc[selected_indexes]

    def _fallback_main_rows(
        self,
        df: pd.DataFrame,
        flags: Dict[str, bool],
        equipment_preferences: List[str],
        preferred_mode: str,
        focus_text: str,
    ) -> pd.DataFrame:
        """Build a ranked fallback pool when the first pass under-fills a day."""
        non_support_df = df[~df["Tags"].str.contains("warm up|cooldown", na=False)].copy()
        ai_equipment_payload = {"equipment_required": equipment_preferences}
        non_support_df = self._filter_rows_by_required_equipment(
            non_support_df,
            ai_equipment_payload,
            cardio_mode=(preferred_mode == "cardio"),
        )
        safe_df = non_support_df[non_support_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
        if safe_df.empty:
            safe_df = non_support_df

        if preferred_mode == "resistance":
            ranked = safe_df[safe_df["Primary Category"].str.lower().str.contains("strength|weight|resistance", na=False)].copy()
            if ranked.empty:
                ranked = safe_df.copy()

            if "upper" in focus_text:
                ranked = ranked[ranked["Body Region"].astype(str).str.lower().str.contains("upper|shoulder|arm|chest|back", na=False)]
            elif "lower" in focus_text:
                ranked = ranked[ranked["Body Region"].astype(str).str.lower().str.contains("lower|leg|hip|glute|calf", na=False)]
            elif "core" in focus_text:
                ranked = ranked[ranked["Body Region"].astype(str).str.lower().str.contains("core|full", na=False)]

            if ranked.empty:
                ranked = safe_df[safe_df["Primary Category"].str.lower().str.contains("strength|weight|resistance", na=False)].copy()
            return self._rank_rows_by_equipment(ranked, equipment_preferences)

        if preferred_mode == "yoga":
            ranked = safe_df[safe_df["Primary Category"].str.lower().str.contains("flexibility|stretching|mobility|yoga", na=False)].copy()
            return self._rank_rows_by_equipment(ranked if not ranked.empty else safe_df, equipment_preferences)

        ranked = safe_df[safe_df["Primary Category"].str.lower().str.contains("cardio", na=False)].copy()
        return self._rank_rows_by_equipment(ranked if not ranked.empty else safe_df, equipment_preferences)

    def _select_resistance_rows(
        self,
        rows: pd.DataFrame,
        target_count: int,
        focus: str,
        equipment_preferences: List[str],
        day_index: int = 0,
        excluded_families: Set[str] | None = None,
    ) -> pd.DataFrame:
        if rows.empty:
            return rows

        ranked = self._rotate_rows(self._rank_rows_by_equipment(rows, equipment_preferences), day_index * max(1, target_count))
        preferred_rows = ranked[
            ranked["Equipments"].astype(str).str.lower().apply(
                lambda value: any(pref in value for pref in equipment_preferences if pref != "bodyweight")
            )
        ]
        if not preferred_rows.empty:
            ranked = pd.concat([preferred_rows, ranked]).drop_duplicates(subset=["Exercise Name"], keep="first")
        body_region = ranked["Body Region"].astype(str).str.lower()

        upper_rows = ranked[body_region.str.contains("upper|shoulder|arm|chest|back", na=False)]
        lower_rows = ranked[body_region.str.contains("lower|leg|hip|glute|calf", na=False)]
        core_rows = ranked[body_region.str.contains("core|full", na=False)]

        selections: List[pd.DataFrame] = []
        if focus == "upper":
            selections.append(self._pick_distinct_rows(upper_rows, target_count, excluded_families=excluded_families))
        elif focus == "lower":
            selections.append(self._pick_distinct_rows(lower_rows, target_count, excluded_families=excluded_families))
        elif focus == "core":
            selections.append(self._pick_distinct_rows(core_rows, target_count, excluded_families=excluded_families))
        else:
            selections.append(self._pick_distinct_rows(upper_rows, 2, excluded_families=excluded_families))
            selections.append(self._pick_distinct_rows(lower_rows, 2, excluded_families=excluded_families))
            selections.append(self._pick_distinct_rows(core_rows, 2, excluded_families=excluded_families))

        selected = pd.concat(selections) if selections else ranked.head(0)
        selected = self._pick_distinct_rows(selected, target_count)

        if len(selected) < target_count:
            remaining = ranked[~ranked["Exercise Name"].isin(selected["Exercise Name"])]
            selected = pd.concat([
                selected,
                self._pick_distinct_rows(remaining, target_count - len(selected), excluded_families=excluded_families),
            ])
            selected = self._pick_distinct_rows(selected, target_count)

        return selected.head(target_count)

    def _build_support_exercises(
        self,
        rows: pd.DataFrame,
        section: str,
        count: int,
        sets: str,
        seen_names: Set[str],
        reps: str | None = None,
        min_seconds: int = 0,
    ) -> List[Dict[str, Any]]:
        exercises: List[Dict[str, Any]] = []
        if rows.empty:
            return exercises

        built_count = 0
        for _, row in rows.iterrows():
            family = self._exercise_family(row.get("Exercise Name", ""))
            if not family or family in seen_names:
                continue
            exercise = self._format_exercise_from_dataset(row, section)
            exercise["sets"] = sets
            exercise["planned_sets_count"] = 1
            if reps is not None:
                exercise["reps"] = reps
            if min_seconds:
                exercise["est_time_sec"] = max(int(exercise.get("est_time_sec", 0) or 0), min_seconds)
                exercise["est_time_human"] = self._humanize_seconds(exercise["est_time_sec"])
                exercise["estimated_calories"] = self._estimate_exercise_calories(float(exercise.get("met_value", 1.5) or 1.5), exercise["est_time_sec"])
                exercise["est_calories"] = f"Est: {exercise['estimated_calories']} Cal"
                exercise["planned_total_cal"] = exercise["estimated_calories"]
            seen_names.add(family)
            exercises.append(exercise)
            built_count += 1
            if built_count >= count:
                break
        return exercises

    def _split_steps(self, raw_steps: Any) -> List[str]:
        if isinstance(raw_steps, list):
            return [str(step).strip() for step in raw_steps if str(step).strip()]

        text = str(raw_steps or "").replace("\r", "\n").strip()
        if not text:
            return []

        parts = [part.strip() for part in re.split(r"\n+|(?=\d+\.\s)", text) if part.strip()]
        cleaned: List[str] = []
        for idx, part in enumerate(parts, start=1):
            if re.match(r"^\d+\.\s", part):
                cleaned.append(part)
            else:
                cleaned.append(f"{idx}. {part}")
        return cleaned

    def _estimate_exercise_time_sec(self, row: pd.Series, section: str, sets: Any) -> int:
        section_defaults = {
            "warmup": 60,
            "cooldown": 60,
            "cardio": 180,
            "hiit": 150,
            "resistance": 75,
            "strength": 75,
            "yoga": 75,
        }
        base_seconds = section_defaults.get(str(section).lower(), 75)
        try:
            sets_count = max(1, int(float(str(sets).strip() or "1")))
        except ValueError:
            sets_count = 1

        rest_text = str(row.get("Rest intervals", "") or "")
        rest_numbers = [int(num) for num in re.findall(r"\d+", rest_text)]
        rest_sec = rest_numbers[0] if rest_numbers else (45 if section in {"resistance", "strength"} else 30)
        total_sec = (base_seconds * sets_count) + (rest_sec * max(0, sets_count - 1))
        return max(45, total_sec)

    def _estimate_exercise_calories(self, met_value: float, est_time_sec: int) -> int:
        calories = (max(float(met_value or 1.5), 1.5) * 3.5 * 70 / 200) * (est_time_sec / 60)
        return max(1, round(calories))

    def _humanize_seconds(self, seconds: int) -> str:
        minutes, secs = divmod(max(int(seconds or 0), 0), 60)
        if minutes and secs:
            return f"{minutes}m {secs}s"
        if minutes:
            return f"{minutes}m"
        return f"{secs}s"

    def _exercise_is_safe_from_row(self, row: pd.Series, flags: Dict[str, bool]) -> bool:
        """
        Check if an exercise from dataset row is safe given clinical flags.
        """
        ex_text = " ".join([
            str(row.get("Exercise Name", "")).lower(),
            str(row.get("Primary Category", "")).lower(),
            str(row.get("Body Region", "")).lower(),
            str(row.get("Health benefit", "")).lower(),
        ])
        
        # Knee safety
        if flags.get("knee_sensitive"):
            unsafe_knee = ["jump", "hop", "plyo", "burpee", "box jump", "lunge", "squat", "box step"]
            if any(term in ex_text for term in unsafe_knee):
                return False
        
        # Floor work restrictions
        if flags.get("avoid_floor_work"):
            unsafe_floor = ["prone", "supine", "floor", "lying", "mat work", "plank", "push-up", "kneeling"]
            if any(term in ex_text for term in unsafe_floor):
                return False
        
        # High impact restrictions
        if flags.get("high_impact_restricted"):
            unsafe_impact = ["running", "sprinting", "jumping", "hopping", "high impact", "plyometric"]
            if any(term in ex_text for term in unsafe_impact):
                return False
        
        # Spinal loading restrictions
        if flags.get("avoid_spinal_loading"):
            unsafe_spine = ["deadlift", "squat", "heavy", "spinal", "back extension", "loaded"]
            if any(term in ex_text for term in unsafe_spine):
                return False
        
        # Overhead restrictions
        if flags.get("overhead_restricted"):
            unsafe_overhead = ["overhead", "press", "shoulder press", "snatch", "clean"]
            if any(term in ex_text for term in unsafe_overhead):
                return False
        
        return True

    def _format_exercise_from_dataset(self, row: pd.Series, section: str) -> Dict[str, Any]:
        """
        Format exercise data from dataset row into standard exercise format.
        """
        name = str(row.get("Exercise Name", "") or "")
        guid = str(row.get("GuidId", "") or "").strip() or None
        sets = self._sanitize_sets(row.get("Sets", "1"), section)
        reps = self._sanitize_reps(row.get("Reps", "10"), section, row.get("Primary Category", ""))
        benefit = str(row.get("Health benefit", "") or "")
        safety_cue = str(row.get("Safety cue", "") or "")
        steps = self._split_steps(row.get("Steps to perform", ""))
        rest_intervals = str(row.get("Rest intervals", "") or "")
        met_value = float(row.get("MET value", 3.0) or 3.0)
        est_time_sec = self._estimate_exercise_time_sec(row, section, sets)
        estimated_calories = self._estimate_exercise_calories(met_value, est_time_sec)
        if section in {"warmup", "cooldown"}:
            sets = "1"

        return {
            "name": name,
            "exercise_name": name,
            "sets": sets,
            "reps": reps,
            "category": section,
            "equipment": str(row.get("Equipments", "Bodyweight") or "Bodyweight"),
            "benefit": benefit,
            "health_benefits": benefit,
            "safety_cue": safety_cue,
            "safety_notes": safety_cue,
            "steps": steps,
            "steps_to_perform": "\n".join(steps),
            "body_region": str(row.get("Body Region", "") or ""),
            "primary_category": str(row.get("Primary Category", "") or ""),
            "tags": str(row.get("Tags", "") or ""),
            "video_url": str(row.get("Video Link", "") or ""),
            "thumbnail_url": "",
            "video_path": "",
            "image_path": "",
            "unique_id": str(row.get("Unique ID", "") or ""),
            "guid_id": guid,
            "gui_id": guid,
            "age_suitability": str(row.get("Age Suitability", "") or ""),
            "goal": str(row.get("Goal", "") or ""),
            "fitness_level": str(row.get("Fitness Level", "") or ""),
            "physical_limitations": str(row.get("Physical limitation", "") or ""),
            "rpe": str(row.get("RPE", "") or ""),
            "intensity_rpe": self._format_rpe_label(row.get("RPE", "")),
            "rest_intervals": rest_intervals,
            "rest": rest_intervals,
            "rest_time": rest_intervals,
            "met_value": met_value,
            "est_calories": f"Est: {estimated_calories} Cal",
            "estimated_calories": estimated_calories,
            "est_time_sec": est_time_sec,
            "est_time_human": self._humanize_seconds(est_time_sec),
            "planned_sets_count": self._sets_count(sets),
            "planned_total_cal": estimated_calories,
            "is_not_suitable_for": str(row.get("is_not_suitable_for (Medical conditions)", "") or ""),
        }
