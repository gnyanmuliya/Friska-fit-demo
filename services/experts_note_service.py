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

from core.azure_ai_parser import AzureAIPrescriptionParser
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

        # STEP 1: Parse clinical context using Azure AI
        ai_prescription = self.ai_parser.parse_notes(notes_text)

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
        }

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
        profile = profile or self._create_profile_from_ai(ai_prescription)
        user_seed = self._get_user_seed(profile)
        
        for day_index, (day_name, activities_text) in enumerate(weekly_schedule.items()):
            random.seed(user_seed + day_index)
            normalized_day_name = str(day_name).capitalize()
            if self._is_rest_day_text(activities_text):
                plan[normalized_day_name] = self._build_structured_day_plan(
                    day_name=normalized_day_name,
                    day_plan={"warmup": [], "main_workout": [], "cooldown": []},
                    activities_text=activities_text,
                    ai_prescription=ai_prescription,
                    is_rest_day=True,
                )
                continue
            
            # Parse activities from the text
            activities = self._parse_day_activities(activities_text, ai_prescription)
            activities = self._apply_session_directives(normalized_day_name, activities, ai_prescription)
            priority_order = {
                "full_body_cardio": 0,
                "cardio": 1,
                "resistance": 2,
                "yoga": 3,
            }
            activities = sorted(
                activities,
                key=lambda item: priority_order.get(str(item.get("type", "")).lower(), 99),
            )
            seen_types: Set[str] = set()
            unique_activities: List[Dict[str, Any]] = []
            for activity in activities:
                activity_type = str(activity.get("type", "")).lower()
                if activity_type in seen_types:
                    continue
                seen_types.add(activity_type)
                unique_activities.append(activity)
            activities = unique_activities
            
            day_plan = {"warmup": [], "main_workout": [], "cooldown": []}
            desired_main_count = sum(int(activity.get("target_count", 0) or 0) for activity in activities) or 6
            
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
            
            plan[normalized_day_name] = self._build_structured_day_plan(
                day_name=normalized_day_name,
                day_plan=day_plan,
                activities_text=activities_text,
                ai_prescription=ai_prescription,
            )
        
        return plan

    def _parse_day_activities(self, activities_text: str, ai_prescription: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse activity descriptions from daily schedule text.
        """
        text_lower = str(activities_text or "").lower()
        has_cardio = "cardio" in text_lower
        has_yoga = "yoga" in text_lower
        has_hiit = any(term in text_lower for term in ["hiit", "interval", "circuit"])
        has_strength = any(term in text_lower for term in ["weight", "weights", "resistance", "strength", "dumbbell", "band"])

        ordered_types: List[Dict[str, Any]] = []

        def _add(activity_type: str, **kwargs: Any) -> None:
            if any(str(item.get("type", "")).lower() == activity_type for item in ordered_types):
                return
            payload = {"type": activity_type}
            payload.update(kwargs)
            ordered_types.append(payload)

        if has_hiit and has_yoga:
            _add("full_body_cardio", target_count=5)
            _add("yoga", target_count=1)
        elif has_hiit:
            _add("full_body_cardio", target_count=5)
        elif has_cardio and has_yoga:
            _add("cardio", target_count=1)
            _add("yoga", target_count=1)
        elif has_strength and has_cardio:
            _add("resistance", target_count=4, focus="full", paired_with="cardio")
            _add("cardio", target_count=1)
        elif has_strength:
            _add("resistance", target_count=6, focus="full")
        elif has_cardio:
            _add("cardio", target_count=1)
        elif has_yoga:
            _add("yoga", target_count=1)
        else:
            _add("cardio", target_count=1)

        return ordered_types

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
    ) -> List[Dict[str, Any]]:
        """
        Generate specific exercises for an activity type from dataset with complete information.
        """
        activity_type = activity.get("type")
        exercises: List[Dict[str, Any]] = []
        flags = clinical_context.get("flags", {})
        target_count = int(activity.get("target_count", 6) or 6)
        equipment_preferences = self._extract_available_equipment(ai_prescription)
        profile = profile or self._create_profile_from_ai(ai_prescription)
        weekly_used_exercises = set(weekly_used_exercises or set())

        if activity_type == "resistance":
            target_count = 4 if "cardio" in str(activity.get("paired_with", "")).lower() else target_count
            resistance_df = df[df["Primary Category"].str.lower().str.contains("strength|weight|resistance", na=False)].copy()
            resistance_df = resistance_df[~resistance_df["Tags"].str.contains("warm up|cooldown", na=False)]
            resistance_df = self._apply_strict_equipment_filter(resistance_df, profile)
            resistance_df = self._filter_rows_by_required_equipment(resistance_df, ai_prescription)
            safe_df = resistance_df[resistance_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
            if safe_df.empty:
                safe_df = resistance_df

            focus = str(activity.get("focus") or "full").lower()
            safe_df = self._filter_rows_by_intensity_band(safe_df, ai_prescription)
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
            exercises.append(self._build_session_payload({"session_type": "cardio"}, ai_prescription))

        elif activity_type == "full_body_cardio":
            target_count = 5
            cardio_df = df[df["Primary Category"].str.lower().str.contains("cardio", na=False)].copy()
            cardio_df = cardio_df[~cardio_df["Tags"].str.contains("warm up|cooldown", na=False)]
            cardio_df = self._apply_strict_equipment_filter(cardio_df, profile)
            cardio_df = self._filter_rows_by_required_equipment(cardio_df, ai_prescription, cardio_mode=True)
            safe_df = cardio_df[cardio_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
            if safe_df.empty:
                safe_df = cardio_df

            safe_df = self._filter_rows_by_intensity_band(safe_df, ai_prescription)
            safe_df = self._exclude_used_rows(safe_df, weekly_used_exercises, excluded_families)
            selected_rows = self._pick_distinct_rows(self._rank_rows_by_equipment(safe_df, equipment_preferences), target_count, excluded_families=excluded_families)
            selected_rows = self._sample_rows(selected_rows, target_count)
            for _, row in selected_rows.iterrows():
                exercises.append(self._format_exercise_from_dataset(row, "cardio"))

        elif activity_type == "yoga":
            exercises.append(self._build_session_payload({"session_type": "yoga"}, ai_prescription))

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
            unsafe_knee = ["jump", "hop", "plyo", "burpee", "box jump", "skipping", "lunge", "squat", "box step"]
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

    def _generate_day_title(self, day_plan: Dict[str, Any], activities_text: str) -> str:
        """
        Generate day title based on the day's main workout exercises.
        """
        day_focus = self._classify_day_focus(day_plan, activities_text)
        return day_focus["workout_title"]

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
        if "pilates" in normalized_activity_text:
            activity_types.append("pilates")

        unique_activities = list(dict.fromkeys(activity_types))

        if "full_body_cardio" in unique_activities:
            category = "Full Body Cardio"
        elif "strength" in unique_activities and "cardio" in unique_activities:
            category = "Strength + Cardio"
        elif "yoga" in unique_activities and "cardio" in unique_activities:
            category = "Yoga + Cardio"
        elif "pilates" in unique_activities and "cardio" in unique_activities:
            category = "Pilates + Cardio"
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
        elif "pilates" in unique_activities:
            category = "Pilates"
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
        ai_prescription: Dict[str, Any],
        is_rest_day: bool = False,
    ) -> Dict[str, Any]:
        """Convert an internal day plan into the frontend response structure."""
        day_focus = self._classify_day_focus(day_plan, activities_text)
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
        if len(existing) >= target_count:
            return existing[:target_count]

        flags = clinical_context.get("flags", {})
        equipment_preferences = self._extract_available_equipment(ai_prescription)
        existing_names = {str(ex.get("name", "")).strip().lower() for ex in existing if str(ex.get("name", "")).strip()}
        normalized_text = str(activities_text or "").lower()
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
        return normalized in {"rest", "rest day", "rest or bike ride/brisk walk"}

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
        days_per_week = str(profile.get("days_per_week") or profile.get("days") or "")
        intensity = str(profile.get("intensity") or profile.get("fitness_level") or "")
        focus = str(profile.get("focus") or profile.get("target_body_parts") or "")
        return hash(primary_goal + days_per_week + intensity + focus) % 10000

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

    def _choose_cardio_mode(self, ai_prescription: Dict[str, Any]) -> str:
        required_equipment = self._extract_available_equipment(ai_prescription)
        cardio_requirements = ai_prescription.get("cardio_requirements", {})
        activities = cardio_requirements.get("activities", []) if isinstance(cardio_requirements, dict) else []
        allowed_modes = [
            ("treadmill", "Treadmill"),
            ("elliptical", "Elliptical"),
            ("cycling", "Cycling"),
            ("brisk walking", "Brisk Walking"),
        ]

        for normalized_key, label in allowed_modes:
            if normalized_key in required_equipment:
                return label

        for activity in activities or []:
            normalized_activity = self._normalize_equipment_label(activity)
            for normalized_key, label in allowed_modes:
                if normalized_activity == normalized_key:
                    return label

        return "Brisk Walking"

    def _build_session_payload(self, activity: Dict[str, Any], ai_prescription: Dict[str, Any]) -> Dict[str, Any]:
        session_type = str(activity.get("session_type") or activity.get("type") or "session").lower()
        duration = int(activity.get("duration") or 30)
        session_defaults = self._extract_session_defaults(ai_prescription)
        default_label_map = {
            "cardio": "Cardio Session",
            "full_body_cardio": "Full Body Cardio",
            "yoga": "Yoga Session",
            "pilates": "Pilates Session",
        }
        label = str(activity.get("label") or "").strip() or self._resolve_session_label(
            session_type,
            session_defaults,
            default_label_map.get(session_type, "Workout Session"),
        )
        duration = int(activity.get("duration") or self._resolve_session_duration(session_type, session_defaults, duration) or 30)

        if session_type in {"cardio", "full_body_cardio"}:
            mode_name = self._choose_cardio_mode(ai_prescription)
            name = f"{label} ({mode_name})"
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
        elif session_type == "pilates":
            name = label
            benefit = "Improves core control, posture, and mobility"
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
            unsafe_knee = ["jump", "hop", "plyo", "burpee", "box jump", "skipping", "lunge", "squat", "box step"]
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
