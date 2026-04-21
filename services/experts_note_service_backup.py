from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Set

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None

import pandas as pd

from core.fitness import (
    ExerciseFilter,
    FitnessDataset,
    _hard_medical_exclusion,
)
from core.fitness_engine import FitnessEngine
from utils.constants import DAY_ORDER, DATASET_DIR
from utils.text_normalizer import normalize_text

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
        Generate a complete workout plan from expert notes.
        
        Steps:
        1. Parse notes to extract clinical context
        2. Inject into default profile
        3. Use FitnessEngine to generate plan with medical guardrails
        4. Verify and filter unsafe exercises
        
        Args:
            notes_text: Raw text from expert/doctor notes
            
        Returns:
            Dictionary with 'plan', 'profile', and 'clinical_context'
        """
        if not notes_text or not notes_text.strip():
            raise ValueError("No notes provided")

        # STEP 1: Parse clinical context from notes
        clinical_context = self._parse_clinical_context(notes_text)

        # STEP 2: Create profile with extracted context
        profile = self._get_default_profile()
        profile["medical_conditions"] = clinical_context["medical_conditions"]
        profile["physical_limitation"] = ", ".join(clinical_context["physical_limitations"]) if clinical_context["physical_limitations"] else "None"
        profile["specific_avoidance"] = ", ".join(clinical_context["restrictions"]) if clinical_context["restrictions"] else "None"
        profile["flags"] = clinical_context["flags"]
        
        logger.info(f"[ExpertsNoteService] Updated profile with clinical context")

        # STEP 3: Generate plan using FitnessEngine (which applies all filters)
        plan = self.engine.generate_plan(profile)

        # STEP 4: Apply safety verification to remove any unsafe exercises
        plan = self._verify_and_filter_plan(plan, clinical_context)

        logger.info(f"[ExpertsNoteService] Plan generated and safety-verified for {len(plan)} days")
        
        return {
            "plan": plan,
            "profile": profile,
            "notes_text": notes_text,
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
