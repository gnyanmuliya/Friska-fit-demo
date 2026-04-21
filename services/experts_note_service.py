from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Set

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None

import pandas as pd

from core.azure_ai_parser import AzureAIPrescriptionParser
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
            "available_equipment": ["Bodyweight Only"],  # Will be updated based on AI prescription
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

    def _generate_plan_from_weekly_schedule(self, df: pd.DataFrame, ai_prescription: Dict[str, Any], 
                                           clinical_context: Dict[str, Any]) -> Dict[str, dict]:
        """Generate workout plan following the exact weekly schedule from AI prescription."""
        weekly_schedule = ai_prescription.get("weekly_schedule", {})
        plan = {}
        
        # Import required classes
        from core.fitness import ExerciseSelector, WorkoutComposer, DatasetRanker
        
        # Create selector and composer
        selector = ExerciseSelector()
        composer = WorkoutComposer(selector)
        composer._dataset = df
        composer._active_clinical_context = clinical_context
        
        # Match prescribed exercises
        prescribed_exercises = self._match_prescribed_exercises(ai_prescription.get("prescribed_exercises", []))
        
        # Process each day in the schedule
        for day_name, day_description in weekly_schedule.items():
            day_name_cap = day_name.capitalize()
            day_plan = {"warmup": [], "main_workout": [], "cooldown": []}
            
            if not day_description or day_description.lower() in ["rest", "rest or bike ride/brisk walk"]:
                # Rest day or optional activity
                plan[day_name_cap] = day_plan
                continue
            
            # Parse day description to determine activities
            activities = self._parse_day_activities(day_description, ai_prescription)
            
            # Generate exercises for each activity type
            main_exercises = []
            
            for activity in activities:
                activity_exercises = self._generate_activity_exercises(
                    activity, df, ai_prescription, prescribed_exercises, selector, composer
                )
                main_exercises.extend(activity_exercises)
            
            day_plan["main_workout"] = main_exercises
            
            # Add basic warmup and cooldown
            day_plan["warmup"] = self._get_basic_warmup_for_day(day_description)
            day_plan["cooldown"] = self._get_basic_cooldown_for_day(day_description)
            
            plan[day_name_cap] = day_plan
        
        return plan

    def _parse_day_activities(self, day_description: str, ai_prescription: Dict[str, Any]) -> List[str]:
        """Parse day description to extract activity types."""
        activities = []
        desc_lower = day_description.lower()
        
        # Check for cardio
        if "cardio" in desc_lower:
            activities.append("cardio")
        
        # Check for weights/resistance
        if any(term in desc_lower for term in ["weight", "weights", "resistance"]):
            activities.append("resistance")
        
        # Check for HIIT/Interval
        if any(term in desc_lower for term in ["hiit", "interval"]):
            activities.append("hiit")
        
        # Check for yoga
        if "yoga" in desc_lower:
            activities.append("yoga")
        
        # If no specific activities found, default to cardio
        if not activities:
            activities.append("cardio")
        
        return activities

    def _generate_activity_exercises(self, activity: str, df: pd.DataFrame, ai_prescription: Dict[str, Any],
                                   prescribed_exercises: List[Dict[str, Any]], selector, composer) -> List[Dict[str, Any]]:
        """Generate exercises for a specific activity type."""
        exercises = []
        
        if activity == "cardio":
            cardio_req = ai_prescription.get("cardio_requirements", {})
            duration = cardio_req.get("duration", "45-60 minutes")
            activities_list = cardio_req.get("activities", ["Walking", "Bike", "Elliptical"])
            
            # Select primary cardio activity
            primary_activity = activities_list[0] if activities_list else "Walking"
            
            exercise = {
                "name": primary_activity,
                "sets": 1,
                "reps": duration,
                "category": "cardio",
                "equipment": "Bodyweight",
                "benefit": f"Cardiovascular exercise for {duration}",
                "safety_cue": "Maintain moderate intensity",
                "steps": "",
                "body_region": "Full Body",
                "primary_category": "Cardio",
                "tags": "cardio",
                "video_url": "",
                "thumbnail_url": "",
                "video_path": "",
                "image_path": "",
            }
            exercises.append(exercise)
            
        elif activity == "resistance":
            resistance_req = ai_prescription.get("resistance_training", {})
            exercise_list = resistance_req.get("exercises", ["Chest press", "Rows", "Shoulder press", "Bicep curls", "Tricep extension", "Squat"])
            sets = resistance_req.get("sets", 3)
            reps = resistance_req.get("reps", "10-15")
            equipment = resistance_req.get("equipment", "Bodyweight")
            
            # Use prescribed exercises if available, otherwise select from list
            if prescribed_exercises:
                for prescribed in prescribed_exercises:
                    if any(term in prescribed["name"].lower() for term in ["press", "row", "curl", "extension", "squat"]):
                        exercises.append(prescribed)
            else:
                # Select exercises from the dataset
                for ex_name in exercise_list[:4]:  # Limit to 4 exercises
                    matched = self._find_exercise_in_dataset(df, ex_name)
                    if matched is not None:
                        exercise = composer.format_exercise(
                            matched, 
                            sets=str(sets),
                            reps=reps,
                            rest="60 seconds",
                            rpe="6-8"
                        )
                        exercises.append(exercise)
                        
        elif activity == "hiit":
            hiit_req = ai_prescription.get("hiit_training", {})
            duration = hiit_req.get("duration", "30-45 mins")
            structure = hiit_req.get("structure", "8 rounds 20 seconds/10 seconds")
            
            # Create HIIT workout structure
            exercise = {
                "name": "HIIT Tabata Workout",
                "sets": 8,
                "reps": "20 seconds work/10 seconds rest",
                "category": "hiit",
                "equipment": "Bodyweight",
                "benefit": f"High-intensity interval training for {duration}",
                "safety_cue": "Alternate high and recovery intensity",
                "steps": structure,
                "body_region": "Full Body",
                "primary_category": "HIIT",
                "tags": "hiit, cardio",
                "video_url": "",
                "thumbnail_url": "",
                "video_path": "",
                "image_path": "",
            }
            exercises.append(exercise)
            
        elif activity == "yoga":
            exercise = {
                "name": "Yoga Class",
                "sets": 1,
                "reps": "60 minutes",
                "category": "yoga",
                "equipment": "Bodyweight",
                "benefit": "Yoga for flexibility and stress relief",
                "safety_cue": "Follow instructor guidance",
                "steps": "",
                "body_region": "Full Body",
                "primary_category": "Mobility/Stretch",
                "tags": "yoga, flexibility",
                "video_url": "",
                "thumbnail_url": "",
                "video_path": "",
                "image_path": "",
            }
            exercises.append(exercise)
        
        return exercises

    def _find_exercise_in_dataset(self, df: pd.DataFrame, exercise_name: str):
        """Find exercise in dataset by name."""
        import difflib
        
        exercise_names = df["Exercise Name"].str.lower()
        matches = difflib.get_close_matches(exercise_name.lower(), exercise_names, n=1, cutoff=0.6)
        
        if matches:
            matched_name = matches[0]
            return df[df["Exercise Name"].str.lower() == matched_name].iloc[0]
        
        return None

    def _get_basic_warmup_for_day(self, day_description: str) -> List[Dict[str, Any]]:
        """Get appropriate warmup for the day."""
        desc_lower = day_description.lower()
        
        if "yoga" in desc_lower:
            return [{
                "name": "Gentle Yoga Warm-Up",
                "sets": "1",
                "reps": "5 minutes",
                "category": "warmup",
                "equipment": "Bodyweight",
            }]
        else:
            return [{
                "name": "Dynamic Warm-Up",
                "sets": "1",
                "reps": "5-10 minutes",
                "category": "warmup",
                "equipment": "Bodyweight",
            }]

    def _get_basic_cooldown_for_day(self, day_description: str) -> List[Dict[str, Any]]:
        """Get appropriate cooldown for the day."""
        desc_lower = day_description.lower()
        
        if "yoga" in desc_lower:
            return [{
                "name": "Yoga Cool-Down",
                "sets": "1",
                "reps": "5 minutes",
                "category": "cooldown",
                "equipment": "Bodyweight",
            }]
        else:
            return [{
                "name": "Static Stretching",
                "sets": "1",
                "reps": "5-10 minutes",
                "category": "cooldown",
                "equipment": "Bodyweight",
            }]

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
        plan = self._generate_plan_from_weekly_schedule(df, ai_prescription, clinical_context)

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
            "fitness_level": "intermediate",  # Default
            "days_per_week": ai_prescription.get("days_per_week", 3),
            "session_duration": ai_prescription.get("session_duration", 45),
            "focus": ai_prescription.get("focus", []),
            "intensity": ai_prescription.get("intensity", "moderate"),
            "target_body_parts": ai_prescription.get("target_body_parts", []),
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

    def _generate_plan_from_weekly_schedule(self, df: pd.DataFrame, ai_prescription: Dict[str, Any], clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate workout plan following the exact weekly schedule from AI prescription.
        """
        weekly_schedule = ai_prescription.get("weekly_schedule", {})
        plan = {}
        
        for day_name, activities_text in weekly_schedule.items():
            normalized_day_name = str(day_name).capitalize()
            if str(activities_text or "").strip().lower() == "rest":
                plan[normalized_day_name] = self._build_structured_day_plan(
                    day_name=normalized_day_name,
                    day_plan={"warmup": [], "main_workout": [], "cooldown": []},
                    activities_text=activities_text,
                    ai_prescription=ai_prescription,
                    is_rest_day=True,
                )
                continue
            
            # Parse activities from the text
            activities = self._parse_day_activities(activities_text)
            
            day_plan = {"warmup": [], "main_workout": [], "cooldown": []}
            
            # Generate main workout exercises
            for activity in activities:
                exercises = self._generate_activity_exercises(df, activity, ai_prescription, clinical_context)
                if activity.get("type") == "cardio":
                    day_plan["main_workout"].extend(exercises)
                elif activity.get("type") == "resistance":
                    day_plan["main_workout"].extend(exercises)
                elif activity.get("type") == "hiit":
                    day_plan["main_workout"].extend(exercises)
                elif activity.get("type") == "yoga":
                    day_plan["cooldown"].extend(exercises)
            
            # Generate fixed warmup exercises
            day_plan["warmup"] = self._generate_warmup_exercises(df, activities_text, clinical_context)
            
            # Generate fixed cooldown exercises
            day_plan["cooldown"] = self._generate_cooldown_exercises(df, activities_text, clinical_context)
            
            plan[normalized_day_name] = self._build_structured_day_plan(
                day_name=normalized_day_name,
                day_plan=day_plan,
                activities_text=activities_text,
                ai_prescription=ai_prescription,
            )
        
        return plan

    def _parse_day_activities(self, activities_text: str) -> List[Dict[str, Any]]:
        """
        Parse activity descriptions from daily schedule text.
        """
        activities = []
        text_lower = activities_text.lower()
        
        # Check for cardio
        if "cardio" in text_lower:
            duration_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*min', text_lower)
            duration = 30  # default
            if duration_match:
                duration = int(duration_match.group(1))
            
            activities.append({
                "type": "cardio",
                "duration": duration,
                "description": "Cardiovascular exercise"
            })
        
        # Check for weights/resistance
        if any(term in text_lower for term in ["weight", "resistance", "strength"]):
            activities.append({
                "type": "resistance",
                "sets": 3,
                "reps": "10-15",
                "description": "Resistance training"
            })
        
        # Check for HIIT
        if "hiit" in text_lower or "interval" in text_lower:
            duration_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?\s*min', text_lower)
            duration = 30  # default
            if duration_match:
                duration = int(duration_match.group(1))
            
            activities.append({
                "type": "hiit",
                "duration": duration,
                "description": "High-intensity interval training"
            })
        
        # Check for yoga
        if "yoga" in text_lower:
            activities.append({
                "type": "yoga",
                "duration": 30,
                "description": "Yoga practice"
            })
        
        return activities

    def _generate_activity_exercises(self, df: pd.DataFrame, activity: Dict[str, Any], ai_prescription: Dict[str, Any], clinical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate specific exercises for an activity type from dataset with complete information.
        """
        activity_type = activity.get("type")
        exercises = []
        flags = clinical_context.get("flags", {})
        
        if activity_type == "cardio":
            # Select cardio exercises from dataset
            cardio_df = df[df["Primary Category"].str.lower().str.contains("cardio", na=False)]
            # Filter out main workout tagged exercises to get variety
            cardio_df = cardio_df[~cardio_df["Tags"].str.contains("warm up|cooldown", na=False)]
            
            if not cardio_df.empty:
                # Apply safety filters
                safe_df = cardio_df[cardio_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
                if safe_df.empty:
                    safe_df = cardio_df
                
                selected = safe_df.sample(min(2, len(safe_df)))
                for _, row in selected.iterrows():
                    exercises.append(self._format_exercise_from_dataset(row, "cardio"))
        
        elif activity_type == "resistance":
            # Select strength exercises from dataset
            resistance_df = df[df["Primary Category"].str.lower().str.contains("strength|weight|resistance", na=False)]
            # Filter out warmup/cooldown
            resistance_df = resistance_df[~resistance_df["Tags"].str.contains("warm up|cooldown", na=False)]
            
            if not resistance_df.empty:
                # Apply safety filters
                safe_df = resistance_df[resistance_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
                if safe_df.empty:
                    safe_df = resistance_df
                
                selected = safe_df.sample(min(4, len(safe_df)))
                for _, row in selected.iterrows():
                    exercises.append(self._format_exercise_from_dataset(row, "resistance"))
        
        elif activity_type == "hiit":
            # Use cardio exercises for HIIT
            hiit_df = df[df["Primary Category"].str.lower().str.contains("cardio", na=False)]
            hiit_df = hiit_df[~hiit_df["Tags"].str.contains("warm up|cooldown", na=False)]
            
            if not hiit_df.empty:
                # Apply safety filters
                safe_df = hiit_df[hiit_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
                if safe_df.empty:
                    safe_df = hiit_df
                
                selected = safe_df.sample(min(3, len(safe_df)))
                for _, row in selected.iterrows():
                    exercises.append(self._format_exercise_from_dataset(row, "hiit"))
        
        elif activity_type == "yoga":
            # Use flexibility/stretching or mobility exercises for yoga
            yoga_df = df[df["Primary Category"].str.lower().str.contains("flexibility|stretching|mobility|yoga", na=False)]
            yoga_df = yoga_df[~yoga_df["Tags"].str.contains("warm up|main work out", na=False)]
            
            if not yoga_df.empty:
                # Apply safety filters
                safe_df = yoga_df[yoga_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
                if safe_df.empty:
                    safe_df = yoga_df
                
                selected = safe_df.sample(min(3, len(safe_df)))
                for _, row in selected.iterrows():
                    exercises.append(self._format_exercise_from_dataset(row, "yoga"))
        
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

    def _generate_warmup_exercises(self, df: pd.DataFrame, activities_text: str, clinical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3 fixed warmup exercises: 1 cardio, 1 upper body mobility, 1 lower body mobility.
        All exercises must be tagged as "Warm up" in the dataset.
        """
        warmup_exercises = []
        flags = clinical_context.get("flags", {})
        
        # Filter warmup exercises from dataset
        warmup_df = df[df["Tags"].str.contains("warm up", na=False)].copy()
        
        # Apply safety filters
        safe_warmup_df = warmup_df[warmup_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
        
        if safe_warmup_df.empty:
            safe_warmup_df = warmup_df  # Fallback if no safe exercises
        
        # 1. Select 1 cardio warmup exercise
        cardio_warmup = safe_warmup_df[safe_warmup_df["Primary Category"].str.lower().str.contains("cardio", na=False)]
        if not cardio_warmup.empty:
            selected = cardio_warmup.sample(1)
            for _, row in selected.iterrows():
                warmup_exercises.append(self._format_exercise_from_dataset(row, "warmup"))
        
        # 2. Select 1 upper body mobility exercise
        upper_mobility = safe_warmup_df[
            (safe_warmup_df["Primary Category"].str.lower().str.contains("mobility", na=False)) &
            (safe_warmup_df["Body Region"].str.lower().str.contains("upper|shoulder|arm", na=False))
        ]
        if not upper_mobility.empty:
            selected = upper_mobility.sample(1)
            for _, row in selected.iterrows():
                warmup_exercises.append(self._format_exercise_from_dataset(row, "warmup"))
        
        # 3. Select 1 lower body mobility exercise
        lower_mobility = safe_warmup_df[
            (safe_warmup_df["Primary Category"].str.lower().str.contains("mobility", na=False)) &
            (safe_warmup_df["Body Region"].str.lower().str.contains("lower|leg|hip", na=False))
        ]
        if not lower_mobility.empty:
            selected = lower_mobility.sample(1)
            for _, row in selected.iterrows():
                warmup_exercises.append(self._format_exercise_from_dataset(row, "warmup"))
        
        # If we don't have 3 exercises, fill with any safe warmup exercises
        while len(warmup_exercises) < 3 and not safe_warmup_df.empty:
            remaining = safe_warmup_df[~safe_warmup_df["Exercise Name"].isin([ex["name"] for ex in warmup_exercises])]
            if not remaining.empty:
                selected = remaining.sample(1)
                for _, row in selected.iterrows():
                    warmup_exercises.append(self._format_exercise_from_dataset(row, "warmup"))
        
        return warmup_exercises

    def _generate_cooldown_exercises(self, df: pd.DataFrame, activities_text: str, clinical_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 3 fixed cooldown exercises: static stretches.
        All exercises must be tagged as "Cooldown" in the dataset.
        """
        cooldown_exercises = []
        flags = clinical_context.get("flags", {})
        
        # Filter cooldown exercises from dataset
        cooldown_df = df[df["Tags"].str.contains("cooldown", na=False)].copy()
        
        # Apply safety filters
        safe_cooldown_df = cooldown_df[cooldown_df.apply(lambda row: self._exercise_is_safe_from_row(row, flags), axis=1)]
        
        if safe_cooldown_df.empty:
            safe_cooldown_df = cooldown_df  # Fallback if no safe exercises
        
        # Select 3 static stretching exercises
        if not safe_cooldown_df.empty:
            selected = safe_cooldown_df.sample(min(3, len(safe_cooldown_df)))
            for _, row in selected.iterrows():
                cooldown_exercises.append(self._format_exercise_from_dataset(row, "cooldown"))
        
        return cooldown_exercises

    def _generate_day_title(self, day_plan: Dict[str, Any], activities_text: str) -> str:
        """
        Generate day title based on the day's main workout exercises.
        """
        day_focus = self._classify_day_focus(day_plan, activities_text)
        return day_focus["workout_title"]

    def _classify_day_focus(self, day_plan: Dict[str, Any], activities_text: str) -> Dict[str, str]:
        """Infer a user-friendly workout category/title from exercises and schedule text."""
        main_exercises = day_plan.get("main_workout", [])
        if not main_exercises and str(activities_text or "").strip().lower() == "rest":
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
            activity_types.append("hiit")
        if "cardio" in normalized_activity_text:
            activity_types.append("cardio")
        if any(term in normalized_activity_text for term in ["strength", "resistance", "weights"]):
            activity_types.append("strength")
        if "circuit" in normalized_activity_text:
            activity_types.append("circuit")
        if "yoga" in normalized_activity_text:
            activity_types.append("yoga")

        unique_activities = list(dict.fromkeys(activity_types))

        if "hiit" in unique_activities:
            category = "HIIT Circuit"
        elif "cardio" in unique_activities and "strength" not in unique_activities:
            category = "Cardio Day"
        elif "strength" in unique_activities:
            if "upper" in body_regions and "lower" in body_regions:
                category = "Full Body Circuit" if "circuit" in unique_activities else "Full Body Strength"
            elif "upper" in body_regions:
                category = "Upper Body Strength"
            elif "lower" in body_regions:
                category = "Lower Body Strength"
            elif "core" in body_regions:
                category = "Core Strength"
            else:
                category = "Strength Day"
        elif "yoga" in unique_activities:
            category = "Yoga Mobility"
        elif "circuit" in unique_activities:
            category = "Full Body Circuit"
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

    def _section_duration_label(self, exercises: List[Dict[str, Any]]) -> str:
        total_sec = sum(int(ex.get("est_time_sec", 0) or 0) for ex in exercises)
        if total_sec <= 0:
            return "None"
        total_min = max(1, round(total_sec / 60))
        return f"{total_min} min" if total_min == 1 else f"{total_min} mins"

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
        sets = str(row.get("Sets", "1") or "1")
        reps = str(row.get("Reps", "10") or "10")
        benefit = str(row.get("Health benefit", "") or "")
        safety_cue = str(row.get("Safety cue", "") or "")
        steps = self._split_steps(row.get("Steps to perform", ""))
        rest_intervals = str(row.get("Rest intervals", "") or "")
        met_value = float(row.get("MET value", 3.0) or 3.0)
        est_time_sec = self._estimate_exercise_time_sec(row, section, sets)
        estimated_calories = self._estimate_exercise_calories(met_value, est_time_sec)

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
            "intensity_rpe": str(row.get("RPE", "") or ""),
            "rest_intervals": rest_intervals,
            "rest": rest_intervals,
            "rest_time": rest_intervals,
            "met_value": met_value,
            "est_calories": f"Est: {estimated_calories} Cal",
            "estimated_calories": estimated_calories,
            "est_time_sec": est_time_sec,
            "est_time_human": self._humanize_seconds(est_time_sec),
            "planned_sets_count": max(1, int(float(sets))) if str(sets).replace(".", "", 1).isdigit() else 1,
            "planned_total_cal": estimated_calories,
            "is_not_suitable_for": str(row.get("is_not_suitable_for (Medical conditions)", "") or ""),
        }
