from json_repair import repair_json
import os
import json
import logging
import os
from typing import Dict, Any, List
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureAIContentFilterError(Exception):
    """Exception raised when Azure AI content filtering blocks a request."""
    pass


class AzureAIPrescriptionParser:
    def __init__(self):
        self.endpoint = "https://nouriqfriskacc7470931625.cognitiveservices.azure.com/"
        self.api_version = "2024-12-01-preview"
        self.model = "gpt-4.1-mini-Fitness-Model"
        api_key = os.getenv("AZURE_AI_KEY")
        if not api_key:
            raise ValueError("AZURE_AI_KEY not set")
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=self.endpoint, api_version=self.api_version)

    def parse_notes(self, notes_text: str) -> Dict[str, Any]:
        if not notes_text:
            raise ValueError("No notes")
        prompt = self._get_parsing_prompt(notes_text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a clinical fitness expert."}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            # Check if this is a content filter error
            error_str = str(e)
            if "content_filter" in error_str or "content management policy" in error_str:
                raise AzureAIContentFilterError(f"Content filter blocked: {error_str}")
            # Re-raise other exceptions
            raise

    def _get_parsing_prompt(self, notes_text: str) -> str:
        """
        Generate the parsing prompt for Azure AI.
        Uses clinical terminology to avoid content filter triggers.
        """
        return f"""
You are a clinical fitness expert specializing in rehabilitation and therapeutic exercise prescription. You extract structured workout plans from medical notes with precision and safety.

CLINICAL NOTES STRUCTURE:
Doctor's notes typically include:
1. PATIENT INFO/HISTORY: Medical status, compliance, functional limitations
2. EXERCISE SESSION: Exercises performed during the visit
3. PLAN OF ACTION: Prescribed plan for patient to follow (PRIMARY SOURCE)
4. SAMPLE WEEK: Specific weekly schedule to follow

EXTRACTION PRIORITY:
- PLAN OF ACTION is the PRIMARY SOURCE - extract all exercise names, frequency, duration, intensity
- If Plan of Action lacks specific exercises, use Exercise Session as reference
- SAMPLE WEEK defines the exact weekly structure to replicate
- Extract medical conditions and functional limitations from patient info

INPUT NOTES:
{notes_text}

INSTRUCTIONS:
- Analyze the entire clinical note structure
- Extract exercise names (formal clinical names only)
- PLAN OF ACTION takes precedence for all parameters
- Extract frequency (days per week), duration, intensity, and specific activities
- Identify medical conditions, functional limitations, contraindications
- Extract prescribed goals (weight loss, strength, rehabilitation, etc.)
- Extract equipment requirements from the note

REQUIRED OUTPUT FORMAT (valid JSON only):
{{
"days_per_week": 6,
"session_duration": "45-60 minutes",
"goal": "Weight Loss",
"focus": ["cardiovascular", "resistance", "core"],
"medical_conditions": [],
"physical_limitations": [],
"activity_restrictions": [],
"intensity": "moderate",
"prescribed_exercises": [
{{
"name": "Rowing",
"sets": 8,
"reps": "20 seconds",
"frequency": "2x per week",
"type": "HIIT"
}},
{{
"name": "Chest Press",
"sets": 3,
"reps": "10-15",
"equipment": "15 lb dumbbells",
"frequency": "2x per week",
"type": "resistance"
}}
],
"avoid_exercises": [],
"equipment_required": ["dumbbells", "resistance band", "treadmill"],
"target_body_parts": ["Full Body"],
"weekly_schedule": {{
"monday": "Cardio and Weights",
"tuesday": "Interval Training and Mobility",
"wednesday": "Cardio and Weights",
"thursday": "Interval Training and Mobility",
"friday": "Cardio",
"saturday": "Mixed Modality",
"sunday": "Rest"
}},
"session_directives": [],
"session_types": [],
"day_type_rules": [
{{"type": "hiit", "mapped_to": "full_body_cardio"}},
{{"type": "interval", "mapped_to": "full_body_cardio"}},
{{"type": "circuit", "mapped_to": "full_body_cardio"}}
],
"cardio_requirements": {{"frequency": "3x per week", "duration": "45-60 minutes", "activities": ["Walking", "Cycling", "Treadmill"]}},
"resistance_training": {{"frequency": "2x per week", "sets": 3, "reps": "10-15"}},
"hiit_training": {{"frequency": "2x per week", "duration": "30-45 mins"}},
"additional_requirements": {{}}
}}

EXTRACTION RULES:
- days_per_week: Count active days from Sample Week
- session_duration: From Plan of Action
- goal: Primary therapeutic or fitness goal
- focus: Key areas (cardiovascular, resistance, flexibility, core, balance)
- medical_conditions: Chronic conditions mentioned
- physical_limitations: Functional limitations or injuries
- activity_restrictions: Contraindicated movements or activities
- intensity: Based on prescription (low, moderate, high)
- prescribed_exercises: Exercise names only (formal clinical names)
- avoid_exercises: Contraindicated exercises
- equipment_required: Hardware listed or implied
- target_body_parts: Body regions targeted
- weekly_schedule: Daily activity labels
- cardio_requirements, resistance_training, hiit_training: Extract specifications

Return ONLY valid JSON. No explanations or preamble.
"""

    def _validate_parsed_output(self, parsed: Dict[str, Any]) -> None:
        """
        Validate that parsed output has required structure.
        """
        required_fields = [
            "days_per_week", "session_duration", "goal", "focus",
            "medical_conditions", "physical_limitations", "activity_restrictions",
            "intensity", "prescribed_exercises", "avoid_exercises", "target_body_parts",
            "weekly_schedule", "cardio_requirements", "resistance_training", "hiit_training"
        ]

        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")

        # Validate types
        if not isinstance(parsed["days_per_week"], int):
            raise ValueError("days_per_week must be integer")

        if not isinstance(parsed["focus"], list):
            raise ValueError("focus must be list")

        if not isinstance(parsed["prescribed_exercises"], list):
            raise ValueError("prescribed_exercises must be list")

        if not isinstance(parsed["weekly_schedule"], dict):
            raise ValueError("weekly_schedule must be dict")

        if "session_directives" in parsed and not isinstance(parsed["session_directives"], list):
            raise ValueError("session_directives must be list")
        if "session_types" in parsed and not isinstance(parsed["session_types"], list):
            raise ValueError("session_types must be list")
        if "day_type_rules" in parsed and not isinstance(parsed["day_type_rules"], list):
            raise ValueError("day_type_rules must be list")
        if "equipment_required" in parsed and not isinstance(parsed["equipment_required"], list):
            raise ValueError("equipment_required must be list")

        # Validate prescribed exercises structure
        for ex in parsed["prescribed_exercises"]:
            if not isinstance(ex, dict) or "name" not in ex:
                raise ValueError("Each prescribed exercise must have 'name' field")

    def extract_high_level_goals(self, notes_text: str) -> Dict[str, Any]:
        """
        Extract high-level program structure without triggering body-related filters.
        Focus: Numbers, days, and general goals.
        
        Returns:
            Dictionary with days_per_week, goal, weekly_schedule, and equipment_required
        """
        if not notes_text:
            raise ValueError("No notes provided")
        
        prompt = f"""You are a clinical data analyst. Extract the high-level program structure from the provided notes into a JSON object.

Focus strictly on:

days_per_week: (integer)

goal: (e.g., 'Weight Loss', 'Strength')

weekly_schedule: (Map days to high-level labels like 'Cardio', 'Rest', or 'Strength')

equipment_required: (List of hardware like 'Dumbbells', 'Treadmill')

Constraint: Do not extract exercise descriptions or functional cues. Return ONLY a valid JSON object.

NOTES:
{notes_text}

REQUIRED OUTPUT FORMAT:
{{
"days_per_week": 5,
"goal": "Weight Loss",
"weekly_schedule": {{
"monday": "Cardio",
"tuesday": "Strength",
"wednesday": "Cardio",
"thursday": "Strength",
"friday": "Cardio",
"saturday": "Rest",
"sunday": "Rest"
}},
"equipment_required": ["Dumbbells", "Treadmill"]
}}

Return ONLY valid JSON. No explanations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical data analyst extracting program structure."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Validate required fields
            required = ["days_per_week", "goal", "weekly_schedule", "equipment_required"]
            for field in required:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            return parsed
        except Exception as e:
            error_str = str(e)
            if "content_filter" in error_str or "content management policy" in error_str:
                raise AzureAIContentFilterError(f"Content filter blocked high-level goals extraction: {error_str}")
            raise

    def extract_clinical_safety(self, notes_text: str) -> Dict[str, Any]:
        """
        Identify medical flags and exclusions.
        Focus: Medical terms and "Avoid" instructions.
        
        Returns:
            Dictionary with medical_conditions, physical_limitations, activity_restrictions, avoid_exercises
        """
        if not notes_text:
            raise ValueError("No notes provided")
        
        prompt = f"""You are a medical safety officer. Analyze these fitness notes to identify safety constraints.

Extract into JSON:

medical_conditions: (List)

physical_limitations: (List, e.g., 'Knee pain')

activity_restrictions: (List of movements to avoid, e.g., 'No jumping')

avoid_exercises: (List of specific exercise names mentioned as contraindicated)

Constraint: Maintain a clinical tone. Return ONLY a valid JSON object.

NOTES:
{notes_text}

REQUIRED OUTPUT FORMAT:
{{
"medical_conditions": ["Hypertension", "Obesity"],
"physical_limitations": ["Knee pain", "Low back pain"],
"activity_restrictions": ["No jumping", "No high impact"],
"avoid_exercises": ["Burpees", "Box jumps"]
}}

Return ONLY valid JSON. No explanations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical safety officer extracting safety constraints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Validate required fields
            required = ["medical_conditions", "physical_limitations", "activity_restrictions", "avoid_exercises"]
            for field in required:
                if field not in parsed:
                    raise ValueError(f"Missing required field: {field}")
            
            return parsed
        except Exception as e:
            error_str = str(e)
            if "content_filter" in error_str or "content management policy" in error_str:
                raise AzureAIContentFilterError(f"Content filter blocked clinical safety extraction: {error_str}")
            raise

    def extract_exercise_repertoire(self, notes_text: str) -> Dict[str, Any]:
        """
        Extract the specific exercises while avoiding body-related filter triggers.
        Focus: Nouns only. By stripping "vivid" descriptions, you avoid the "Sexual/Medium Severity" flag.
        
        Returns:
            Dictionary with prescribed_exercises list containing name, sets, reps, and type
        """
        if not notes_text:
            raise ValueError("No notes provided")
        
        prompt = f"""You are a kinesiology expert. Extract only the names and parameters of prescribed exercises from these notes.

Safety Rule: To comply with safety filters, extract ONLY the formal exercise names (e.g., 'Pilates Hundred', 'Bird Dog'). Do NOT include functional descriptions or vivid movement instructions.

Extract into JSON:

prescribed_exercises: List of objects with name, sets, reps, and type.

Constraint: If an exercise description is too descriptive, simplify it to the common exercise name. Return ONLY a valid JSON object.

NOTES:
{notes_text}

REQUIRED OUTPUT FORMAT:
{{
"prescribed_exercises": [
{{
"name": "Pilates Hundred",
"sets": 3,
"reps": "100",
"type": "core"
}},
{{
"name": "Bird Dog",
"sets": 3,
"reps": "12 per side",
"type": "stability"
}},
{{
"name": "Chest Press",
"sets": 3,
"reps": "10-12",
"type": "resistance"
}}
]
}}

Return ONLY valid JSON. No explanations."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a kinesiology expert extracting exercise names using clinical terminology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # Validate required fields
            if "prescribed_exercises" not in parsed:
                raise ValueError("Missing required field: prescribed_exercises")
            
            if not isinstance(parsed["prescribed_exercises"], list):
                raise ValueError("prescribed_exercises must be a list")
            
            # Validate each exercise has a name
            for ex in parsed["prescribed_exercises"]:
                if not isinstance(ex, dict) or "name" not in ex:
                    raise ValueError("Each exercise must have a 'name' field")
            
            return parsed
        except Exception as e:
            error_str = str(e)
            if "content_filter" in error_str or "content management policy" in error_str:
                raise AzureAIContentFilterError(f"Content filter blocked exercise repertoire extraction: {error_str}")
            raise
