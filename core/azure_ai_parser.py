import os
import json
import logging
import os
from typing import Dict, Any, List
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a clinical fitness expert."}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _get_parsing_prompt(self, notes_text: str) -> str:
        """
        Generate the parsing prompt for Azure AI.
        """
        return f"""
You are a clinical fitness expert specializing in rehabilitation and therapeutic exercise prescription. You extract structured workout plans from doctor's notes with precision and safety.

DOCTOR'S NOTES STRUCTURE ANALYSIS:
Doctor's notes typically follow this structure:
1. PATIENT INFO/HISTORY: Patient reports, current status, compliance, limitations
2. EXERCISE SESSION: Exercises performed during the visit (warm-up, main workout)
3. PLAN OF ACTION: Prescribed plan for patient to follow (PRIMARY SOURCE)
4. SAMPLE WEEK: Specific weekly schedule to follow

EXTRACTION PRIORITY:
- PLAN OF ACTION is the MOST IMPORTANT - follow this strictly for exercises, frequency, duration
- If Plan of Action lacks specific exercises, use Exercise Session exercises as reference
- SAMPLE WEEK provides the exact schedule structure to replicate
- Extract limitations/conditions from patient info paragraphs

INPUT NOTES:
{notes_text}

INSTRUCTIONS:
- Analyze the entire note structure carefully
- PLAN OF ACTION takes precedence over all other sections
- SAMPLE WEEK defines the exact weekly structure
- Extract specific exercises mentioned in Plan of Action
- If no exercises in Plan of Action, use Exercise Session as reference for similar exercises
- Look for frequency (days per week), duration, intensity, and specific activities
- Identify any restrictions, limitations, or contraindications
- Extract goals (weight loss, strength, flexibility, etc.)

REQUIRED OUTPUT FORMAT (strict JSON):
{{
"days_per_week": 6,
"session_duration": "45-60 minutes",
"goal": "Weight Loss",
"focus": ["cardiovascular fitness", "resistance training", "core strength"],

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
"monday": "Cardio/weights",
"tuesday": "HIIT/Interval 30-45 mins/Yoga",
"wednesday": "Cardio/weights",
"thursday": "HIIT/Interval 30-45 mins/Yoga",
"friday": "Cardio",
"saturday": "Cardio/weight or HIIT/Interval 30-45 mins",
"sunday": "Rest or bike ride/brisk walk"
}},

"session_directives": [
{{
"day": "tuesday",
"session_type": "yoga",
"label": "Join Yoga Session",
"duration_min": 30,
"instruction": "Join yoga session",
"source_text": "Tuesday yoga class 6:30pm-7:00pm"
}},
{{
"day": "friday",
"session_type": "cardio",
"label": "Cardio Session",
"duration_min": 30,
"instruction": "Brisk walk or treadmill cardio",
"source_text": "Cardio 30 minutes"
}}
],

"session_types": [
{{
"type": "yoga",
"mode": "class",
"label": "Join Yoga Session",
"duration": 30
}},
{{
"type": "cardio",
"mode": "session",
"label": "Cardio Session",
"duration": 30
}}
],

"day_type_rules": [
{{
"type": "hiit",
"mapped_to": "full_body_cardio"
}},
{{
"type": "interval",
"mapped_to": "full_body_cardio"
}},
{{
"type": "circuit",
"mapped_to": "full_body_cardio"
}}
],

"cardio_requirements": {{
"frequency": "3x per week",
"duration": "45-60 minutes",
"target_calories": "400-650",
"activities": ["Bike", "Walking", "Elliptical", "Treadmill", "Aerobic classes"]
}},

"resistance_training": {{
"frequency": "2x per week",
"equipment": "15-20 lb dumbbells",
"sets": 3,
"reps": "10-15",
"exercises": ["Chest press", "Rows", "Shoulder press", "Bicep curls", "Tricep extension", "Squat"]
}},

"hiit_training": {{
"frequency": "2x per week",
"duration": "30-45 mins",
"structure": "8 exercises/8 rounds 20 seconds on/10 seconds off",
"target_calories": "300-600"
}},

"additional_requirements": {{
"steps_per_day": 10000,
"yoga_classes": ["Tuesday 6:30pm-7:00pm", "Thursday 6:30pm-7:00pm"],
"flexibility_work": "Use stretch out strap at home",
"monitoring": ["Resting heart rate", "Blood pressure", "Nutrition logging", "Calories and steps tracking"]
}}
}}

EXTRACTION RULES:
- days_per_week: Count from Sample Week (exclude rest days if specified)
- session_duration: From Plan of Action cardio/resistance/HIIT durations
- goal: Primary goal (Weight Loss, Strength, Rehabilitation, etc.)
- focus: Key areas emphasized (cardio, resistance, flexibility, etc.)
- medical_conditions: Chronic conditions mentioned
- physical_limitations: Injuries, pain, restrictions mentioned
- activity_restrictions: Avoidances, contraindications
- intensity: Based on prescribed loads/durations
- prescribed_exercises: ALL exercises mentioned in Plan of Action or Exercise Session
- avoid_exercises: Any contraindicated exercises
- equipment_required: Extract all available or prescribed equipment from the note such as dumbbells, resistance band, treadmill, bike, elliptical.
- target_body_parts: Body areas targeted
- weekly_schedule: Copy Sample Week exactly
- session_directives: Extract explicit daily sessions/classes from the note such as yoga session, yoga class, pilates class, cardio session, HIIT/interval/circuit block. Preserve join/class wording when present.
- session_types: Extract reusable session definitions. Preserve the exact coach wording in label.
- day_type_rules: Extract mapping rules such as HIIT/Interval/Circuit -> full_body_cardio.
- cardio_requirements: Extract cardio specifications
- resistance_training: Extract weight training details
- hiit_training: Extract HIIT/interval specifications
- additional_requirements: Other instructions (steps, yoga, monitoring, etc.)

Return ONLY valid JSON. No explanations or additional text.
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
