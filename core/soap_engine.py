from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    pdfplumber = None

from core.fitness_engine import generate_plan_local
from core.models import SoapParseResult
from services.dataset_service import DatasetService
from utils.constants import DAY_ORDER
from utils.text_normalizer import normalize_text


class SoapEngine:
    def __init__(self) -> None:
        self.fitness_df = DatasetService.load_fitness_dataset()

    def extract_pdf_text(self, pdf_file: Any) -> str:
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Use pasted text or install requirements.txt.")
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        return full_text

    def parse_text(self, text: str) -> SoapParseResult:
        source = str(text or "").strip()
        history = self._grab(source, "History of present illness", "Plan of Action|Functional & Diet Findings")
        plan_of_action = self._grab(source, "Plan of Action", "Smart Goals|Summary|Exercise Session")
        findings = self._grab(source, "Functional & Diet Findings", "Exercise Session|Plan of Action|Vitals")
        exercise_session = self._grab(source, "Exercise Session", "Personalized Meal Plan|Vitals|Tasks and Time Tracking")
        vitals = self._extract_vitals(source)
        restrictions = self._extract_restrictions(source)
        prescribed = self._extract_exercise_mentions(source)
        inferred_profile = self._build_profile(source, vitals, restrictions, prescribed)
        frequency = self._extract_frequency(source)

        return SoapParseResult(
            source_text=source,
            history=history,
            findings=findings,
            plan_of_action=plan_of_action,
            exercise_session=exercise_session,
            structured_vitals=vitals,
            inferred_profile=inferred_profile,
            prescribed_exercises=prescribed,
            restrictions=restrictions,
            frequency_per_week=frequency,
        )

    def generate_plan_from_text(self, text: str) -> Dict[str, Any]:
        parsed = self.parse_text(text)
        profile = dict(parsed.inferred_profile)
        profile["prescribed_exercises"] = parsed.prescribed_exercises
        profile["restrictions"] = parsed.restrictions
        if parsed.frequency_per_week:
            profile["weekly_days"] = parsed.frequency_per_week
        plan = generate_plan_local(profile)
        return {"parsed": parsed.to_dict(), "plan": plan, "profile": profile}

    def _grab(self, text: str, start_key: str, end_key: str) -> str:
        pattern = rf"{start_key}(.*?)(?={end_key})"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_vitals(self, text: str) -> Dict[str, str]:
        vitals = {"Weight": "N/A", "BP": "N/A", "BMI": "N/A"}
        patterns = {
            "Weight": [
                r"weight[:\s]+([\d\.]+\s*(?:kg|kgs|lb|lbs)?)",
                r'"Weight\s*\\n?"\s*,\s*"([\d\.]+\s*lbs?)',
            ],
            "BP": [
                r"(?:bp|blood pressure)[:\s]+([\d]{2,3}/[\d]{2,3}\s*(?:mmhg)?)",
                r'"BP\s*\\n?"\s*,\s*"([\d/]+\s*mmHg)',
            ],
            "BMI": [
                r"bmi[:\s]+([\d\.]+)",
                r'"BMI\s*\\n?"\s*,\s*"([\d\.]+)',
            ],
        }
        for key, regexes in patterns.items():
            for regex in regexes:
                match = re.search(regex, text, re.IGNORECASE | re.DOTALL)
                if match:
                    vitals[key] = match.group(1).strip()
                    break
        return vitals

    def _extract_restrictions(self, text: str) -> List[str]:
        text_key = normalize_text(text)
        restriction_map = {
            "knee pain": ["knee pain", "acl", "meniscus", "patella"],
            "low back pain": ["low back", "lumbar", "back pain", "sciatica"],
            "shoulder pain": ["shoulder pain", "rotator cuff", "impingement"],
            "avoid floor work": ["avoid floor", "cannot lie down", "difficulty getting up from floor"],
            "diabetes": ["diabetes", "blood sugar"],
            "hypertension": ["hypertension", "high blood pressure"],
        }
        restrictions: List[str] = []
        for label, cues in restriction_map.items():
            if any(cue in text_key for cue in cues):
                restrictions.append(label)
        return restrictions

    def _extract_exercise_mentions(self, text: str) -> List[str]:
        text_key = normalize_text(text)
        found: List[str] = []
        for name in self.fitness_df["exercise_name"].tolist():
            normalized = normalize_text(name)
            if normalized and normalized in text_key:
                found.append(str(name))
        return found[:8]

    def _build_profile(
        self,
        text: str,
        vitals: Dict[str, str],
        restrictions: List[str],
        prescribed: List[str],
    ) -> Dict[str, Any]:
        goal = self._infer_goal(text)
        days = self._extract_days(text)
        age_match = re.search(r"\b(age|aged)\s*[:\-]?\s*(\d{1,2})\b", text, re.IGNORECASE)
        age = int(age_match.group(2)) if age_match else 49
        weight_kg = self._weight_to_kg(vitals.get("Weight", "70"))
        return {
            "age": age,
            "weight_kg": weight_kg,
            "goal": goal,
            "primary_goal": goal,
            "fitness_level": self._infer_fitness_level(text),
            "body_region": self._infer_body_region(text),
            "session_duration": self._infer_duration(text),
            "weekly_days": len(days) if days else 3,
            "days": days or DAY_ORDER[:3],
            "equipment": self._infer_equipment(text),
            "location": "Home",
            "restrictions": restrictions,
            "prescribed_exercises": prescribed,
            "blood_pressure": vitals.get("BP", "N/A"),
        }

    def _extract_frequency(self, text: str) -> Optional[int]:
        match = re.search(r"(\d)\s*(?:days|x)\s*(?:per)?\s*week", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def _extract_days(self, text: str) -> List[str]:
        found: List[str] = []
        for day in DAY_ORDER:
            if re.search(rf"\b{day}\b", text, re.IGNORECASE):
                found.append(day)
        return found

    def _infer_goal(self, text: str) -> str:
        key = normalize_text(text)
        if any(token in key for token in ["weight loss", "fat loss", "diabetes"]):
            return "Weight Loss"
        if any(token in key for token in ["strength", "muscle", "resistance"]):
            return "Muscle Gain"
        if any(token in key for token in ["mobility", "flexibility", "range of motion"]):
            return "Mobility"
        return "Weight Maintenance"

    def _infer_fitness_level(self, text: str) -> str:
        key = normalize_text(text)
        if "advanced" in key:
            return "Advanced"
        if "intermediate" in key:
            return "Intermediate"
        return "Beginner"

    def _infer_body_region(self, text: str) -> str:
        key = normalize_text(text)
        if "upper" in key or "shoulder" in key or "neck" in key:
            return "Upper Body"
        if "lower" in key or "knee" in key or "hip" in key:
            return "Lower Body"
        if "core" in key or "abdominal" in key:
            return "Core"
        return "Full Body"

    def _infer_duration(self, text: str) -> str:
        match = re.search(r"(\d{1,2})\s*(?:minutes|min)", text, re.IGNORECASE)
        if match:
            return f"{match.group(1)} min"
        return "30 min"

    def _infer_equipment(self, text: str) -> List[str]:
        key = normalize_text(text)
        found: List[str] = []
        equipment_terms = [
            "dumbbells",
            "resistance bands",
            "kettlebell",
            "bench",
            "foam roller",
        ]
        for term in equipment_terms:
            if term in key:
                found.append(term.title())
        return found or ["No Equipment"]

    def _weight_to_kg(self, raw_value: str) -> float:
        raw = str(raw_value or "").strip().lower()
        match = re.search(r"([\d\.]+)", raw)
        if not match:
            return 70.0
        value = float(match.group(1))
        if "lb" in raw:
            value *= 0.45359237
        return round(value, 1)
