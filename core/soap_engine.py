from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    pdfplumber = None

from core.fitness import ClinicalExtractionTool, PrescriptionParserTool
from core.fitness_engine import generate_plan_local
from core.models import SoapParseResult
from services.dataset_service import DatasetService
from utils.constants import DAY_ORDER
from utils.text_normalizer import normalize_text

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


class SoapEngine:
    USE_PARSER_TOOL = False  # Set to True to enable PrescriptionParserTool

    def __init__(self) -> None:
        self.fitness_df = DatasetService.load_fitness_dataset()
        self.soap_df = DatasetService.load_soap_dataset()
        self.clinical_tool = ClinicalExtractionTool()
        self.parser_tool = PrescriptionParserTool()

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
        extracted = self.clinical_tool.parse_sections(source)
        vitals = extracted.get("structured_vitals", {})
        restrictions = self._extract_restrictions(source)
        prescribed = self._extract_exercise_mentions(source)
        frequency = self._extract_frequency(source)
        inferred_profile = self._build_profile(source, vitals, restrictions, prescribed, frequency)

        return SoapParseResult(
            source_text=source,
            history=extracted.get("history", ""),
            findings=extracted.get("findings", ""),
            plan_of_action=extracted.get("plan_of_action", ""),
            exercise_session=extracted.get("exercise_session", ""),
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

        # Ensure full days are used, no collapsing
        if not profile.get("days"):
            profile["days"] = DAY_ORDER[:3]
            profile["weekly_days"] = 3
            profile["structured_days"] = [{"day_index": idx, "day_name": day} for idx, day in enumerate(profile["days"])]

        plan = self._build_plan(parsed, profile)
        return {"parsed": parsed.to_dict(), "plan": plan, "profile": profile}

    def _build_plan(self, parsed: SoapParseResult, profile: Dict[str, Any]) -> Dict[str, Any]:
        expected_days = profile.get("weekly_days", 3)
        logger.info(f"[SoapEngine] Expected days: {expected_days}, Profile days: {profile.get('days', [])}")

        # Force full plan mode
        profile["is_minimal_plan"] = False
        profile["structured_mode"] = True
        profile["force_full_week"] = True

        # Ensure correct day iteration
        profile["structured_days"] = [
            {
                "day_index": i,
                "day_name": day,
                "day_type": profile.get("day_focus_map", {}).get(day, "General")
            }
            for i, day in enumerate(profile["days"])
        ]

        # Ensure the build process is resilient to invalid profiles
        if profile.get("weekly_days", 0) <= 0 or not profile.get("days"):
            logger.warning("Profile weekly_days invalid or missing days, enforcing safe fallback.")
            profile["weekly_days"] = 3
            profile["days"] = DAY_ORDER[:3]
            profile["structured_days"] = [
                {"day_index": i, "day_name": day, "day_type": "General"}
                for i, day in enumerate(profile["days"])
            ]

        # Debug logs
        logger.info(f"FINAL PROFILE DAYS: {profile['days']}")
        logger.info(f"FINAL WEEKLY DAYS USED: {profile['weekly_days']}")
        logger.info(f"STRUCTURED DAYS: {profile['structured_days']}")

        if self.USE_PARSER_TOOL:
            prescription_narrative = self._build_prescription_narrative(parsed)
            try:
                result = self._run_async(self.parser_tool.execute(prescription_narrative, profile))
                if result and result.success:
                    plans_json = result.data.get("plans_json") or result.data.get("json_plan") or {}
                    generated_days = len(plans_json)
                    logger.info(f"[SoapEngine] Parser generated {generated_days} days")
                    if generated_days == expected_days and generated_days >= 2:
                        logger.info("[SoapEngine] Parser output valid, using it")
                        return plans_json
                    else:
                        logger.warning(f"[SoapEngine] Parser output invalid ({generated_days} days), falling back to core engine")
                else:
                    logger.warning("[SoapEngine] Parser failed, falling back to core engine")
            except Exception as exc:
                logger.error(f"[SoapEngine] Parser exception: {exc}, falling back to core engine")

        logger.info("[SoapEngine] Using core engine for plan generation")
        return generate_plan_local(profile)

    def _build_prescription_narrative(self, parsed: SoapParseResult) -> str:
        narrative_parts: List[str] = []
        if parsed.history:
            narrative_parts.append(f"History: {parsed.history}")
        if parsed.findings:
            narrative_parts.append(f"Findings: {parsed.findings}")
        if parsed.plan_of_action:
            narrative_parts.append(f"Plan: {parsed.plan_of_action}")
        if parsed.exercise_session:
            narrative_parts.append(f"Session: {parsed.exercise_session}")
        if parsed.structured_vitals:
            narrative_parts.append(f"Vitals: {parsed.structured_vitals}")
        return "\n".join(narrative_parts)

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

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
        source_df = self.soap_df if not self.soap_df.empty else self.fitness_df
        for name in source_df["exercise_name"].tolist():
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
        frequency: Optional[int] = None,
    ) -> Dict[str, Any]:
        goal = self._infer_goal(text)
        week_structure = self._build_week_structure(text)
        weekly_days = len(week_structure)
        days = [d["day_name"] for d in week_structure]
        day_focus_map = {d["day_name"]: d["focus"] for d in week_structure}
        structured_days = [{"day_index": idx, "day_name": day} for idx, day in enumerate(days)]

        # Hard fallback
        if not days:
            weekly_days = 3
            days = DAY_ORDER[:3]
            week_structure = [{"day_name": d, "focus": "General"} for d in days]
            day_focus_map = {d: "General" for d in days}
            structured_days = [{"day_index": idx, "day_name": day} for idx, day in enumerate(days)]

        logger.info(f"Weekly days: {weekly_days}")

        age_match = re.search(r"\b(age|aged)\s*[:\-]?\s*(\d{1,2})\b", text, re.IGNORECASE)
        age = int(age_match.group(2)) if age_match else 49
        weight_kg = self._weight_to_kg(vitals.get("Weight", "70"))

        profile: Dict[str, Any] = {
            "age": age,
            "weight_kg": weight_kg,
            "goal": goal,
            "primary_goal": goal,
            "fitness_level": self._infer_fitness_level(text),
            "body_region": self._infer_body_region(text),
            "session_duration": self._infer_duration(text),
            "weekly_days": weekly_days,
            "days": days,
            "structured_days": structured_days,
            "profile_days": week_structure,
            "day_focus_map": day_focus_map,
            "equipment": self._infer_equipment(text),
            "location": "Home",
            "restrictions": restrictions,
            "prescribed_exercises": prescribed,
            "blood_pressure": vitals.get("BP", "N/A"),
        }
        return profile

    def _extract_frequency(self, text: str) -> Optional[int]:
        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5
        }
        text_lower = text.lower()
        
        # Numeric pattern
        numeric_match = re.search(r"(\d+)\s*(?:days?|times?|x)\s*(?:per\s*)?week", text_lower)
        if numeric_match:
            return int(numeric_match.group(1))
        
        # Word-based pattern
        word_match = re.search(r"(one|two|three|four|five)\s*(?:days?|times?)\s*(?:per\s*)?week", text_lower)
        if word_match:
            word = word_match.group(1)
            return word_to_num.get(word)
        
        return None

    def _extract_days(self, text: str) -> List[str]:
        result: List[str] = []
        seen: set[str] = set()
        day_pattern = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
        for match in re.finditer(day_pattern, text, re.IGNORECASE):
            day_name = match.group(1).title()
            if day_name not in seen:
                seen.add(day_name)
                result.append(day_name)
        return result

    def _detect_schedule_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(day.lower() in text_lower for day in DAY_ORDER):
            return "explicit_days"
        if re.search(r"(?:\d+|one|two|three|four|five)\s*(?:times?|days?)\s*(?:per\s*)?week", text_lower):
            return "frequency_based"
        if re.search(r"(?:\d+|one|two|three|four|five)\s*day.*(?:tabata|cardio|strength|mobility|flexibility|recovery|core|upper|lower|full)", text_lower, re.IGNORECASE):
            return "split_based"
        return "default"

    def _has_split_indicator(self, text: str) -> bool:
        text_lower = text.lower()
        return bool(re.search(r"(tabata|cardio|strength|mobility|flexibility|recovery|core|upper body|lower body|full body|interval|circuit)", text_lower))

    def _is_weak_week_structure(self, text: str, week_structure: List[Dict[str, str]]) -> bool:
        if len(week_structure) <= 1:
            return True
        if all(str(item.get("focus", "General")).strip().lower() == "general" for item in week_structure):
            return True
        if not self._has_split_indicator(text):
            return True
        return False

    def _extract_json_text(self, raw_text: str) -> str:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        return match.group(0) if match else ""

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            return {}

    def _extract_response_text(self, response: Any) -> str:
        try:
            # Azure / OpenAI ChatCompletion format
            if hasattr(response, "choices"):
                return response.choices[0].message.content.strip()

            # Fallback
            return str(response)
        except Exception:
            return ""

    def _openai_client(self) -> Optional[Any]:
        if getattr(self, "_ai_client", None) is not None:
            return self._ai_client
        try:
            from openai import AzureOpenAI
            from dotenv import load_dotenv
            load_dotenv()
            logger = logging.getLogger(__name__)
            api_key = os.getenv("AZURE_AI_KEY")
            endpoint = "https://nouriqfriskacc7470931625.cognitiveservices.azure.com/"
            api_version = "2024-12-01-preview"
            if not api_key:
                logger.warning("OpenAI API key is not configured; AI schedule extraction disabled.")
                self._ai_client = None
            else:
                self._ai_client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
        except Exception as exc:
            logger.warning("OpenAI client init failed: %s — AI schedule extraction disabled.", exc)
            self._ai_client = None
        return self._ai_client

    def _ai_extract_schedule(self, text: str) -> Dict[str, Any]:
        client = self._openai_client()
        if client is None:
            return {"weekly_days": 0, "plan": []}

        prompt = f"""
You are a clinical exercise expert.

Your task is to fully interpret the workout prescription.

IMPORTANT:
- ALWAYS calculate TOTAL weekly days
- NEVER ignore context like "other days", "remaining days"
- Combine all instructions into ONE complete plan
- If "other X days" appears → it refers to the remaining days in the week

---

EXAMPLE:

Input:
"Resistance training 3 times per week, Tabata 1 day per week, other two days strength"

Interpretation:
- Total days in week: 3
- Resistance/Strength: 3 times = 2 days (the "other two days")
- Tabata/Cardio: 1 day
- Total: 3 days ✓

Output:
{{
  "weekly_days": 3,
  "plan": [
    {{"focus": "Strength", "count": 2}},
    {{"focus": "Cardio", "subtype": "Tabata", "count": 1}}
  ]
}}

---

RULES:
- Extract ALL frequency mentions
- If total frequency is explicitly given → it is the source of truth
- "Other X days" means: remaining days = total - (explicitly mentioned days)
- Sum of counts MUST equal weekly_days
- NEVER return partial plan
- If multiple modalities mentioned, distribute among them

---

Clinical note:
{text}

Return ONLY valid JSON (no markdown, no explanation).
"""
        try:
            deployment = "gpt-4.1-mini-Fitness-Model"
            response = client.chat.completions.create(model=deployment, messages=[{"role": "user", "content": prompt}], max_tokens=250)
            raw = self._extract_response_text(response)
            logger.info(f"RAW AI RESPONSE: {raw}")
            logger.info(f"OUTPUT TEXT: {getattr(response, 'output_text', None)}")
            json_text = self._extract_json_text(raw)
            logger.info(f"CLEAN AI TEXT: {json_text}")
            parsed = self._safe_json_parse(json_text)
            if not parsed:
                logger.warning("AI returned invalid JSON, fallback triggered")
                return {"weekly_days": 0, "plan": []}
            logger.info(f"PARSED AI PLAN: {parsed}")

            weekly_days = int(parsed.get("weekly_days") or 0)
            plan_items: List[Dict[str, Any]] = []
            for item in parsed.get("plan", []) or []:
                if not isinstance(item, dict):
                    continue
                focus = str(item.get("focus", "General")).strip().title()
                subtype = str(item.get("subtype", "")).strip().title()
                try:
                    count = int(item.get("count", 0))
                except Exception:
                    continue
                if focus and count > 0:
                    plan_item = {"focus": focus, "count": count}
                    if subtype:
                        plan_item["subtype"] = subtype
                    plan_items.append(plan_item)

            # Validation: Check if AI returned too few days
            if weekly_days < 2:
                logger.warning("AI returned too few days (%d), correcting from text frequency", weekly_days)
                freq = self._extract_frequency(text)
                if freq:
                    weekly_days = freq
                    logger.info("Corrected weekly_days to %d from text frequency", weekly_days)

            # Hard correction: if AI output is incomplete, rebuild plan
            freq = self._extract_frequency(text) or weekly_days

            # Detect cardio days from current plan
            cardio_days = 0
            for item in plan_items:
                if item["focus"].lower() == "cardio":
                    cardio_days += item["count"]

            # If AI output is incomplete → rebuild plan
            if len(plan_items) <= 1 or sum(i["count"] for i in plan_items) < freq:
                logger.warning("AI incomplete → rebuilding plan from text")

                if cardio_days > 0:
                    strength_days = freq - cardio_days
                else:
                    strength_days = freq

                plan_items = []

                if strength_days > 0:
                    plan_items.append({"focus": "Strength", "count": strength_days})

                if cardio_days > 0:
                    plan_items.append({"focus": "Cardio", "count": cardio_days})

                weekly_days = freq

            # Final validation
            total = sum(item["count"] for item in plan_items)

            if total != weekly_days:
                logger.warning("Final mismatch → fixing: total=%s weekly_days=%s", total, weekly_days)

                if total < weekly_days:
                    plan_items.append({
                        "focus": "Strength",
                        "count": weekly_days - total
                    })

            logger.info(f"AI weekly_days: {weekly_days}")
            logger.info(f"AI plan: {plan_items}")
            logger.info(f"FINAL AI STRUCTURE: {{'weekly_days': {weekly_days}, 'plan': {plan_items}}}")
            return {"weekly_days": weekly_days, "plan": plan_items}
        except Exception as exc:
            logger.warning("AI schedule extraction failed: %s", exc)
            return {"weekly_days": 0, "plan": []}

    def _convert_ai_to_week_structure(self, ai_output: Dict[str, Any]) -> List[Dict[str, str]]:
        weekly_days = ai_output.get("weekly_days") or 0
        plan = ai_output.get("plan") or []
        if not isinstance(weekly_days, int) or weekly_days <= 0 or not isinstance(plan, list):
            return []

        week_structure: List[Dict[str, str]] = []
        max_days = min(weekly_days, len(DAY_ORDER))

        for item in plan:
            focus = str(item.get("focus", "General")).strip().title()
            try:
                count = int(item.get("count", 0))
            except Exception:
                count = 0
            for _ in range(count):
                if len(week_structure) >= max_days:
                    break
                week_structure.append({"day_name": DAY_ORDER[len(week_structure)], "focus": focus or "General"})
            if len(week_structure) >= max_days:
                break

        while len(week_structure) < max_days:
            week_structure.append({"day_name": DAY_ORDER[len(week_structure)], "focus": "General"})

        return week_structure

    def _build_week_structure(self, text: str) -> List[Dict[str, str]]:
        ai_output = self._ai_extract_schedule(text)
        logger.info(f"AI Output: {ai_output}")
        if ai_output.get("weekly_days", 0) >= 2:
            ai_week_structure = self._convert_ai_to_week_structure(ai_output)
            if not ai_week_structure:
                ai_week_structure = [
                    {"day_name": DAY_ORDER[i], "focus": "General"}
                    for i in range(min(ai_output.get("weekly_days", 0), len(DAY_ORDER)))
                ]
            logger.info("Using AI-derived week structure")
            logger.info(f"Week structure: {ai_week_structure}")
            return ai_week_structure

        schedule_type = self._detect_schedule_type(text)
        logger.info(f"Detected schedule type: {schedule_type}")

        week_structure: List[Dict[str, str]] = []
        if schedule_type == "explicit_days":
            days = self._extract_days(text)
            week_structure = [{"day_name": day, "focus": "General"} for day in days]
        elif schedule_type == "split_based":
            week_structure = self._parse_split_prescription(text)
        elif schedule_type == "frequency_based":
            frequency = self._extract_frequency(text) or 0
            if frequency > 0:
                week_structure = [{"day_name": DAY_ORDER[i], "focus": "General"} for i in range(min(frequency, len(DAY_ORDER)))]
        else:
            week_structure = []

        logger.info(f"Week structure: {week_structure}")
        return week_structure

    def _parse_split_prescription(self, text: str) -> List[Dict[str, str]]:
        text_lower = text.lower()
        splits: List[Dict[str, str]] = []
        day_index = 0

        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

        # Pattern A: "1 day tabata"
        split_pattern_a = r"(\d+)\s*day(?:s)?\s*(tabata|cardio|strength|mobility|flexibility|recovery|core|upper\s*body|lower\s*body|full\s*body|general)"
        matches_a = re.findall(split_pattern_a, text_lower, re.IGNORECASE)

        # Pattern B: "tabata workout 1 day per week"
        split_pattern_b = r"(tabata|cardio|strength|mobility|flexibility|recovery|core|upper\s*body|lower\s*body|full\s*body|general).*?(\d+|one|two|three|four|five)\s*day"
        matches_b = re.findall(split_pattern_b, text_lower, re.IGNORECASE)

        all_matches = matches_a + matches_b

        for match in all_matches:
            if len(match) == 2:
                count_str, focus = match
                if count_str.isdigit():
                    count = int(count_str)
                else:
                    count = word_to_num.get(count_str, 1)
                for _ in range(count):
                    if day_index < len(DAY_ORDER):
                        splits.append({"day_name": DAY_ORDER[day_index], "focus": focus.title()})
                        day_index += 1

        # If no splits found, fallback to frequency
        if not splits:
            frequency = self._extract_frequency(text) or 0
            if frequency > 0:
                splits = [{"day_name": DAY_ORDER[i], "focus": "General"} for i in range(min(frequency, len(DAY_ORDER)))]

        return splits

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
