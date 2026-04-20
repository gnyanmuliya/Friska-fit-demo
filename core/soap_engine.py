from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    pdfplumber = None

from core.fitness import ClinicalExtractionTool, PrescriptionParserTool, FitnessPlanGeneratorTool
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

        if profile.get("weekly_days", 0) <= 0 or not profile.get("days"):
            logger.warning("Profile weekly_days invalid or missing days, enforcing safe fallback.")
            profile["weekly_days"] = 3
            profile["days"] = DAY_ORDER[:3]
            profile["structured_days"] = [
                {"day_index": i, "day_name": day, "day_type": "General"}
                for i, day in enumerate(profile["days"])
            ]

        logger.info(f"FINAL PROFILE DAYS: {profile['days']}")
        logger.info(f"FINAL WEEKLY DAYS USED: {profile['weekly_days']}")
        logger.info(f"STRUCTURED DAYS: {profile.get('structured_days', [])}")

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
                    logger.warning(f"[SoapEngine] Parser output invalid ({generated_days} days), falling back to core engine")
                else:
                    logger.warning("[SoapEngine] Parser failed, falling back to core engine")
            except Exception as exc:
                logger.error(f"[SoapEngine] Parser exception: {exc}, falling back to core engine")

        profile = self._inject_focus_for_engine(profile)
        parsed_output = self._build_parsed_output(parsed)
        logger.info("[SoapEngine] Using fitness plan generator for exact matching")
        plan = self._build_fitness_plan(profile, parsed_output)
        plan = self._apply_restrictions_to_plan(plan, profile.get("restrictions", []))
        return self._normalize_plan_titles(plan)

    def _apply_restrictions_to_plan(self, plan: Dict[str, Any], restrictions: List[str]) -> Dict[str, Any]:
        if not isinstance(plan, dict) or not restrictions:
            return plan

        for day_name, day_plan in plan.items():
            if isinstance(day_plan, dict):
                plan[day_name] = self._apply_restrictions(day_plan, restrictions)

        return plan

    def _normalize_plan_titles(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return plan

        for day, content in plan.items():
            if not isinstance(content, dict):
                continue

            category = str(content.get("main_workout_category", "")).strip()
            title = str(content.get("workout_title", "")).strip()
            normalized = category.lower() if category else title.lower()

            if "upper" in normalized:
                content["main_workout_category"] = "Upper Body Strength"
                content["workout_title"] = "Upper Body Strength Workout"
            elif "lower" in normalized:
                content["main_workout_category"] = "Lower Body Strength"
                content["workout_title"] = "Lower Body Strength Workout"
            elif "cardio" in normalized or "tabata" in normalized:
                content["main_workout_category"] = "Cardio Conditioning"
                content["workout_title"] = "Cardio Conditioning Workout"
            elif "full" in normalized or "general" in normalized:
                content["main_workout_category"] = "Full Body Strength"
                content["workout_title"] = "Full Body Strength Workout"
            elif normalized:
                content["main_workout_category"] = f"{category.title()}"
                content["workout_title"] = f"{content["main_workout_category"]} Workout"

        return plan

    def _normalize_row(self, row):
        return {k.strip().lower(): v for k, v in row.items()}

    def _normalize_tags(self, tags: Any) -> List[str]:
        tags_text = str(tags or "").strip().lower()
        if not tags_text:
            return []
        # Support comma and slash separators and normalize spacing
        return [tag.strip() for tag in re.split(r"[,/]+", tags_text) if tag.strip()]

    def _parse_rpe_range(self, rpe: Any) -> tuple[Optional[float], Optional[float]]:
        rpe_text = str(rpe or "").strip().lower().replace("rpe", "").replace("na", "").replace("n/a", "").replace("–", "-").replace("—", "-")
        if not rpe_text:
            return None, None
        parts = [part.strip() for part in rpe_text.split("-") if part.strip()]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                continue
        if not values:
            return None, None
        if len(values) == 1:
            return values[0], values[0]
        return min(values), max(values)

    def _normalize_split_focus(self, focus: str) -> str:
        focus = (focus or "").strip().lower()
        if "upper" in focus:
            return "Upper Body"
        if "lower" in focus:
            return "Lower Body"
        if "full" in focus:
            return "Full Body"
        if "cardio" in focus or "tabata" in focus:
            return "Cardio"
        if "resistance" in focus or "strength" in focus or "muscle" in focus:
            return "Strength"
        if "mobility" in focus or "flexibility" in focus:
            return "Mobility"
        return "General"

    def _exercise_matches_restrictions(self, row: Any, restrictions: List[str]) -> bool:
        if not restrictions:
            return False

        row_text = " ".join(
            str(row.get(col, "") or "").lower()
            for col in [
                "Exercise Name",
                "exercise_name",
                "tags",
                "Tags",
                "is_not_suitable_for",
                "physical limitation",
                "primary_category",
                "Body Region",
                "body_region",
            ]
        )

        for restriction in restrictions:
            if restriction == "knee pain":
                if any(term in row_text for term in ["jump", "hop", "plyo", "burpee", "box jump", "skipping", "lunge", "squat"]):
                    return True
            if restriction == "low back pain":
                if any(term in row_text for term in ["deadlift", "good morning", "bend", "back extension", "superman", "hip hinge"]):
                    return True
            if restriction == "shoulder pain":
                if any(term in row_text for term in ["overhead", "press", "snatch", "bench", "lat raise", "upright row"]):
                    return True
            if restriction.lower() in row_text:
                return True

        return False

    def _get_tagged_exercises(self, tag, equipment_preference: Optional[List[str]] = None, restrictions: Optional[List[str]] = None):
        df = self.soap_df if not self.soap_df.empty else self.fitness_df
        df = df.copy()
        df["tags_list"] = df.get("tags", pd.Series()).apply(self._normalize_tags)

        filtered = df[df["tags_list"].apply(lambda tags: tag.lower().strip() in tags)]
        if restrictions:
            filtered = filtered[~filtered.apply(lambda row: self._exercise_matches_restrictions(row, restrictions), axis=1)]

        if equipment_preference:
            equipment_lower = [eq.lower() for eq in equipment_preference]
            filtered["_equip_preferred"] = filtered.get("equipments", pd.Series()).astype(str).str.lower().apply(
                lambda x: any(eq in x for eq in equipment_lower)
            )
            preferred = filtered[filtered["_equip_preferred"]].copy()
            fallback = filtered[~filtered["_equip_preferred"]].copy()
            filtered = pd.concat([preferred, fallback], ignore_index=True)
            filtered = filtered.drop(columns=["_equip_preferred"], errors="ignore")

        return filtered.to_dict("records")

    def _format_exercise(self, row, force_single_set: bool = False):
        """Format a SOAP dataset row into an exercise dict."""
        if isinstance(row, dict):
            ex_name = (row.get("Exercise Name") or row.get("exercise_name") or 
                      row.get("exercise name") or "")
            sets_val = row.get("Sets") or row.get("sets") or ""
            reps_val = row.get("Reps") or row.get("reps") or ""
            rest_val = row.get("Rest intervals") or row.get("rest_intervals") or row.get("rest intervals") or ""
            rpe_val = row.get("RPE") or row.get("rpe") or ""
            benefit = row.get("Health benefit") or row.get("health_benefit") or row.get("health benefit") or ""
            equip = row.get("Equipments") or row.get("equipments") or ""
            body_region = row.get("Body Region") or row.get("body_region") or row.get("body region") or ""
            steps = row.get("Steps to perform") or row.get("steps_to_perform") or row.get("steps") or ""
            safety = row.get("Safety cue") or row.get("safety_cue") or row.get("safety_notes") or ""
        else:
            ex_name = row.get("Exercise Name", "") or row.get("exercise_name", "") or ""
            sets_val = row.get("Sets", "") or row.get("sets", "") or ""
            reps_val = row.get("Reps", "") or row.get("reps", "") or ""
            rest_val = (row.get("Rest intervals", "") or row.get("rest_intervals", "") or 
                       row.get("rest intervals", "") or "")
            rpe_val = row.get("RPE", "") or row.get("rpe", "") or ""
            benefit = (row.get("Health benefit", "") or row.get("health_benefit", "") or 
                      row.get("health benefit", "") or "")
            equip = row.get("Equipments", "") or row.get("equipments", "") or ""
            body_region = (row.get("Body Region", "") or row.get("body_region", "") or 
                          row.get("body region", "") or "")
            steps = (row.get("Steps to perform", "") or row.get("steps_to_perform", "") or 
                    row.get("steps", "") or "")
            safety = (row.get("Safety cue", "") or row.get("safety_cue", "") or 
                     row.get("safety_notes", "") or "")

        sets = str(sets_val).strip()
        reps = str(reps_val).strip().replace("�", "-")
        rest = str(rest_val).strip().replace("�", "-")
        rpe = str(rpe_val).strip()

        if force_single_set:
            sets = "1"

        if rpe.lower() in {"n/a", "na", "none", "nan"}:
            rpe = ""

        if reps and not any(unit in reps.lower() for unit in ["sec", "second", "min", "minute", "each"]):
            if re.match(r"^\d+(?:-\d+)?$", reps):
                reps = f"{reps} sec"

        return {
            "exercise_name": str(ex_name).strip(),
            "name": str(ex_name).strip(),
            "sets": sets,
            "reps": reps,
            "rest": rest,
            "rpe": rpe,
            "benefit": str(benefit).strip(),
            "equipment": str(equip).strip(),
            "body_region": str(body_region).strip(),
            "steps_to_perform": str(steps).strip(),
            "safety_cue": str(safety).strip()
        }

    def _build_warmup(self, focus, equipment: Optional[List[str]] = None, restrictions: Optional[List[str]] = None, day_name: str = ""):
        """Build warmup with 1 cardio, 1 upper mobility, 1 lower mobility from SOAP dataset."""
        if self.soap_df.empty:
            return []

        df = self.soap_df.copy()
        df["tags_list"] = df.get("tags", pd.Series()).apply(self._normalize_tags)

        if restrictions:
            df = df[~df.apply(lambda row: self._exercise_matches_restrictions(row, restrictions), axis=1)]

        warmup_pool = df[df["tags_list"].apply(lambda tags: "warm up" in tags)]
        if warmup_pool.empty:
            return []

        warmup_pool["rpe_min"], warmup_pool["rpe_max"] = zip(*warmup_pool.get("rpe", warmup_pool.get("RPE", pd.Series())).apply(self._parse_rpe_range))

        def sort_candidates(pool):
            sort_columns = [col for col in ["rpe_min", "exercise_name", "Exercise Name"] if col in pool.columns]
            return pool.sort_values(by=sort_columns, na_position="last")

        if equipment:
            equipment_lower = [eq.lower() for eq in equipment]
            warmup_pool["_equip_preferred"] = warmup_pool.get("equipments", pd.Series()).astype(str).str.lower().apply(
                lambda x: any(eq in x for eq in equipment_lower)
            )
            warmup_pool = pd.concat([warmup_pool[warmup_pool["_equip_preferred"]], warmup_pool[~warmup_pool["_equip_preferred"]]], ignore_index=True)
            warmup_pool = warmup_pool.drop(columns=["_equip_preferred"], errors="ignore")

        day_index = DAY_ORDER.index(day_name) if day_name in DAY_ORDER else 0

        def pick_from_pool(pool, predicate=None, used_names=None):
            used_names = used_names or set()
            pool_sorted = sort_candidates(pool)
            for idx in range(len(pool_sorted)):
                ex = pool_sorted.iloc[(day_index + idx) % len(pool_sorted)]
                ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip()
                if not ex_name or ex_name in used_names:
                    continue
                if predicate is None or predicate(ex):
                    used_names.add(ex_name)
                    formatted = self._format_exercise(ex, force_single_set=True)
                    return formatted, used_names
            return None, used_names

        def warmup_intensity_ok(ex):
            rpe_min = ex.get("rpe_min")
            if rpe_min is None:
                return True
            return rpe_min < 7

        result = []
        used_names = set()

        cardio_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: str(ex.get("primary_category", "")).lower() == "cardio",
            used_names,
        )
        if cardio_ex:
            cardio_ex["reps"] = "1-2 min"
            result.append(cardio_ex)

        upper_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: "upper" in str(ex.get("body_region", "")).lower()
                       and bool(re.search(r"mobility|activation", str(ex.get("primary_category", "")), re.I))
                       and warmup_intensity_ok(ex),
            used_names,
        )
        if upper_ex:
            result.append(upper_ex)

        lower_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: "lower" in str(ex.get("body_region", "")).lower()
                       and bool(re.search(r"mobility|activation", str(ex.get("primary_category", "")), re.I))
                       and warmup_intensity_ok(ex),
            used_names,
        )
        if lower_ex:
            result.append(lower_ex)

        while len(result) < 3 and len(warmup_pool) > 0:
            ex, used_names = pick_from_pool(warmup_pool, None, used_names)
            if not ex:
                break
            result.append(ex)

        return result[:3]

        if equipment:
            equipment_lower = [eq.lower() for eq in equipment]
            warmup_pool["_equip_preferred"] = warmup_pool.get("equipments", pd.Series()).astype(str).str.lower().apply(
                lambda x: any(eq in x for eq in equipment_lower)
            )
            warmup_pool = pd.concat([warmup_pool[warmup_pool["_equip_preferred"]], warmup_pool[~warmup_pool["_equip_preferred"]]], ignore_index=True)
            warmup_pool = warmup_pool.drop(columns=["_equip_preferred"], errors="ignore")

        day_index = DAY_ORDER.index(day_name) if day_name in DAY_ORDER else 0

        def pick_from_pool(pool, predicate=None, used_names=None):
            used_names = used_names or set()
            for idx in range(len(pool)):
                ex = pool.iloc[(day_index + idx) % len(pool)]
                ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip()
                if not ex_name or ex_name in used_names:
                    continue
                if predicate is None or predicate(ex):
                    used_names.add(ex_name)
                    formatted = self._format_exercise(ex, force_single_set=True)
                    return formatted, used_names
            return None, used_names

        result = []
        used_names = set()

        cardio_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: str(ex.get("primary_category", "")).lower() == "cardio",
            used_names,
        )
        if cardio_ex:
            cardio_ex["reps"] = "1-2 min"
            result.append(cardio_ex)

        upper_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: "upper" in str(ex.get("body_region", "")).lower() and bool(re.search(r"mobility|activation", str(ex.get("primary_category", "")), re.I)),
            used_names,
        )
        if upper_ex:
            result.append(upper_ex)

        lower_ex, used_names = pick_from_pool(
            warmup_pool,
            lambda ex: "lower" in str(ex.get("body_region", "")).lower() and bool(re.search(r"mobility|activation", str(ex.get("primary_category", "")), re.I)),
            used_names,
        )
        if lower_ex:
            result.append(lower_ex)

        while len(result) < 3 and len(warmup_pool) > 0:
            ex, used_names = pick_from_pool(warmup_pool, None, used_names)
            if not ex:
                break
            result.append(ex)

        return result[:3]

    def _build_cooldown(self, focus, equipment: Optional[List[str]] = None, restrictions: Optional[List[str]] = None, day_name: str = ""):
        """Build cooldown with 3 stretching exercises from SOAP dataset."""
        if self.soap_df.empty:
            return []

        df = self.soap_df.copy()
        df["tags_list"] = df.get("tags", pd.Series()).apply(self._normalize_tags)

        if restrictions:
            df = df[~df.apply(lambda row: self._exercise_matches_restrictions(row, restrictions), axis=1)]

        cooldown_pool = df[df["tags_list"].apply(lambda tags: "cooldown" in tags)]
        if cooldown_pool.empty:
            return []

        stretch_pool = cooldown_pool[
            cooldown_pool.get("primary_category", pd.Series()).astype(str).str.contains("Stretch|Flexibility|Mobility|Recovery", case=False, na=False, regex=True)
        ]
        if stretch_pool.empty:
            stretch_pool = cooldown_pool

        if equipment:
            equipment_lower = [eq.lower() for eq in equipment]
            stretch_pool["_equip_preferred"] = stretch_pool.get("equipments", pd.Series()).astype(str).str.lower().apply(
                lambda x: any(eq in x for eq in equipment_lower)
            )
            stretch_pool = pd.concat([stretch_pool[stretch_pool["_equip_preferred"]], stretch_pool[~stretch_pool["_equip_preferred"]]], ignore_index=True)
            stretch_pool = stretch_pool.drop(columns=["_equip_preferred"], errors="ignore")

        day_index = DAY_ORDER.index(day_name) if day_name in DAY_ORDER else 0
        result = []
        seen = set()

        for idx in range(min(3, len(stretch_pool))):
            ex = stretch_pool.iloc[(day_index + idx) % len(stretch_pool)]
            ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip().lower()
            if ex_name and ex_name not in seen:
                result.append(self._format_exercise(ex, force_single_set=True))
                seen.add(ex_name)

        idx = 0
        while len(result) < 3 and idx < len(stretch_pool):
            ex = stretch_pool.iloc[(day_index + idx) % len(stretch_pool)]
            ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip().lower()
            if ex_name and ex_name not in seen:
                result.append(self._format_exercise(ex, force_single_set=True))
                seen.add(ex_name)
            idx += 1

        return result[:3]

        stretch_pool = cooldown_pool[
            cooldown_pool.get("primary_category", pd.Series()).astype(str).str.contains("Stretch|Flexibility|Mobility|Recovery", case=False, na=False, regex=True)
        ]
        if stretch_pool.empty:
            stretch_pool = cooldown_pool

        if equipment:
            equipment_lower = [eq.lower() for eq in equipment]
            stretch_pool["_equip_preferred"] = stretch_pool.get("equipments", pd.Series()).astype(str).str.lower().apply(
                lambda x: any(eq in x for eq in equipment_lower)
            )
            stretch_pool = pd.concat([stretch_pool[stretch_pool["_equip_preferred"]], stretch_pool[~stretch_pool["_equip_preferred"]]], ignore_index=True)
            stretch_pool = stretch_pool.drop(columns=["_equip_preferred"], errors="ignore")

        day_index = DAY_ORDER.index(day_name) if day_name in DAY_ORDER else 0
        result = []
        seen = set()

        for idx in range(min(3, len(stretch_pool))):
            ex = stretch_pool.iloc[(day_index + idx) % len(stretch_pool)]
            ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip().lower()
            if ex_name and ex_name not in seen:
                result.append(self._format_exercise(ex, force_single_set=True))
                seen.add(ex_name)

        idx = 0
        while len(result) < 3 and idx < len(stretch_pool):
            ex = stretch_pool.iloc[(day_index + idx) % len(stretch_pool)]
            ex_name = str(ex.get("exercise_name", "") or ex.get("Exercise Name", "")).strip().lower()
            if ex_name and ex_name not in seen:
                result.append(self._format_exercise(ex, force_single_set=True))
                seen.add(ex_name)
            idx += 1

        return result[:3]

    def _refine_main_workout(self, day_plan):
        main = day_plan.get("main_workout", [])

        dataset_main = self._get_tagged_exercises("main")

        if not main:
            return [self._format_exercise(e) for e in dataset_main[:5]]

        return main

    def _apply_restrictions(self, day_plan, restrictions):
        if not restrictions:
            return day_plan

        all_blocks = ["warmup", "main_workout", "cooldown"]

        for block in all_blocks:
            exercises = day_plan.get(block, [])
            filtered = []

            for ex in exercises:
                if self._exercise_matches_restrictions(ex, restrictions):
                    continue
                filtered.append(ex)

            day_plan[block] = filtered

        return day_plan

    def _prioritize_soap_dataset(self, day_plan):
        soap_names = set(self.soap_df["exercise_name"].str.lower().tolist())

        for block in ["warmup", "main_workout", "cooldown"]:
            exercises = day_plan.get(block, [])

            exercises.sort(
                key=lambda x: str(x.get("exercise_name", "")).lower() not in soap_names
            )

            day_plan[block] = exercises

        return day_plan

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

    def _build_parsed_output(self, parsed: SoapParseResult) -> Dict[str, Any]:
        explicit_days = {}
        day_focus_map = parsed.inferred_profile.get("day_focus_map", {}) or {}
        for day in parsed.inferred_profile.get("days", []) or []:
            focus = day_focus_map.get(day, "General")
            explicit_days[day] = [focus]

        mandatory_exercises = []
        for item in parsed.prescribed_exercises or []:
            if isinstance(item, str) and item.strip():
                mandatory_exercises.append({"name": item.strip(), "category": "main"})

        parsed_output: Dict[str, Any] = {
            "extract": {
                "schedule": {"explicit_days": explicit_days}
            },
            "mandatory": mandatory_exercises,
        }
        return parsed_output

    def _build_fitness_plan(self, profile: Dict[str, Any], parsed_output: Dict[str, Any]) -> Dict[str, Any]:
        try:
            planner = FitnessPlanGeneratorTool()
            result = self._run_async(planner.execute(constraints=profile, parsed_output=parsed_output))
            if result and result.success:
                plan = result.data.get("plans_json") or result.data.get("json_plan") or {}
                return self._fill_missing_warmup_cooldown(plan, profile)
            logger.warning("[SoapEngine] FitnessPlanGeneratorTool failed, falling back to generate_plan_local")
        except Exception as exc:
            logger.error(f"[SoapEngine] FitnessPlanGeneratorTool exception: {exc}", exc_info=True)
        return self._fill_missing_warmup_cooldown(generate_plan_local(profile), profile)

    def _fill_missing_warmup_cooldown(self, plan: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return plan

        for day_name, content in plan.items():
            if not isinstance(content, dict):
                continue

            focus = content.get("main_workout_category", "")
            equipment = profile.get("equipment") if isinstance(profile, dict) else None
            restrictions = profile.get("restrictions") if isinstance(profile, dict) else None

            # Always populate warmup and cooldown from SOAP dataset
            content["warmup"] = self._build_warmup(focus, equipment=equipment, restrictions=restrictions, day_name=day_name)
            content["cooldown"] = self._build_cooldown(focus, equipment=equipment, restrictions=restrictions, day_name=day_name)

            plan[day_name] = content
        
        return plan

        return plan

    def _normalize_focus_label(self, focus: str) -> str:
        focus = (focus or "").strip().lower()

        if "upper" in focus:
            return "Upper Body Focus"
        if "lower" in focus:
            return "Lower Body Focus"
        if "full" in focus:
            return "Full Body Focus"
        if "cardio" in focus:
            return "Cardio Focus"
        return "Full Body Focus"

    def _inject_focus_for_engine(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        focus_map = profile.get("day_focus_map", {})

        for day in profile.get("structured_days", []):
            focus = focus_map.get(day["day_name"], "General")
            focus_lower = (focus or "").lower()

            if "upper" in focus_lower:
                day["muscle_focus"] = "upper"
            elif "lower" in focus_lower:
                day["muscle_focus"] = "lower"
            elif "full" in focus_lower:
                day["muscle_focus"] = "full"
            else:
                day["muscle_focus"] = "full"

        return profile

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
            "five": 5,
            "six": 6,
            "seven": 7,
        }
        text_lower = text.lower()

        matches = re.findall(r"\b(\d+|one|two|three|four|five|six|seven)\b\s*(?:x|times?|days?)\b", text_lower)
        total = 0
        for match in matches:
            if match.isdigit():
                total += int(match)
            else:
                total += word_to_num.get(match, 0)

        if total:
            return total

        numeric_match = re.search(r"(\d+)\s*(?:days?|times?|x)\s*(?:per\s*)?week", text_lower)
        if numeric_match:
            return int(numeric_match.group(1))

        word_match = re.search(r"(one|two|three|four|five|six|seven)\s*(?:days?|times?)\s*(?:per\s*)?week", text_lower)
        if word_match:
            return word_to_num.get(word_match.group(1))

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

        return [day for day in DAY_ORDER if day in result]

    def _detect_schedule_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(day.lower() in text_lower for day in DAY_ORDER):
            return "explicit_days"
        if re.search(r"(?:\d+|one|two|three|four|five|six|seven)\s*(?:times?|days?)\s*(?:per\s*)?week", text_lower):
            return "frequency_based"
        if re.search(r"(?:\d+|one|two|three|four|five|six|seven)\s*day.*(?:tabata|cardio|strength|resistance|mobility|flexibility|recovery|core|upper|lower|full)", text_lower, re.IGNORECASE):
            return "split_based"
        return "default"

    def _has_split_indicator(self, text: str) -> bool:
        text_lower = text.lower()
        return bool(re.search(r"(tabata|cardio|strength|resistance|mobility|flexibility|recovery|core|upper body|lower body|full body|interval|circuit)", text_lower))

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

            # 🔥 Streamlit + Local compatibility
            try:
                import streamlit as st
                api_key = st.secrets.get("AZURE_AI_KEY")
            except Exception:
                api_key = None

            if not api_key:
                api_key = os.getenv("AZURE_AI_KEY")

            logger.info(f"API KEY FOUND: {bool(api_key)}")

            endpoint = "https://nouriqfriskacc7470931625.cognitiveservices.azure.com/"
            api_version = "2024-12-01-preview"

            if not api_key:
                logger.warning("OpenAI API key is not configured; AI disabled.")
                self._ai_client = None
            else:
                self._ai_client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )

        except Exception as exc:
            logger.warning("OpenAI client init failed: %s — AI disabled.", exc)
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
            if "upper" in focus.lower():
                focus = "Upper Body"
            elif "lower" in focus.lower():
                focus = "Lower Body"
            elif "full" in focus.lower():
                focus = "Full Body"
            elif "cardio" in focus.lower():
                focus = "Cardio"
            else:
                focus = "Full Body"
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
            if ai_week_structure:
                if all(item["focus"].lower() == "strength" for item in ai_week_structure) or all(item["focus"].lower() == "general" for item in ai_week_structure):
                    logger.info("Adding variation → converting some days to Full Body")
                    for i in range(len(ai_week_structure)):
                        if i % 2 == 0:
                            ai_week_structure[i]["focus"] = "Full Body"
                logger.info("Using AI-derived week structure")
                logger.info(f"Week structure: {ai_week_structure}")
                return ai_week_structure
            logger.warning("AI returned weekly_days but no usable week structure; falling back to parser heuristics")

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

        if week_structure and (all(item["focus"].lower() == "strength" for item in week_structure) or all(item["focus"].lower() == "general" for item in week_structure)):
            logger.info("Adding variation → converting some days to Full Body")
            for i in range(len(week_structure)):
                if i % 2 == 0:
                    week_structure[i]["focus"] = "Full Body"

        logger.info(f"Week structure: {week_structure}")
        return week_structure

    def _parse_split_prescription(self, text: str) -> List[Dict[str, str]]:
        text_lower = text.lower()
        splits: List[Dict[str, str]] = []
        day_index = 0

        word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
        }

        split_terms = r"(?:tabata|cardio|strength|resistance|mobility|flexibility|recovery|core|upper\s*body|lower\s*body|full\s*body|general)"
        split_pattern_a = rf"(\d+|one|two|three|four|five|six|seven)\s*(?:day|days|times?)\s*(?:per\s*)?(?:week)?\s*{split_terms}"
        split_pattern_b = rf"{split_terms}.*?(\d+|one|two|three|four|five|six|seven)\s*(?:day|days|times?)"

        matches_a = re.findall(split_pattern_a, text_lower, re.IGNORECASE)
        matches_b = re.findall(split_pattern_b, text_lower, re.IGNORECASE)

        for match in matches_a:
            if len(match) == 2:
                count_str, focus = match
            else:
                continue
            count = int(count_str) if count_str.isdigit() else word_to_num.get(count_str, 1)
            focus = self._normalize_split_focus(focus)
            for _ in range(count):
                if day_index < len(DAY_ORDER):
                    splits.append({"day_name": DAY_ORDER[day_index], "focus": focus})
                    day_index += 1

        for match in matches_b:
            if len(match) == 2:
                focus, count_str = match
            else:
                continue
            count = int(count_str) if count_str.isdigit() else word_to_num.get(count_str, 1)
            focus = self._normalize_split_focus(focus)
            for _ in range(count):
                if day_index < len(DAY_ORDER):
                    splits.append({"day_name": DAY_ORDER[day_index], "focus": focus})
                    day_index += 1

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
        equipment_terms = {
            "dumbbells": "Dumbbells",
            "dumbbell": "Dumbbells",
            "dbs": "Dumbbells",
            "db": "Dumbbells",
            "resistance bands": "Resistance Bands",
            "resistance band": "Resistance Bands",
            "resistance": "Resistance Bands",
            "band": "Resistance Bands",
            "bands": "Resistance Bands",
            "kettlebell": "Kettlebell",
            "kb": "Kettlebell",
            "bench": "Bench",
            "foam roller": "Foam Roller",
        }
        for term, label in equipment_terms.items():
            if term in key and label not in found:
                found.append(label)
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
