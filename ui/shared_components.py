from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import streamlit as st


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --panel-bg-light: #eef6f4;
                --panel-border-light: #cfe1dc;
                --panel-text-light: #1f3440;
                --panel-muted-light: #5b7280;
                --panel-bg-dark: #14232a;
                --panel-border-dark: #29414a;
                --panel-text-dark: #eef7f5;
                --panel-muted-dark: #c4d6d1;
                --toggle-active-bg: linear-gradient(135deg, #0f596b 0%, #1f7a8c 100%);
                --toggle-active-text: #f7fcff;
                --toggle-active-border: #49b3c8;
                --toggle-hover-light: #dfeee9;
                --toggle-hover-dark: #1d3138;
            }

            html, body, [class*="css"] {
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            }

            [data-testid="stSidebar"] {
                display: none;
            }

            .topbar {
                background: linear-gradient(135deg, #12343b 0%, #1f5c69 52%, #3f8c7a 100%);
                border-radius: 18px;
                padding: 1.7rem 2rem;
                margin-bottom: 1rem;
                color: white;
                box-shadow: 0 18px 40px rgba(15, 32, 39, 0.18);
            }

            .topbar-eyebrow {
                margin: 0 0 0.35rem;
                font-size: 0.82rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                opacity: 0.76;
            }

            .topbar h1 {
                margin: 0;
                font-size: 2.4rem;
                letter-spacing: -0.04em;
            }

            div[role="radiogroup"] {
                background: var(--panel-bg-light);
                border: 1px solid var(--panel-border-light);
                border-radius: 16px;
                padding: 0.45rem;
                gap: 0.5rem;
                margin-bottom: 1.4rem;
            }

            div[role="radiogroup"] label {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 12px;
                padding: 0.55rem 1rem;
                color: var(--panel-text-light) !important;
                transition: background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
            }

            div[role="radiogroup"] label:hover {
                background: var(--toggle-hover-light);
            }

            div[role="radiogroup"] label:has(input:checked) {
                background: var(--toggle-active-bg);
                border-color: var(--toggle-active-border);
                box-shadow: 0 8px 18px rgba(15, 89, 107, 0.22);
                color: var(--toggle-active-text) !important;
            }

            div[role="radiogroup"] label:has(input:checked) p,
            div[role="radiogroup"] label:has(input:checked) span,
            div[role="radiogroup"] label:has(input:checked) div {
                color: var(--toggle-active-text) !important;
                font-weight: 600;
            }

            div[role="radiogroup"] label p,
            div[role="radiogroup"] label span,
            div[role="radiogroup"] label div {
                color: inherit !important;
            }

            .section-title {
                font-size: 1rem;
                font-weight: 700;
                color: #23404c;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin: 1.2rem 0 0.7rem;
            }

            .exercise-card {
                background: #f8fbfc;
                border: 1px solid #dbe8ec;
                border-radius: 14px;
                padding: 1rem 1rem 0.9rem;
                margin-bottom: 0.85rem;
            }

            .exercise-name {
                font-size: 1rem;
                font-weight: 700;
                color: #132f3a;
                margin-bottom: 0.35rem;
            }

            .exercise-meta {
                font-size: 0.85rem;
                color: #476270;
                margin-bottom: 0.5rem;
            }

            .exercise-copy {
                font-size: 0.9rem;
                color: #2f4854;
                line-height: 1.55;
            }

            @media (prefers-color-scheme: dark) {
                div[role="radiogroup"] {
                    background: var(--panel-bg-dark);
                    border-color: var(--panel-border-dark);
                }

                div[role="radiogroup"] label {
                    color: var(--panel-text-dark) !important;
                }

                div[role="radiogroup"] label:hover {
                    background: var(--toggle-hover-dark);
                }

                div[role="radiogroup"] label:not(:has(input:checked)) {
                    color: var(--panel-muted-dark) !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _parse_first_number(text: Any) -> Optional[float]:
    if text in (None, ""):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(text).replace("\u2013", "-"))
    return float(match.group()) if match else None


def _estimate_duration_seconds(exercise: dict) -> Optional[int]:
    est_time_sec = _to_float(exercise.get("est_time_sec"))
    if est_time_sec and est_time_sec > 0:
        return int(est_time_sec)

    sets = max(1, int(_parse_first_number(exercise.get("sets")) or 1))
    reps_text = str(exercise.get("reps", "") or "").lower().replace("\u2013", "-")
    rest_text = str(exercise.get("rest", "") or "").lower().replace("\u2013", "-")

    rep_numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", reps_text)]
    rest_numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", rest_text)]
    rep_value = rep_numbers[0] if rep_numbers else 10.0
    if len(rep_numbers) >= 2:
        rep_value = sum(rep_numbers[:2]) / 2.0

    rest_value = rest_numbers[0] if rest_numbers else 30.0
    if len(rest_numbers) >= 2:
        rest_value = sum(rest_numbers[:2]) / 2.0

    is_hold = "sec" in reps_text or "hold" in reps_text or reps_text.endswith("s")
    active_seconds = rep_value if is_hold else rep_value * 4.0
    total_seconds = (active_seconds * sets) + (rest_value * max(0, sets - 1))
    return int(total_seconds) if total_seconds > 0 else None


def _resolve_exercise_calories(exercise: dict, user_profile: Optional[Dict[str, Any]]) -> Optional[int]:
    for key in ("estimated_calories", "planned_total_cal"):
        value = _to_float(exercise.get(key))
        if value is not None and value >= 0:
            return int(round(value))

    met_value = _to_float(exercise.get("met_value"))
    profile = user_profile or {}
    weight_kg = _to_float(profile.get("weight_kg")) or _to_float(profile.get("weight"))
    if met_value is None or weight_kg is None or weight_kg <= 0:
        return None

    duration_seconds = _estimate_duration_seconds(exercise)
    if not duration_seconds:
        return None

    calories = (met_value * 3.5 * weight_kg / 200.0) * (duration_seconds / 60.0)
    return int(round(calories))


def render_plan(
    plan: Dict[str, dict],
    json_download_name: str = "workout_plan.json",
    user_profile: Optional[Dict[str, Any]] = None,
) -> None:
    if not plan:
        st.info("No plan available yet.")
        return

    plan_tab, raw_tab = st.tabs(["Workout Plan", "JSON"])
    with plan_tab:
        for day, day_data in plan.items():
            with st.expander(f"{day} - {day_data.get('main_workout_category', 'Workout Session')}", expanded=True):
                for title, section_key in [
                    ("Warmup", "warmup"),
                    ("Main Workout", "main_workout"),
                    ("Cooldown", "cooldown"),
                ]:
                    exercises = day_data.get(section_key, [])
                    if not exercises:
                        continue
                    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
                    for exercise in exercises:
                        _render_exercise_card(exercise, user_profile=user_profile)

    with raw_tab:
        final_json = json.dumps(plan, indent=2)
        st.code(final_json, language="json")
        st.download_button(
            label="Download Plan as JSON",
            data=final_json,
            file_name=json_download_name,
            mime="application/json",
        )


def _render_exercise_card(exercise: dict, user_profile: Optional[Dict[str, Any]] = None) -> None:
    rpe_value = str(exercise.get("intensity_rpe", "N/A")).replace("RPE", "").strip()
    calories = _resolve_exercise_calories(exercise, user_profile)
    calories_text = f"Calories: {calories} kcal" if calories is not None else None
    st.markdown(
        f"""
        <div class="exercise-card">
            <div class="exercise-name">{exercise.get("name", "Exercise")}</div>
            <div class="exercise-meta">
                Sets: {exercise.get("sets", "N/A")} |
                Reps: {exercise.get("reps", "N/A")} |
                RPE: {rpe_value or "N/A"} |
                Rest: {exercise.get("rest", "N/A")}{f" | {calories_text}" if calories_text else ""}
            </div>
            <div class="exercise-copy">{exercise.get("benefit", "No benefit listed.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1.3, 1])
    with cols[0]:
        if exercise.get("equipment"):
            st.caption(f"Equipment: {exercise['equipment']}")
        if exercise.get("safety_cue"):
            st.caption(f"Safety cue: {exercise['safety_cue']}")
        
        # Handle both "steps" (list) and "steps_to_perform" (string) formats
        steps_data = exercise.get("steps") or exercise.get("steps_to_perform")
        if steps_data:
            with st.expander("Steps to perform"):
                if isinstance(steps_data, list):
                    for step in steps_data:
                        st.write(f"- {step}")
                else:
                    # If it's a string, split by newlines or display as-is
                    step_text = str(steps_data).strip()
                    if "\n" in step_text:
                        for line in step_text.split("\n"):
                            if line.strip():
                                st.write(line.strip())
                    else:
                        st.write(step_text)

    with cols[1]:
        if exercise.get("video_url"):
            st.video(exercise["video_url"])
        elif exercise.get("thumbnail_url"):
            st.image(exercise["thumbnail_url"], use_container_width=True)
        else:
            st.info("Video unavailable for this exercise.")
