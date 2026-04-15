from __future__ import annotations

import json
from typing import Dict

import streamlit as st


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            }

            .hero-header {
                background: linear-gradient(135deg, #0f2027 0%, #203a43 48%, #2c5364 100%);
                border-radius: 18px;
                padding: 2.6rem 2rem;
                margin-bottom: 1.8rem;
                color: white;
                box-shadow: 0 18px 40px rgba(15, 32, 39, 0.18);
            }

            .hero-header h1 {
                margin: 0 0 0.45rem;
                font-size: 2.4rem;
                letter-spacing: -0.04em;
            }

            .hero-header p {
                margin: 0;
                font-size: 1.02rem;
                opacity: 0.82;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_plan(plan: Dict[str, dict], json_download_name: str = "workout_plan.json") -> None:
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
                        _render_exercise_card(exercise)

    with raw_tab:
        final_json = json.dumps(plan, indent=2)
        st.code(final_json, language="json")
        st.download_button(
            label="Download Plan as JSON",
            data=final_json,
            file_name=json_download_name,
            mime="application/json",
        )


def _render_exercise_card(exercise: dict) -> None:
    rpe_value = str(exercise.get("intensity_rpe", "N/A")).replace("RPE", "").strip()
    st.markdown(
        f"""
        <div class="exercise-card">
            <div class="exercise-name">{exercise.get("name", "Exercise")}</div>
            <div class="exercise-meta">
                Sets: {exercise.get("sets", "N/A")} |
                Reps: {exercise.get("reps", "N/A")} |
                RPE: {rpe_value or "N/A"} |
                Rest: {exercise.get("rest", "N/A")}
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
        if exercise.get("steps"):
            with st.expander("Steps to perform"):
                for step in exercise["steps"]:
                    st.write(f"- {step}")

    with cols[1]:
        if exercise.get("video_url"):
            st.video(exercise["video_url"])
        elif exercise.get("thumbnail_url"):
            st.image(exercise["thumbnail_url"], use_container_width=True)
        else:
            st.info("Video unavailable for this exercise.")
