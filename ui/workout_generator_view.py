from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from core.formatters import build_download_name
from services.workout_service import WorkoutService
from ui.shared_components import render_plan
from utils.constants import (
    BODY_PARTS,
    DAY_ORDER,
    DURATIONS,
    EQUIPMENT_LIST,
    FITNESS_LEVELS,
    GENDERS,
    LOCATIONS,
    MEDICAL_CONDITIONS,
    PHYSICAL_LIMITATIONS,
    PRIMARY_GOALS,
    SECONDARY_GOALS,
    UNIT_SYSTEMS,
)


def _calc_bmi(weight_kg: float, height_cm: float) -> Optional[float]:
    try:
        height_m = height_cm / 100
        if height_m <= 0:
            return None
        return round(weight_kg / (height_m * height_m), 1)
    except Exception:
        return None


def _bmi_label(bmi: Optional[float]) -> str:
    if bmi is None:
        return "-"
    if bmi < 18.5:
        return f"{bmi} - Underweight"
    if bmi < 25:
        return f"{bmi} - Normal"
    if bmi < 30:
        return f"{bmi} - Overweight"
    return f"{bmi} - Obese"


def render_workout_generator_view() -> None:
    st.markdown("## Workout Plan Generator")
    st.caption("Generate a complete weekly plan from the local fitness dataset.")

    if "generated_plan" not in st.session_state:
        st.session_state.generated_plan = None
    if "workout_profile" not in st.session_state:
        st.session_state.workout_profile = None

    with st.form("workout_generator_form"):
        st.markdown("### Personal Information")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            name = st.text_input("Full Name", placeholder="e.g. Alex Johnson")
        with c2:
            age = st.number_input("Age", min_value=5, max_value=100, value=30, step=1)
        with c3:
            gender = st.selectbox("Gender", GENDERS, index=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=75.0, step=0.5, format="%.1f")
        with c5:
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5, format="%.1f")
        with c6:
            bmi = _calc_bmi(weight, height)
            st.metric("BMI (auto-calculated)", _bmi_label(bmi))

        unit_system = st.selectbox("Unit System", UNIT_SYSTEMS, index=0)

        st.markdown("### Goals & Targets")
        g1, g2 = st.columns(2)
        with g1:
            primary_goal = st.selectbox("Primary Goal", PRIMARY_GOALS, index=0)
        with g2:
            secondary_goal = st.selectbox("Secondary Goal", SECONDARY_GOALS, index=0)
        target_body_parts = st.multiselect("Target Body Parts", BODY_PARTS, default=["Full Body"])

        st.markdown("### Fitness Profile")
        f1, f2 = st.columns(2)
        with f1:
            fitness_level = st.selectbox("Fitness Level", FITNESS_LEVELS, index=0)
        with f2:
            workout_location = st.selectbox("Workout Location", LOCATIONS, index=0)

        f3, f4 = st.columns(2)
        with f3:
            days_per_week = st.slider("Days Per Week", min_value=1, max_value=7, value=4)
        with f4:
            session_duration = st.selectbox("Session Duration", DURATIONS, index=0)

        available_equipment = st.multiselect("Available Equipment", EQUIPMENT_LIST, default=["No Equipment"])

        st.markdown("### Health & Medical")
        h1, h2 = st.columns(2)
        with h1:
            medical_conditions = st.multiselect(
                "Medical Conditions",
                MEDICAL_CONDITIONS,
                placeholder="Select medical conditions",
            )
        with h2:
            physical_limitations = st.multiselect(
                "Physical Limitations",
                PHYSICAL_LIMITATIONS,
                placeholder="Select physical limitations",
            )
        specific_avoidance = st.text_input(
            "Exercises / Movements to Avoid",
            placeholder="e.g. Burpees, Heavy squats",
        )

        submitted = st.form_submit_button("Generate Structured Workout Plan", type="primary")

    if submitted:
        service = WorkoutService()
        day_names = DAY_ORDER[: int(days_per_week)]
        weight_kg = weight if "Metric" in unit_system else round(weight * 0.45359237, 1)
        height_cm = height if "Metric" in unit_system else round(height * 2.54, 1)
        parsed_limitations = [item.strip() for item in physical_limitations if item.strip()]
        parsed_avoidance = [item.strip() for item in specific_avoidance.split(",") if item.strip()]
        primary_body_region = target_body_parts[0] if target_body_parts else "Full Body"
        restrictions = parsed_limitations + parsed_avoidance
        profile: Dict[str, object] = {
            "name": name.strip() or "User",
            "age": age,
            "gender": gender,
            "unit_system": unit_system,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": _calc_bmi(weight_kg, height_cm),
            "goal": primary_goal,
            "primary_goal": primary_goal,
            "secondary_goal": secondary_goal,
            "target_body_parts": target_body_parts,
            "body_region": primary_body_region,
            "fitness_level": fitness_level,
            "medical_conditions": medical_conditions or ["NONE"],
            "physical_limitations": ", ".join(parsed_limitations) if parsed_limitations else "None",
            "specific_avoidance": specific_avoidance.strip() or "None",
            "session_duration": session_duration,
            "location": workout_location,
            "workout_location": workout_location,
            "equipment": available_equipment,
            "available_equipment": available_equipment,
            "days": day_names,
            "days_per_week": day_names,
            "weekly_days": len(day_names),
            "restrictions": restrictions,
            "limitations": parsed_limitations,
        }
        st.session_state.workout_profile = profile
        st.session_state.generated_plan = service.build_plan(profile)

    if st.session_state.generated_plan:
        render_plan(
            st.session_state.generated_plan,
            json_download_name=build_download_name("workout_generator_plan"),
        )
