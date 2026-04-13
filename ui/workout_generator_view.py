from __future__ import annotations

from typing import Dict

import streamlit as st

from core.formatters import build_download_name
from services.workout_service import WorkoutService
from ui.shared_components import render_plan
from utils.constants import (
    BODY_PARTS,
    DURATIONS,
    EQUIPMENT_LIST,
    FITNESS_LEVELS,
    GENDERS,
    LOCATIONS,
    PRIMARY_GOALS,
    SECONDARY_GOALS,
    UNIT_SYSTEMS,
)


def render_workout_generator_view() -> None:
    st.markdown("## Workout Plan Generator")
    st.caption("Generate a complete weekly plan from the local fitness dataset.")

    if "generated_plan" not in st.session_state:
        st.session_state.generated_plan = None
    if "workout_profile" not in st.session_state:
        st.session_state.workout_profile = None

    with st.form("workout_generator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=90, value=35)
            gender = st.selectbox("Gender", GENDERS, index=0)
            unit_system = st.selectbox("Units", UNIT_SYSTEMS, index=0)
            weight = st.number_input("Weight", min_value=40.0, max_value=180.0, value=72.0)
            height = st.number_input("Height", min_value=140.0, max_value=220.0, value=170.0)
        with col2:
            primary_goal = st.selectbox("Primary Goal", PRIMARY_GOALS, index=0)
            secondary_goal = st.selectbox("Secondary Goal", SECONDARY_GOALS, index=0)
            body_region = st.selectbox("Body Focus", BODY_PARTS, index=0)
            fitness_level = st.selectbox("Fitness Level", FITNESS_LEVELS, index=0)
            session_duration = st.selectbox("Session Duration", DURATIONS, index=4)
        with col3:
            location = st.selectbox("Location", LOCATIONS, index=0)
            equipment = st.multiselect("Available Equipment", EQUIPMENT_LIST, default=["No Equipment"])
            days = st.multiselect(
                "Workout Days",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                default=["Monday", "Wednesday", "Friday"],
            )
            notes = st.text_area("Notes / Restrictions", placeholder="Example: avoid knee pain triggers")

        submitted = st.form_submit_button("Generate Structured Workout Plan", type="primary")

    if submitted:
        service = WorkoutService()
        restrictions = [item.strip() for item in notes.split(",") if item.strip()]
        profile: Dict[str, object] = {
            "age": age,
            "gender": gender,
            "unit_system": unit_system,
            "weight_kg": weight if "Metric" in unit_system else round(weight * 0.45359237, 1),
            "height_cm": height if "Metric" in unit_system else round(height * 2.54, 1),
            "goal": primary_goal,
            "primary_goal": primary_goal,
            "secondary_goal": secondary_goal,
            "body_region": body_region,
            "fitness_level": fitness_level,
            "session_duration": session_duration,
            "location": location,
            "equipment": equipment,
            "days": days,
            "weekly_days": len(days),
            "restrictions": restrictions,
        }
        st.session_state.workout_profile = profile
        st.session_state.generated_plan = service.build_plan(profile)

    if st.session_state.generated_plan:
        render_plan(
            st.session_state.generated_plan,
            json_download_name=build_download_name("workout_generator_plan"),
        )
