from __future__ import annotations

import json

import streamlit as st

from core.formatters import build_download_name
from services.experts_note_service import ExpertsNoteService
from ui.shared_components import render_plan


def render_experts_note_view() -> None:
    st.markdown("## Expert / Doctor Notes Plan")
    st.caption("Upload expert or doctor notes (PDF or text) to generate a personalized workout plan using default health parameters.")

    if "experts_result" not in st.session_state:
        st.session_state.experts_result = None

    service = ExpertsNoteService()
    input_mode = st.radio("Input Type", ["Paste Text", "PDF Upload"], horizontal=True)
    source_text = ""

    if input_mode == "PDF Upload":
        uploaded_file = st.file_uploader("Upload Expert Notes (PDF)", type=["pdf"])
        if uploaded_file is not None:
            try:
                with st.spinner("Reading PDF..."):
                    source_text = service.extract_pdf_text(uploaded_file)
                    st.text_area("Extracted PDF Text", source_text, height=220)
            except RuntimeError as exc:
                st.error(str(exc))
    else:
        source_text = st.text_area(
            "Paste Expert / Doctor Notes",
            height=320,
            placeholder="Paste expert or doctor notes here (recommendations, observations, prescriptions, etc.)...",
        )

    if st.button("Generate Plan from Notes", type="primary"):
        if source_text.strip():
            with st.spinner("Generating workout plan..."):
                try:
                    st.session_state.experts_result = service.generate_plan_from_notes(source_text)
                except Exception as exc:
                    st.error(f"Error generating plan: {str(exc)}")
        else:
            st.warning("Please provide expert notes first.")

    if st.session_state.experts_result:
        result = st.session_state.experts_result
        plan = result.get("plan", {})
        profile = result.get("profile", {})

        plan_tab, profile_tab = st.tabs(["View Workout Plan", "Applied Profile"])
        with plan_tab:
            render_plan(
                plan,
                json_download_name=build_download_name("experts_note_workout_plan"),
                user_profile=profile,
            )

        with profile_tab:
            st.subheader("Default Profile Used")
            st.json(profile)
            st.info(
                "Since no intake form was completed, this plan was generated using default health and fitness parameters. "
                "Adjust profile parameters in the Fitness Plan Generator for a more personalized experience."
            )
