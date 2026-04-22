from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st

from core.formatters import build_download_name
from services.experts_note_service import ExpertsNoteService
from ui.shared_components import render_plan


SAMPLE_NOTES_DIR = Path(__file__).resolve().parents[1] / "Experts note samples"


def _load_sample_note_cards(service: ExpertsNoteService) -> None:
    sample_files = sorted(SAMPLE_NOTES_DIR.glob("*.pdf"), key=lambda path: path.name.lower())

    with st.expander("Sample Notes Library", expanded=False):
        st.caption("Use any sample PDF directly for testing, or open and download the original note.")

        if not sample_files:
            st.info("No sample PDFs found in the Experts note samples folder.")
            return

        for index, sample_path in enumerate(sample_files, start=1):
            sample_bytes = sample_path.read_bytes()
            encoded_pdf = base64.b64encode(sample_bytes).decode("utf-8")
            open_href = f"data:application/pdf;base64,{encoded_pdf}"

            with st.container(border=True):
                st.markdown(f"**Sample {index}**")
                st.caption(sample_path.name)

                def load_sample_note(path: Path = sample_path) -> None:
                    st.session_state.experts_source_text = service.extract_pdf_text(path)
                    st.session_state.experts_input_mode = "Paste Text"
                    st.session_state.experts_selected_sample = path.name

                st.button(
                    "Use in Experts Note",
                    key=f"use_sample_{index}",
                    use_container_width=True,
                    on_click=load_sample_note,
                )

                st.markdown(
                    f"""
                    <a href="{open_href}" target="_blank" style="text-decoration:none;">
                        Open PDF
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download PDF",
                    data=sample_bytes,
                    file_name=sample_path.name,
                    mime="application/pdf",
                    key=f"download_sample_{index}",
                    use_container_width=True,
                )


def render_experts_note_view() -> None:
    st.markdown("## Expert / Doctor Notes Plan")
    st.caption("Upload expert or doctor notes, or use the sample-note panel, to generate a personalized workout plan using default health parameters.")

    # Initialize every widget-backed key before any widgets are instantiated.
    if "experts_result" not in st.session_state:
        st.session_state.experts_result = None
    if "experts_input_mode" not in st.session_state:
        st.session_state.experts_input_mode = "Paste Text"
    if "experts_source_text" not in st.session_state:
        st.session_state.experts_source_text = ""
    if "experts_selected_sample" not in st.session_state:
        st.session_state.experts_selected_sample = None

    service = ExpertsNoteService()
    main_col, samples_col = st.columns([1.9, 1], gap="large")

    with main_col:
        input_mode = st.radio(
            "Input Type",
            ["Paste Text", "PDF Upload"],
            horizontal=True,
            key="experts_input_mode",
        )
        source_text = st.session_state.experts_source_text

        if st.session_state.experts_selected_sample:
            st.info(f"Loaded sample note: {st.session_state.experts_selected_sample}")

        if input_mode == "PDF Upload":
            uploaded_file = st.file_uploader("Upload Expert Notes (PDF)", type=["pdf"])
            if uploaded_file is not None:
                try:
                    with st.spinner("Reading PDF..."):
                        source_text = service.extract_pdf_text(uploaded_file)
                        st.session_state.experts_source_text = source_text
                        st.text_area("Extracted PDF Text", source_text, height=220, disabled=True)
                except RuntimeError as exc:
                    st.error(str(exc))
        else:
            source_text = st.text_area(
                "Paste Expert / Doctor Notes",
                height=320,
                key="experts_source_text",
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
            st.success("Plan generated successfully. Scroll down below to view it.")

    with samples_col:
        _load_sample_note_cards(service)

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
