from __future__ import annotations

import json

import streamlit as st

from core.formatters import build_download_name
from services.soap_parser_service import SoapParserService
from ui.shared_components import render_plan


def render_soap_testing_view() -> None:
    st.markdown("## SOAP Note Based Plan")
    st.caption("Paste a SOAP note or upload a PDF to parse clinical context and build a local workout plan.")

    if "soap_result" not in st.session_state:
        st.session_state.soap_result = None

    service = SoapParserService()
    input_mode = st.radio("Input Type", ["Paste Text", "PDF Upload"], horizontal=True)
    source_text = ""

    if input_mode == "PDF Upload":
        uploaded_file = st.file_uploader("Upload SOAP PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                with st.spinner("Reading PDF..."):
                    source_text = service.engine.extract_pdf_text(uploaded_file)
                    st.text_area("Extracted PDF Text", source_text, height=220)
            except RuntimeError as exc:
                st.error(str(exc))
    else:
        source_text = st.text_area(
            "Paste SOAP / clinical note",
            height=320,
            placeholder="Paste the full SOAP or clinical note here...",
        )

    if st.button("Generate Plan from SOAP", type="primary"):
        if source_text.strip():
            st.session_state.soap_result = service.parse_and_build_plan(source_text)
        else:
            st.warning("Add SOAP text first.")

    if st.session_state.soap_result:
        result = st.session_state.soap_result
        plan = result.get("plan", {})
        parsed = result.get("parsed", {})
        profile = result.get("profile", {})

        plan_tab, clinical_tab = st.tabs(["View Workout Plan", "Clinical Data"])
        with plan_tab:
            render_plan(
                plan,
                json_download_name=build_download_name("soap_workout_plan"),
                user_profile=profile,
            )

        with clinical_tab:
            st.subheader("Structured Vitals")
            st.table(parsed.get("structured_vitals", {}))
            st.subheader("Inferred Profile")
            st.json(profile)
            st.subheader("Parsed SOAP Sections")
            st.code(json.dumps(parsed, indent=2), language="json")
