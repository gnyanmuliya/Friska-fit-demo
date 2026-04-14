import asyncio
import json

import streamlit as st

from core.fitness import ClinicalExtractionTool, PrescriptionParserTool


st.set_page_config(page_title="Friska Coach | Clinical Intake", page_icon="PT", layout="wide")

st.title("SOAP Note integration Pipeline")
st.markdown("Extract clinical data from a PDF or pasted SOAP note and generate a structured JSON workout plan.")


def _safe_weight_kg(weight_value: str) -> str:
    raw = str(weight_value or "").strip()
    if not raw or raw == "N/A":
        return "70"
    try:
        value = float(raw.split()[0])
    except Exception:
        return "70"
    if "lb" in raw.lower():
        value = value * 0.45359237
    return str(round(value, 1))


def _build_user_profile(extracted_data: dict) -> dict:
    vitals = extracted_data.get("structured_vitals") or {}
    return {
        "weight_kg": _safe_weight_kg(vitals.get("Weight", "70")),
        "blood_pressure": vitals.get("BP", "N/A"),
        "bp": vitals.get("BP", "N/A"),
        "age": 49,
        "primary_goal": "Diabetes Management",
    }


tool = ClinicalExtractionTool()
input_mode = st.radio("Input Type", ["PDF Upload", "Paste Text"], horizontal=True)

raw_text = ""
extracted_data = None

if input_mode == "PDF Upload":
    uploaded_file = st.file_uploader("Upload Patient PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Extracting clinical data..."):
            raw_text = tool.extract_raw_text(uploaded_file)
            extracted_data = tool.parse_sections(raw_text)
else:
    raw_text = st.text_area("Paste SOAP / clinical note", height=320, placeholder="Paste the full clinical note here...")
    if raw_text.strip():
        with st.spinner("Extracting clinical data..."):
            extracted_data = tool.parse_sections(raw_text)

if extracted_data:
    prescription_narrative = f"""
    History: {extracted_data.get('history', 'Not found')}
    Findings: {extracted_data.get('findings', 'Not found')}
    Plan: {extracted_data.get('plan_of_action', 'Not found')}
    Session: {extracted_data.get('exercise_session', 'Not found')}
    Vitals: {json.dumps(extracted_data.get('structured_vitals', {}))}
    """

    user_profile = _build_user_profile(extracted_data)

    st.divider()
    if st.button("Generate Structured Workout Plan"):
        parser = PrescriptionParserTool()
        result = asyncio.run(parser.execute(prescription_narrative, user_profile))

        if result.success:
            st.session_state.plans = result.data.get("plans_json", {})
            st.session_state.vitals = extracted_data.get("structured_vitals", {})
            st.session_state.profile = result.data.get("profile", user_profile)
            st.session_state.narrative = prescription_narrative
            st.success("Plan generated successfully.")
        else:
            st.error(f"Error: {result.error}")

    if "plans" in st.session_state:
        tab1, tab2 = st.tabs(["View Workout Plan", "Clinical Data"])

        with tab1:
            for day, day_data in st.session_state.plans.items():
                with st.expander(f"{day} - {day_data.get('main_workout_category', 'Workout Session')}", expanded=True):
                    def render_simple_section(exercises, title):
                        if not exercises:
                            return
                        st.markdown(f"#### {title}")
                        for ex in exercises:
                            st.markdown(f"**{ex['name']}**")
                            st.caption(
                                f"Sets: {ex['sets']} | Reps: {ex['reps']} | "
                                f"RPE: {ex.get('intensity_rpe', 'N/A')} | Rest: {ex.get('rest', 'None')}"
                            )
                            st.write(f"*Benefit:* {ex.get('benefit', 'N/A')}")
                            if ex.get("equipment"):
                                st.write(f"*Equipment:* {ex['equipment']}")
                            if ex.get("steps"):
                                with st.expander("Steps to perform"):
                                    for step in ex["steps"]:
                                        st.write(f"- {step}")
                            st.divider()

                    render_simple_section(day_data.get("warmup", []), "Warmup")
                    render_simple_section(day_data.get("main_workout", []), "Main Workout")
                    render_simple_section(day_data.get("cooldown", []), "Cooldown")

            st.divider()
            final_json = json.dumps(st.session_state.plans, indent=4)
            age_value = st.session_state.get("profile", {}).get("age", user_profile["age"])
            st.download_button(
                label="Download Plan as JSON",
                data=final_json,
                file_name=f"workout_plan_{age_value}_yo.json",
                mime="application/json",
            )

        with tab2:
            st.subheader("Extracted Patient Vitals")
            st.table(st.session_state.get("vitals", {}))
            st.subheader("Parser Profile")
            st.json(st.session_state.get("profile", user_profile))
            st.subheader("Raw Narrative Context")
            st.info(st.session_state.get("narrative", prescription_narrative))
else:
    st.info("Upload a PDF or paste a note to generate the plan.")
