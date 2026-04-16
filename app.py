import streamlit as st

from ui.shared_components import render_global_styles
from ui.soap_testing_view import render_soap_testing_view
from ui.workout_generator_view import render_workout_generator_view


st.set_page_config(
    page_title="Friska AI Fitness",
    page_icon="PT",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_global_styles()

st.markdown(
    """
    <header class="topbar">
        <div>
            <p class="topbar-eyebrow">Friska Wellness Demo</p>
            <h1>Friska AI</h1>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Menu",
    ["Fitness Plan Generator", "SOAP Note Based Plan"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Fitness Plan Generator":
    render_workout_generator_view()
else:
    render_soap_testing_view()
