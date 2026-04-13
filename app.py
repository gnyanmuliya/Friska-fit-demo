import streamlit as st

from ui.shared_components import render_global_styles
from ui.soap_testing_view import render_soap_testing_view
from ui.workout_generator_view import render_workout_generator_view


st.set_page_config(
    page_title="Friska AI Fitness",
    page_icon="PT",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_global_styles()

st.markdown(
    """
    <section class="hero-header">
        <h1>Friska AI Fitness Ecosystem</h1>
        <p>Single-page Streamlit demo for local workout generation and SOAP note testing.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Demo Controls")
    mode = st.radio(
        "Select Mode",
        ["Workout Generator", "SOAP Note Testing"],
        horizontal=False,
    )
    st.markdown("---")
    st.markdown("**Project Rules**")
    st.caption(
        "Local CSV only\n"
        "\nNo API calls\n"
        "\nNo tokens\n"
        "\nVideo mapping from Exercise videos.csv"
    )

if mode == "Workout Generator":
    render_workout_generator_view()
else:
    render_soap_testing_view()
