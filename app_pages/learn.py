import os

import streamlit as st
import streamlit.components.v1 as components

OFFLINE = "docs/Valley_Crystal_Study_Lab.html"


@st.cache_data(show_spinner="Loading the course…")
def course_html():
    with open(OFFLINE, encoding="utf-8") as f:
        return f.read()


def render():
    st.title("Learn the project from zero")
    st.caption("The full study course: 22 chapters from waves and band gaps to every AI method, with live demos. "
               "Use the chapter list on the left inside the frame. The same course opens on its own by double-clicking "
               f"`{OFFLINE}`.")
    if not os.path.exists(OFFLINE):
        st.warning(f"{OFFLINE} not found. Build it with:  python learn/make_offline_site.py")
        return
    components.html(course_html(), height=1100, scrolling=True)
