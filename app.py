"""The project website (Streamlit). Start it from the Project folder:

    streamlit run app.py

It opens in the browser. Needs models/optionb/ (made by: python src/shapes/save_models.py).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Valley Crystal Designer", page_icon="🔷", layout="wide")

import app_common  # noqa: E402,F401  (sets paths, working folder)
from app_pages import about, accuracy, data, forward, home, inverse, learn, results  # noqa: E402

try:
    app_common.bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pages = {
    "Project": [st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
                st.Page(learn.render, title="Learn (course)", icon=":material/school:", url_path="learn")],
    "Tools": [st.Page(inverse.render, title="Inverse designer", icon=":material/auto_awesome:", url_path="designer"),
              st.Page(forward.render, title="Forward predictor", icon=":material/insights:", url_path="forward")],
    "Results": [st.Page(results.render, title="Results", icon=":material/table_chart:", url_path="results"),
                st.Page(accuracy.render, title="How accuracy is measured", icon=":material/rule:", url_path="accuracy"),
                st.Page(data.render, title="Data", icon=":material/dataset:", url_path="data")],
    "Info": [st.Page(about.render, title="About", icon=":material/info:", url_path="about")],
}
st.navigation(pages).run()
