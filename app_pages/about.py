import streamlit as st


def render():
    st.title("About this project")
    st.markdown("""
**AI-Assisted Inverse Design of Terahertz Photonic Devices for Next-Generation 6G Systems**
M.Tech. Data Science thesis project (P29), IIT Guwahati, Phase I (Semester III, 2026).
Student: Binit Shakya (254161023). Supervisors: Prof. Debabrata Sikdar and Prof. Teena Sharma.
""")
    st.subheader("Project folder")
    st.code("""Project/
├─ app.py, app_common.py, app_pages/   this website (Streamlit)
├─ design.py                           instant designs from the command line
├─ update_all.py                       after new COMSOL data: import and rebuild everything
├─ src/shapes/                         hole shapes, PWE solver, all the AI, figures, guides
├─ src/comsol/                         COMSOL sweep files and importers
├─ data/pwe/                           6,000 PWE cells
├─ data/comsol/                        all COMSOL data (raw exports + cleaned tables)
├─ models/optionb/                     the trained networks
├─ results/optionb/                    every number and figure
├─ docs/                               study guide, cheat sheet, offline course
└─ learn/                              exercises and the course website""", language="text")
    st.subheader("Commands")
    st.code("""streamlit run app.py                    # this website
python design.py 300 330                # designs for a target, in seconds
python update_all.py                    # after new COMSOL exports in data/comsol/new/""", language="bash")
    st.subheader("Methods and tools")
    st.markdown("""
- Fast solver: 2D plane-wave expansion (439 plane waves, Ho's inverse rule), calibrated against COMSOL (ε = 8.914).
- Reference simulator: COMSOL Multiphysics 6.4, 2D eigenfrequency study at k = M and K.
- AI: PyTorch neural-network ensembles, tandem network, conditional VAE, multi-start gradient search; scikit-learn
  Gaussian processes for baselines and the COMSOL correction.
- AI tools: Claude (Anthropic) was used as a coding and writing assistant; the COMSOL simulations were run by the student.
""")
