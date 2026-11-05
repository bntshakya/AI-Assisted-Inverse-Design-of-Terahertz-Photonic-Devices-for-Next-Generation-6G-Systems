# Putting the website online (optional)

The website runs locally with `streamlit run app.py`. To get a public link, use Streamlit Community Cloud (free).
You need your own GitHub account; nothing here signs in for you.

1. Create a GitHub repository and upload the `Project` folder **without** `.venv/` and the large raw COMSOL CSVs
   (`data/comsol/*.csv` files of hundreds of MB). Keep: `app.py`, `app_common.py`, `app_pages/`, `src/`,
   `models/optionb/`, `results/optionb/`, `data/pwe/`, `data/comsol/extra_*.csv`, `data/comsol/band_edges.csv`,
   `docs/Valley_Crystal_Study_Lab.html`, `requirements_app.txt`, `.streamlit/`.
2. Go to share.streamlit.io, sign in with GitHub, click **New app**, pick the repository, main file `app.py`,
   and under *Advanced settings* set the requirements file to `requirements_app.txt`.
3. Deploy. The first start takes a few minutes (it installs PyTorch).

The app then works the same as locally (designs, PWE checks, results); it runs on a CPU there, so designing takes
a few seconds longer.
