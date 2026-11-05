import streamlit as st

from app_common import ACCURACY_TEXT, fmt2, numbers


def render():
    N = numbers()
    f = N["final"]
    st.title("How accuracy is measured")
    st.markdown("This project predicts **numbers** (frequencies in GHz), not categories, so there is no single "
                "'accuracy %' as in classification. Each part is scored in the way that matches what it does.")
    st.markdown(ACCURACY_TEXT)
    st.subheader("The four scores, side by side")
    st.markdown("""
| Part | Input → output | Score | How it is computed |
|---|---|---|---|
| Forward model | shape → f_lower, f_upper | mean absolute error (GHz), 95 % bound, share within 0.5 / 1 GHz | prediction vs simulation, on shapes never used in training |
| Inverse model | target → shapes | **success rate**: share of targets whose design has both edges within 1 GHz | design re-simulated with PWE, compared with the target, 100 unseen targets, 5 runs |
| Final proof | AI design → COMSOL | designs within 1 GHz of their target in COMSOL | the accurate simulator, independent of everything the AI learned |
| Fast solver | shape → PWE vs COMSOL | mean absolute error (GHz) per shape family | the same shape simulated by both |
""")
    st.subheader("Worked example")
    st.markdown("""
Target **300 / 330 GHz**. A design is re-simulated and gives **299.4 / 330.3 GHz**.

- errors: |299.4 − 300| = **0.6 GHz** and |330.3 − 330| = **0.3 GHz**
- design error = the worse of the two = **0.6 GHz** → below 1 GHz → **success**
- had it given 298.8 / 330.1, the design error would be 1.2 GHz → **miss**, even though the upper edge is almost perfect.
""")
    if f:
        a = f["accuracy"]["forward_pwe_test"]
        st.subheader("What the numbers say for this project")
        st.markdown(f"""
- Forward network: mean error **{fmt2(a['mae'])} GHz**; 95 % of predictions within **{fmt2(a['p95'])} GHz**.
- PWE vs COMSOL: mean error **{fmt2(f['pwe_vs_comsol_all']['mae'])} GHz** on {f['pwe_vs_comsol_all']['n']:,} gapped cells.
- PWE + COMSOL correction: **{fmt2(f['mf_cv10']['all']['pwe_plus_corr']['mae'])} GHz** (10-fold cross-validation).
""")
    st.subheader("What it means in the app")
    st.markdown("""
- **Forward predictor:** each model shows its *typical error* and *95 % bound* (measured once on unseen shapes) and how much
  its networks disagree for your shape. Press **Check with PWE** to get the true error for that shape.
- **Inverse designer:** every design shows its **PWE error**; "within 1 GHz = yes" is exactly the success criterion of the report.
""")
