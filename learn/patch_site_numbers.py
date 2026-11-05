"""Put the final numbers into the study website's data file (learn/site/web_data.json) without retraining anything.

Adds: final_numbers (flat numbers + the 'do we need PWE' verdict), final (per-family PWE accuracy, correction,
freeform comparison, accuracy statistics), verify (the COMSOL-verified AI designs), and refreshes
results.comsol_vs_pwe. Run from the Project folder after make_final_numbers.py:
    python learn/patch_site_numbers.py
export_web_models.py calls the same function at its end.
"""
import json
import os

import pandas as pd

SITE = "learn/site/web_data.json"
R = "results/optionb"


def patch(data):
    if os.path.exists(f"{R}/final_numbers.json"):
        data["final_numbers"] = json.load(open(f"{R}/final_numbers.json"))
    if os.path.exists(f"{R}/final_comsol.json"):
        F = json.load(open(f"{R}/final_comsol.json"))
        data["final"] = {"n_cells": F["n_cells"], "n_gapped": F["n_gapped"], "families": F["families"],
                         "pwe_vs_comsol_all": F["pwe_vs_comsol_all"], "mf_cv10": F["mf_cv10"],
                         "mf_leave_family_out": F["mf_leave_family_out"],
                         "freeform_comparison": F["freeform_comparison"], "accuracy": F["accuracy"]}
    if os.path.exists(f"{R}/comsol_verify.csv"):
        data["verify"] = pd.read_csv(f"{R}/comsol_verify.csv").to_dict(orient="records")
    if os.path.exists(f"{R}/comsol_vs_pwe.json"):
        data.setdefault("results", {})["comsol_vs_pwe"] = json.load(open(f"{R}/comsol_vs_pwe.json"))
    return data


def main():
    data = json.load(open(SITE))
    data = patch(data)
    with open(SITE, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"patched {SITE} ({os.path.getsize(SITE) / 1e6:.1f} MB): "
          f"{', '.join(k for k in ('final_numbers', 'final', 'verify') if k in data)}")


if __name__ == "__main__":
    main()
