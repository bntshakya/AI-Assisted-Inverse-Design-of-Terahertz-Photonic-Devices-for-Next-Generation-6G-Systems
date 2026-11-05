"""After new COMSOL exports: import them and rebuild everything, in one command.

    python update_all.py                 # import new files + final COMSOL study + saved models + numbers, figures,
                                         # poster, report PDF, guide, cheat sheet, website data, offline site (~1 h)
    python update_all.py --only-import   # just import the COMSOL files and print the checks
    python update_all.py --reimport      # import every file again;  --with-main  also rerun run_optionb.py

Put the Global Evaluation exports in data/comsol/new/ with these names (files already imported are skipped):
    global_eval_fourier_batch*.csv          freeform (batch 1 and batch_ALL are imported together)
    global_eval_circles_fourier_model.csv   circles run as Fourier holes in fourier_holes.mph
    global_eval_circle_batch*.csv           circles from a model with r1, r2 parameters
    global_eval_sharp_batch*.csv            sharp triangles
    global_eval_rounded_batch*.csv          rounded triangles (the first 200 are already imported separately)
    global_eval_verify_designs.csv          the AI designs from sweep_verify_designs.txt
Each step's output goes to the screen; a failed step stops the script.
"""
import argparse
import os
import shutil
import glob
import subprocess
import sys
import time

PY = sys.executable
NEW = "data/comsol/new"
IMPORTS = [("global_eval_fourier_batch*.csv", "fourier", None),
           ("global_eval_circle_batch*.csv", "circle", "circle_lhs"),
           ("global_eval_circles_fourier_model*.csv", "fourier", "circle_lhs"),  # circles run as Fourier holes (c, s = 0)
           ("global_eval_sharp_batch*.csv", "sharp", "sharp_lhs"),
           ("global_eval_rounded_batch*.csv", "rounded", "rounded_lhs2"),
           ("global_eval_verify_designs.csv", "fourier", "verify"),
           ("global_eval_verify_corrected_designs.csv", "fourier", "verify_corr")]


def run(title, args):
    print(f"\n=== {title} ===\n$ {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args)
    if r.returncode:
        sys.exit(f"step failed: {title}")
    print(f"--- done in {(time.time() - t0) / 60:.1f} min", flush=True)


def output_of(fam, name):
    return f"data/comsol/extra_{name or ('fourier' if fam == 'fourier' else fam + '_lhs')}.csv"


def compile_report():
    rep = os.path.join("..", "Report", "final")
    tex = "254161023_MTP1_Report"
    for cmd in (["pdflatex", "-interaction=nonstopmode", tex + ".tex"], ["bibtex", tex],
                ["pdflatex", "-interaction=nonstopmode", tex + ".tex"], ["pdflatex", "-interaction=nonstopmode", tex + ".tex"]):
        subprocess.run(cmd, cwd=rep, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"--- report: {os.path.join(rep, tex + '.pdf')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-import", action="store_true", help="import the COMSOL files and print the checks only")
    ap.add_argument("--reimport", action="store_true", help="import every file again, even if already imported")
    ap.add_argument("--with-main", action="store_true", help="also rerun run_optionb.py (PWE-based results; ~20 min)")
    a = ap.parse_args()
    t0 = time.time()
    found = False
    for pattern, fam, name in IMPORTS:
        files = sorted(glob.glob(f"{NEW}/{pattern}"))
        if not files:
            continue
        out = output_of(fam, name)
        if not a.reimport and os.path.exists(out) and os.path.getmtime(out) > max(os.path.getmtime(f) for f in files):
            print(f"up to date: {out}")
            continue
        found = True
        args = ["src/comsol/import_global_eval.py", *files, "--family", fam] + (["--name", name] if name else [])
        run(f"import {fam} ({len(files)} file(s))", args)
    if glob.glob(f"{NEW}/global_eval_verify_designs.csv"):
        run("compare AI designs with their targets", ["src/comsol/compare_verify.py"])
    if glob.glob(f"{NEW}/global_eval_verify_corrected_designs.csv"):
        run("compare COMSOL-aimed designs with their targets", ["src/comsol/compare_verify.py", "--plan",
            "data/comsol/new/verify_corrected_designs.csv", "--got", "data/comsol/extra_verify_corr.csv",
            "--out", "results/optionb/comsol_verify_corrected.csv"])
    if not found:
        print("nothing new to import from data/comsol/new/ (see the names at the top of this file)")
    if a.only_import:
        return
    if a.with_main:
        run("main experiment (about 20 min)", ["src/shapes/run_optionb.py"])
    run("final COMSOL study (about 35 min)", ["src/shapes/final_comsol_study.py"])
    run("save networks for design.py and the app (about 12 min)", ["src/shapes/save_models.py"])
    run("final numbers", ["src/shapes/make_final_numbers.py"])
    run("final figures", ["src/shapes/make_final_figures.py"])
    run("poster", ["src/shapes/make_poster.py"])
    compile_report()
    run("study guide PDF", ["src/shapes/build_guide.py"])
    run("PWE cheat sheet", ["src/shapes/build_pwe_cheatsheet.py"])
    run("website numbers", ["learn/patch_site_numbers.py"])
    for f in ("pwe_validation.png", "inverse.png", "accuracy_cdf.png"):
        if os.path.exists(f"results/optionb/figs/{f}"):
            shutil.copy(f"results/optionb/figs/{f}", f"learn/site/{f}")
    run("offline website file", ["learn/make_offline_site.py"])
    print(f"\nall done in {(time.time() - t0) / 60:.0f} min. Report and poster: ../Report/final/. Offline website: "
          "docs/Valley_Crystal_Study_Lab.html. The online (claude.ai) copy still needs republishing by Claude.")


if __name__ == "__main__":
    main()
