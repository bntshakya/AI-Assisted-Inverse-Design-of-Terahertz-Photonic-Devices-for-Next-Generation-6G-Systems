"""Run every exercise and rebuild docs/Learn_From_Zero.pdf (about 4 minutes).
Run from the Project folder:  python learn/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["ex01_waves.py", "ex02_fourier_series.py", "ex03_hole_shapes.py", "ex04_pwe_1d.py", "ex05_pwe_2d.py",
           "ex06_design_to_band_edges.py", "ex07_surrogate_gp_vs_nn.py", "ex08_tandem_toy.py", "ex09_cvae_toy.py",
           "ex10_multistart_toy.py", "ex11_multifidelity_toy.py", "make_concept_figs.py", "build_learn.py"]

env = dict(os.environ, PYTHONIOENCODING="utf-8")
for name in SCRIPTS:
    print(f"\n===== {name}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, name)], env=env)
    if r.returncode:
        sys.exit(f"{name} failed (exit {r.returncode})")
print("\nall done -> docs/Learn_From_Zero.pdf")
