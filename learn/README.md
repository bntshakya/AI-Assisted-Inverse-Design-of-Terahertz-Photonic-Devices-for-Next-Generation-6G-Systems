# Learning from zero: exercises

Companion scripts for `docs/Learn_From_Zero.pdf`. Each script runs on its own, prints numbers and saves a
picture in `learn/figs/`. Change the CAPITAL settings at the top of a script and run it again.

```
cd "E:\MTech\MTP project\Project"
.venv\Scripts\activate
python learn\ex01_waves.py
```

| Script | Topic | Guide part |
|---|---|---|
| `ex01_waves.py` | frequency, wavelength, why a = 242.5 µm | 2 |
| `ex02_fourier_series.py` | Fourier series of the permittivity pattern | 3 |
| `ex03_hole_shapes.py` | hole shapes as r(θ); the 3-fold rule; the old triangle's numbers | 4 |
| `ex04_pwe_1d.py` | plane-wave method in 1D vs the exact answer | 5 |
| `ex05_pwe_2d.py` | the project's 2D solver on circular holes | 6 |
| `ex06_design_to_band_edges.py` | design numbers → holes → fabrication check → band edges | 6 |
| `ex07_surrogate_gp_vs_nn.py` | Gaussian process vs neural network | 7 |
| `ex08_tandem_toy.py` | direct inverse vs tandem network | 8 |
| `ex09_cvae_toy.py` | conditional VAE: many designs per target | 8 |
| `ex10_multistart_toy.py` | multi-start search and surrogate exploitation | 8 |
| `ex11_multifidelity_toy.py` | cheap simulator + learned correction | 9 |
| `make_concept_figs.py` | diagrams used in the guide | – |
| `build_learn.py` | builds the PDF | – |
| `run_all.py` | runs everything and rebuilds the PDF (~4 min) | – |

The exercises only read the project; they never change project data or results.
