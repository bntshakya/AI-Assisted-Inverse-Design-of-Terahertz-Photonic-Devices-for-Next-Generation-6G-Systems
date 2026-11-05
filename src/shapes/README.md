# Option B: freeform C3 hole shapes

Inverse design where the hole shape is free, not only L1 and L2. The full explanation, with
the physics and the results, is in `docs/OptionB_Study_Guide.pdf`.

## Files
| File | What it does |
|---|---|
| `geometry.py` | Lattice; hole families (circle, rounded, sharp, fourier, optional rotation); the 18-number C3 Fourier descriptor; fabrication validity checks |
| `pwe.py` | 2D plane-wave-expansion band solver (TE). About 0.17 s per cell at M and K |
| `validate_pwe.py` | Fits ε against both COMSOL datasets and validates all 128 cells (`results/pwe_validation.json`) |
| `gen_dataset.py` | Samples valid cells (`fourier` or `library`), labels them with PWE, and optionally checks the M/K shortcut (`data/pwe/*.csv`) |
| `check_mk_shortcut.py` | Checks that band edges sit only at M (k = 1) and K (k = 2): COMSOL cells, mixed PWE cells, and a dense circle-pair grid (`results/optionb/mk_check.json`) |
| `dataset.py` | Loads COMSOL and PWE data into one table with the 18 features |
| `add_comsol_dataset.py` | Adds a new COMSOL sweep (for example circular holes) and checks it against PWE |
| `models.py` | Forward ensemble, validity penalty, tandem NN, conditional VAE, multi-start search |
| `run_optionb.py` | All experiments (`results/optionb/results.json`) plus the showcase designs for COMSOL |
| `make_figures.py`, `build_guide.py` | Figures and the PDF guide |

## Run order (from the Project folder)
```
python src/shapes/validate_pwe.py                       # ~1 min
python src/shapes/gen_dataset.py --kind fourier --n 4000 # ~5 min
python src/shapes/gen_dataset.py --kind library --n 2000 # ~2 min
python src/shapes/check_mk_shortcut.py                  # ~2 min
python src/shapes/run_optionb.py                        # ~15 min (GPU)
python src/shapes/make_figures.py
python src/shapes/build_guide.py                        # docs/OptionB_Study_Guide.pdf
```

## Adding a COMSOL sweep (done for the round-cell data on 2026-10-28)
```
python src/shapes/add_comsol_dataset.py "data/comsol/r=a_divby(3sqrt(3))_(Parametric sweep at k=1,2).csv" --family circle
python src/shapes/check_comsol_rerun.py "data/comsol/parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)) (k=1,2).csv" --corner 1
```
`add_comsol_dataset.py` expects the per-mesh-node layout of the COMSOL exports: X, Y, p1, p2, k, lambda, freq
(p1 = hole A, p2 = hole B). If the columns differ, give `--usecols p1 p2 k freq` (0-based column positions).
It writes `data/comsol/extra_<family>.csv` and prints the PWE-vs-COMSOL error. Then rerun
`run_optionb.py`, `make_figures.py` and `build_guide.py`.

`check_comsol_rerun.py` compares a k = 1, 2 re-run with an old full-path sweep of the same cells
(`results/optionb/comsol_rerun_check.json`). Do not add such a re-run as data: the cells are already in it.

## Caveats
- The PWE solver is calibrated to COMSOL with ε = 8.91. Final designs must be verified in COMSOL;
  see `data/pwe/showcase_for_comsol.txt`.
- A band gap alone does not prove valley topology. The valley Chern sign is not computed yet.
- The M/K shortcut is checked, not proven (section 6.4 of the guide has the references). For a new hole family,
  run a few cells on the full path first.
