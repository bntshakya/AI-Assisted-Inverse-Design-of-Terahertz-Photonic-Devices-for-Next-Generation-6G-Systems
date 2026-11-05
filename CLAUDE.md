# CLAUDE.md — MTP: AI-Assisted Inverse Design of THz Valley Photonic Crystals (6G)

Context handoff from a Claude (claude.ai) session on 2026-10-26. Read this first.
Full research plan (literature, ranked ideas, roadmap): https://claude.ai/artifact/KKNG923wXQXysDpfoLD4Jz

## Deadline
MTP evaluation in the first week of November 2026 (about 10 days from 2026-10-26).
Priority = the "10-day plan" below. Bigger ideas are next-semester material.

## The device
- Silicon valley photonic crystal (VPC), rhombic/honeycomb lattice, a = 242.5 um
  (same platform as Yang et al., Nature Photonics 2020: gap ~0.32-0.35 THz).
- Two holes per unit cell (VPC-A / VPC-B). Each hole = triangle with rounded corners:
  L = straight (flat-edge) length, r = corner arc radius, r1 = 2*L1/(3*sqrt(3)), r2 = 2*L2/(3*sqrt(3)).
  So r is NOT a free parameter today; it scales with L.
- L1 != L2 breaks inversion symmetry and opens a valley-Hall bandgap; the gap grows with |L1 - L2|.
- COMSOL 6.4 model is 2D (CSV header "Dimension 2"), eigenfrequency sweep along the k-path, 2 bands.

## Data (data/comsol, gitignored)
- Raw CSVs (~4.2 GB and ~4.5 GB): one row per MESH NODE (~29.5M rows), columns
  X, Y, L1, L2, k, lambda (complex), emw.freq (GHz). Each frequency is repeated ~820 times (once per node).
  NEVER load a whole CSV into memory; use chunks (see src/utils/process_data.py).
  1. "parameter_sweep_values (Varied L1, L2 and r = 2Lby 3sqrt(3)).csv" = rounded corners, 9 header lines
  2. "r1=r2=0 ; varied L1 & L2 (Parametric sweep).csv" = sharp corners, 2 header lines
  Both: same 8 x 8 grid, 281 k-points (0.1..2.9), exactly 2 eigenfrequencies per (geometry, k).
- src/utils/process_bands.py reduces BOTH files (seconds to reload afterwards) to:
  data/comsol/bands.csv (corner, L1_um, L2_um, k, f1_GHz, f2_GHz; 35,968 rows) and
  data/comsol/band_edges.csv (corner, L1_um, L2_um, f_lower, f_upper, gap; 128 rows). corner: 1 rounded, 0 sharp.
  Gapped (gap > 0.1 GHz): 61 rounded, 63 sharp (sharp L1=L2=140 has 0.001 GHz). Sharp gap 3.0-40.2 GHz.
  Rounded band edges match train/val/test.pt to 1.5e-5 GHz. Use these CSVs for new work.
- k-path in the model: Gamma=0, M=1, K=2, Gamma=3. Band 1 max is at K (k=2) in all 128; band 2 min is at M (k=1)
  for rounded (63 of 64) but at K for sharp (63 of 64).
  Using only k = 1 and 2 reproduces every band edge exactly (all 128 geometries, both corner types).
- New sweeps: src/comsol/ (README.md = step-by-step). make_samples.py -> Latin-hypercube geometries +
  Parametric Sweep load file; run_mph.py (Python/MPh, resumable, sharded) or parse_table.py (GUI
  Global Evaluation export) -> data/comsol/new/band_edges_new.csv. Not yet run against real COMSOL.
- L1, L2 are stored in METRES (e.g. 1.4e-4) even though the header says um. Plots wrongly label them um.
- Sweep: L1 = 140..189 um, L2 = 91..140 um, 7 um steps (8 x 8 = 64). 61 have a gap (L1=140 with L2>=126 do not).
- process_data.py -> per geometry: f_lower = max over k of band 1, f_upper = min over k of band 2.
  train/val/test.pt = 42/9/10 rows; norm_stats.pt from train only. Gap range 0.57-43.6 GHz.
- op.pt (180 MB) = output of the OLD pipeline (process_data_old.py, raw rows with freq). Obsolete.

## Code
- src/models/forward_model.py   MLP 2->128->256->128->2 with BatchNorm (L1,L2 -> f_lower,f_upper)
- src/models/backward_model.py  MLP 2->256->512->512->256->2 with BatchNorm
- src/training/train_forward.py, train_tandem.py (loss = EM MSE + 0.01 * geometry MSE, Adam 1e-3, batch 8, 5000 epochs)
- src/training/evaluate_tandem.py -> results/tandem_evaluation.png
- src/utils/inference.py (target band edges -> L1, L2)
- src/benchmark/ (data.py, methods.py, run.py): direct inverse NN vs tandem NN vs GP + least-squares
  inverse, tasks "edges" (one-to-one) and "gap" (many-to-one), 10 random splits, corner type as a fixed
  condition input; scored by a per-corner GP oracle (stand-in for COMSOL). Output: results/benchmark.json.
- main.py, config/config.yaml, generate_ppt.py are EMPTY.
- Run on Windows with the project's .venv (GPU): `.venv\Scripts\activate` then e.g. `python src/training/train_forward.py`.
  Scripts use relative paths: run from the Project folder.

## Option B: freeform C3 hole shapes (src/shapes, built 2026-10-28)
Study guide: docs/OptionB_Study_Guide.pdf (built by src/shapes/build_guide.py). README: src/shapes/README.md.
- Hole geometry of the old data (checked against slides + simulation): rounded = triangle of side L intersected with
  a disk of radius rho = 2L/(3 sqrt3) centred on the hole (NOT a fillet); sharp = triangle of side L. Flat sides face
  the 3 neighbouring holes; corners point into hexagon centres. Sites A=(a1+a2)/3, B=2(a1+a2)/3, a1=a(1,0), a2=a(1/2,sqrt3/2).
- Any C3 hole = r(theta) = r0 + sum c_n cos n theta + s_n sin n theta, n = 3,6,9,12 (local frame: theta=0 towards a
  neighbour of the other type). 9 numbers/hole, 18/cell = the shape-agnostic ML input (geometry.FEATURE_NAMES).
- pwe.py: 2D plane-wave solver (TE). With eps_Si = 8.914 (fitted; slides say 11.7 -> COMSOL 2D model probably uses a slab
  effective index n~2.99, user to confirm) it reproduces all 128 COMSOL cells: band-edge MAE 0.2-0.9 GHz, gap corr 0.997-0.9999.
- M/K shortcut verified (check_mk_shortcut.py -> results/optionb/mk_check.json): 128 COMSOL cells (full path), 658 mixed
  PWE cells + 80 circle pairs R 20-95 um (dense path + 2D irreducible wedge): every edge at M or K within 0.003 GHz.
  Not a theorem: cite Harrison et al. J. Phys. A 40, 7597 (2007), Craster et al. PRB 86 (2012), Maurin et al. IJSS 135,
  26 (2018); physics + circle caveat (band 2 can dip between K and M for circles in 3D slabs): Shalaev et al. Nat. Nanotech.
  14, 31 (2019), who also use a 2D COMSOL model with n_eff = 2.965 (supports our fitted n = 2.986). Guide section 6.4.
- Data: data/pwe/fourier.csv (4000 cells, both holes Fourier, harmonics 3,6) and library.csv (2000, each hole circle/
  rounded/sharp/fourier with rotation). Validity: wall >= 25 um, r >= 10 um, corner radius >= 8 um.
- run_optionb.py: NN ensemble surrogate (18 feats -> edges) vs GP learning curve; multi-fidelity GP on COMSOL - PWE;
  inverse in 10-D Fourier space: tandem, CVAE, multi-start gradient, CVAE+refine; every design checked + PWE-verified.
  Results: results/optionb/results.json; COMSOL check file data/pwe/showcase_for_comsol.txt.
- New COMSOL runs (added 2026-10-28, per-mesh-node CSVs at k = 1, 2 only, 9 '%' header lines, ~5,500 rows per solution):
  * "r=a_divby(3sqrt(3))_(Parametric sweep at k=1,2).csv" = round cell, two circles, r1 = 46.67-60.67 um (hole A),
    r2 = 32.67-46.67 um (hole B), 7 x 7 grid, 2.333 um steps (= 1.0-1.3 and 0.7-1.0 x a/(3 sqrt3)). The "k=1" and "k=2"
    files are exact halves of it. Imported with add_comsol_dataset.py -> data/comsol/extra_circle.csv (49 cells, 33 with
    gap, up to 28.3 GHz; equal circles: -10.7 GHz, Dirac at K 326.8 but band 2 at M 316.1). Band 1 top at K and band 2
    bottom at M in all 49. PWE (eps fitted on triangles only) MAE 0.92 / 0.39 GHz, gap MAE 0.53: calibration transfers.
  * "parameter_sweep_values (...) (k=1,2).csv" = same 64 rounded cells again, finer mesh. Matches the 281-k sweep within
    0.03 GHz (gap 0.012): k = 1, 2 route and old mesh both fine. NOT added as data (duplicates). check_comsol_rerun.py.
- Multi-fidelity with 3 families (157 gapped COMSOL cells): 10-fold CV f_lower MAE 0.89 -> 0.09 GHz, gap 0.90 -> 0.29.
  Leave-family-out: helps for circles (0.92 -> 0.13) but hurts rounded (0.88 -> 1.13) and sharp (0.87 -> 1.76).
- Repeatability: run_optionb.py --seed N --out results/optionb/seeds/seedN --no-showcase (seed 0 = main run);
  summarize_seeds.py -> results/optionb/seed_summary.json (mean ± std); both guides read it.
- Beginner material: docs/Learn_From_Zero.pdf (learn/build_learn.py) + 11 runnable exercises in learn/ (README there).
  The user is new to the physics and ML: explain from first principles, worked numbers, step by step.
- COMSOL freeform runs (2026-10-29): user built Fourier holes (Parametric Curve + Convert to Solid, Air via its
  selection) in the circle model; steps in data/comsol/new/FOURIER_COMSOL_STEPS.txt. Global Evaluation tables
  (no X, Y) imported with src/comsol/import_global_eval.py --family fourier -> data/comsol/extra_fourier.csv.
  The rhombic COMSOL cell puts each hole centre 70 um from two cell edges: holes reaching further are cut off in
  COMSOL (not periodic) -> wrong. The importer drops them; make_fourier_samples.py --max-reach 68 avoids them.
  Batch 1: 97 valid cells, PWE vs COMSOL MAE 0.48 / 0.50 GHz (max 1.3); circle checks within 0.013 GHz.
  Gallery designs 0, 1 in COMSOL: 299.10/330.25 and 299.11/330.06 GHz vs target 300/330 (first COMSOL verification).
  23 of 24 gallery designs fit inside the cell. Batches 2-5 regenerated (all inside the cell).
- Rounded LHS triangles at k = 1, 2 (batch 1 of 5): data/comsol/extra_rounded_lhs.csv (200 cells), Normal mesh;
  sits 0.07 / 0.12 GHz below the old grid (mesh), check cells within 0.13 GHz.
- Add further sweeps the same way: python src/shapes/add_comsol_dataset.py "<csv>" --family circle|rounded|sharp

## History (why the design changed)
An earlier version (from a Gemini session) predicted a single "freq" from (L1, L2) using every raw row.
That target is not a function: each geometry has 2 bands at every k (e.g. ~28 GHz and ~490 GHz at k=0.1),
so the model learned an average and inference was off by ~20 GHz. Switching to band edges was correct.

## Measured in review (same 42 train rows, 10 test rows, physical units)
| Model | Forward MAE [f_lower, f_upper] GHz | Inverse geometry MAE [L1, L2] um |
| --- | --- | --- |
| Current NN / tandem | 1.15, 2.05 (max 3.4, 4.1) | 3.83, 2.55 (max 12.3, 6.5) |
| Quadratic fit | 0.046, 0.089 | 0.27, 0.15 |
| Cubic polynomial | 0.006, 0.006 | 0.065, 0.023 |
| Gaussian process (RBF + white, normalized inputs) | 0.009, 0.007 | 0.045, 0.011 |
- Tandem's NN-reconstructed f_upper is off by up to 12.6 GHz. The slide claim "almost perfectly on y=x" is false.
- Jacobian determinant of the fitted forward map is negative everywhere (-205 to -80): the 2-param map is
  ONE-TO-ONE. Non-uniqueness (the tandem's motivation) does not exist until more parameters are added.

## Known bugs / weaknesses (fix first)
1. evaluate.py calls get_dataloaders("data/comsol/op.pt") -> should be data_dir="data/comsol".
2. Units: convert L to um once in processing; fix labels.
3. Designs are only "verified" by the forward NN; none re-simulated in COMSOL.
4. Test set lies on the training grid (interpolation only); add off-grid test points.
5. BatchNorm with batch 8 on 42 rows; no seeds; no LR schedule; model selected on 9 val rows -> use k-fold CV.
6. inference.py accepts infeasible targets and silently extrapolates -> feasibility check + uncertainty.
7. COMSOL export per mesh node -> export a Global Evaluation table (L1, L2, k, band, freq) instead.
8. Band-edge logic assumes exactly 2 eigenvalues per k.
9. data/comsol/exports/ai_predicted_params.txt is from the old pipeline.
10. __pycache__ and model weights are committed; empty main.py/config.

## 10-day plan (no new COMSOL sweep needed)
- A. Honest baselines (GP + cubic vs NN) in GHz/um; re-simulate 5-10 predicted off-grid designs in COMSOL.
- B. Uncertainty-aware inverse: GP returns geometry +- std; flag unreachable targets.
- C. Retrospective active learning: start from 8 of 61 geometries, pick next by GP uncertainty vs random; plot error vs #simulations.
- D. Fabrication-tolerance map (GHz shift per um of L error, from GP gradients) + design map vs
  IEEE 802.15.3d band 252.72-321.84 GHz; scale invariance (2D, non-dispersive Si): scaling a by s divides all
  frequencies by s -> retarget channels; verify with 1-2 COMSOL runs.
- E. (if time) Process the sharp-corner CSV; transfer learning rounded <-> sharp.
- Then: slides (problem, audit, A-D results, next-semester plan).

## Next-semester ideas (see the doc for details and references)
2 expand design space (free r/L, rotation, hole shape) + scale/inversion-symmetry priors (L1<->L2 swap should
keep bulk bands, flip valley Chern sign: verify first) · 3 active learning with COMSOL in the loop via MPh (Python API)
· 4 topology-aware multi-objective design (gap width vs valley Chern number, cf. Satra et al. 2026) · 5 generative
inverse (MDN / conditional INN / guided diffusion, cf. Zhang et al. 2026) · 6 full-dispersion surrogate (POD+MLP)
· 7 fabrication-aware / multi-fidelity · 8 edge-state/interface design · 9 LLM agent driving COMSOL.

## Working rules
- Report errors in GHz and um, never only normalized loss.
- Every new model must beat the GP and cubic baselines.
- Do not edit or commit the raw CSVs; data/comsol is gitignored.
- Ask before long COMSOL runs; the user runs COMSOL manually on Windows.

- PWE vs COMSOL training (compare_training.py, 97 freeform COMSOL cells, 5-fold x 3 seeds, 2026-10-30): PWE solver 0.51/0.40 GHz, NN-on-PWE 0.56/0.33, COMSOL-only best 2.6/1.0 (GP freeform), PWE+GP correction 0.31-0.34/0.29-0.36. Correction works even with 0 freeform COMSOL cells (learned from triangles/circles). Verdict: keep PWE + COMSOL correction.
- PWE convergence (check_pwe_convergence.py, 225 COMSOL cells, eps 8.914; results/optionb/pwe_convergence.json): 177/301/439/693/1011
  plane waves -> mean change vs finest 1.34/0.74/0.47/0.21/0 GHz (f_lower). Error vs COMSOL about equal at 439 and 1011
  (eps calibrated at 439 absorbs the shift). 439 = 0.22 s/cell, 1011 = 1.0 s/cell. Added to site ch 8 + PWE cheat sheet.
- AI designs for COMSOL verification (make_verify_designs.py): data/comsol/new/sweep_verify_designs.txt, 14 cells =
  2 gallery checks + 3 designs each for 300/330, 305/325, 310/340, 320/335 (PWE within 0.25 GHz, reach <= 68 um).
  After COMSOL: export global_eval_verify_designs.csv -> update_all.py (imports as extra_verify.csv, runs compare_verify.py).
- Instant design (2026-11-02): src/shapes/save_models.py trains + saves fwd ensemble/tandem/CVAE + COMSOL correction GP to
  models/optionb/ (bundle.pt, correction.pkl); src/shapes/designer.py = shared helpers; design.py (CLI) and app.py
  (Streamlit, installed in .venv) at Project root; update_all.py = import COMSOL exports + rebuild everything.
  HOW_TO_RUN.md. Study website: learn/site (published artifact Jma82mDR7wCz1joz4W7KP6), 22 chapters.
- Offline website: learn/make_offline_site.py packs learn/site (html + web_data.json + pngs) into one file,
  docs/Valley_Crystal_Study_Lab.html (3.7 MB, double-click, no server). update_all.py runs it after export_web_models.
- data/comsol/FILES_FOR_COMSOL/: one MERGED sweep file per model (sweep_verify_designs, sweep_freeform_ALL 910, sweep_circles_ALL 1003, sweep_sharp_ALL 1003, sweep_rounded_ALL 800; split batches in subfolders) + README_FIRST.txt; exports named global_eval_<fam>_batch_ALL.csv. Exports still go to data/comsol/new/ (update_all.py reads there).
- COMSOL verification of AI designs (run 0, 2026-11-03; data/comsol/extra_verify.csv, results/optionb/comsol_verify.csv):
  2 checks reproduced exactly (299.10/330.25, 299.11/330.06). 12 new designs (targets 300/330, 305/325, 310/340, 320/335):
  10 of 12 within 1 GHz in COMSOL, worst 1.05, mean worst-edge error 0.74 GHz. Errors are systematic (COMSOL f_lower
  0.5-1 GHz below PWE). On these unseen designs: PWE vs COMSOL MAE 0.74/0.21; network 0.72/0.27; network + COMSOL
  correction 0.23/0.30 GHz -> designing with the correction (design.py --comsol) should close the f_lower offset.
- Run 1 (2026-11-03): 910 freeform cells (sweep_freeform_ALL) imported with batch 1 -> extra_fourier.csv now 1,007 valid
  freeform COMSOL cells (299 gapped). PWE - COMSOL MAE 0.44 / 0.53 GHz (same as with 97). Backup of the 97-cell file:
  data/comsol/extra_fourier_batch1_only.csv.bak. Not yet retrained / compare_training not rerun.
- Final COMSOL runs checked (2026-11-03): sharp_ALL (1003) = real sharp triangles (checks 0.13-0.43 GHz below old grid,
  mesh; PWE-sharp MAE 0.54/0.48, PWE-rounded 15-19 GHz). rounded_ALL (800) = real rounded (PWE-rounded MAE 0.90/0.20).
  circles run (r1/r2 in the Fourier model) was INVALID (4 distinct freqs) -> renamed BAD_*; re-run as Fourier circles:
  FILES_FOR_COMSOL/2_circles (fourier_holes model)/sweep_circles_ALL_fourier_model.txt -> global_eval_circles_fourier_model.csv.
- FINAL PIPELINE (2026-11-04): src/shapes/final_comsol_study.py (all COMSOL data; fixed-kernel GPs; writes final_comsol.json
  + comsol_vs_pwe.json) -> save_models.py (adds COMSOL-trained ensemble + LightGP correction) -> make_final_numbers.py
  (Report/final/numbers.tex, results/optionb/final_numbers.json, FINAL_NUMBERS.md) -> make_final_figures.py -> make_poster.py
  -> compile Report/final/254161023_MTP1_Report.tex (official NeurIPS template; local nicefrac.sty stand-in) -> guide,
  cheat sheet -> learn/patch_site_numbers.py -> make_offline_site.py. update_all.py runs all of it (~1 h).
  Streamlit website: app.py + app_common.py + app_pages/ (home, learn, inverse, forward, results, accuracy, data, about);
  designer.py models: pwe | corr | comsol.
- FINAL RESULTS (2026-11-05, ALL DATA: 4,198 COMSOL cells, 3,281 gapped; circles re-run imported as
  data/comsol/extra_circle_lhs.csv, 1,000 new + 3 checks within 0.023 GHz): PWE vs COMSOL 0.79/0.34 (3,267 gapped);
  correction 10-fold all families 0.14/0.23 (circles 0.03/0.03). Freeform (1,007): solver 0.45/0.43, nn_pwe+corr 0.21/0.30
  (best, mean 0.25), solver+corr 0.15/0.46, nn_comsol_free 0.32/0.23 (best COMSOL-only, mean 0.27). Circles made the freeform
  f_upper correction worse (was 0.20): PWE f_upper is +0.40 GHz for circles but -0.50 for freeform (sign flips with harmonic
  content) -> report/FINAL_NUMBERS state it ("a correction needs COMSOL cells of the shapes being designed"). Verdict wording
  in make_final_numbers.py has a 'nearly caught up' branch (margin < 0.05 GHz). Report 12 pages, conclusion p8-9.
  designer.design(model='corr') shifts the target by the mean correction; 12 corr designs (made with the 3,198-cell correction):
  data/comsol/new/sweep_verify_corrected_designs.txt (FILES_FOR_COMSOL/5_OPTIONAL...). Raw circle export kept at
  data/comsol/sweep_circles_ALL_fourier.csv (copied to new/global_eval_circles_fourier_model.csv for update_all.py).
