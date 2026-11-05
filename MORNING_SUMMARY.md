# What happened overnight (2 Nov 2026)

Nothing you have already studied changed. Only additions:

## 1. PWE convergence check (one new fact to know)
225 COMSOL cells re-solved with 177 to 1,011 plane waves.
- At 439 waves (the project setting) the band edges are within ~0.5 GHz of the finest setting.
- The shift is the same direction for every cell (+0.47 / +0.58 GHz), so the fitted ε = 8.91 absorbs it:
  agreement with COMSOL is ~0.5 GHz with 439 waves and also with 1,011 (which are 5× slower).
- **What to say:** "439 waves are within half a GHz of converged, and the calibration absorbs the rest."
- Added to: website chapter 8, PWE cheat sheet (Q "Are 439 plane waves enough?"), report section 5.

## 2. A COMSOL file of AI designs to verify (do this first in COMSOL)
`data/comsol/new/sweep_verify_designs.txt`: 14 shapes, model `fourier_holes.mph`,
save the export as `global_eval_verify_designs.csv`.
- 2 repeats of designs you already simulated (checks) + 12 new AI designs for 4 targets
  (300/330, 305/325, 310/340, 320/335). PWE puts all 12 within 0.25 GHz of their targets.
- If COMSOL agrees, "2 designs verified" becomes "14 designs verified".

## 3. Instant designing (no more 20-minute wait)
- `python design.py 300 330` gives 4 designs, each PWE-checked, in ~6 s.
- `streamlit run app.py` opens the designer web app (choose methods, see cells, band diagrams, download COMSOL file).
- Both use networks saved tonight in `models/optionb/`.

## 4. One command after new COMSOL data
Put the exports in `data/comsol/new/` and run `python update_all.py`. Names are in `HOW_TO_RUN.md`.

## Your plan
1. COMSOL: run `sweep_verify_designs.txt` first, then freeform batch 2.
2. Study: website chapters 1, 3, 4, 6 (day 1).
3. Send Claude the exports.
