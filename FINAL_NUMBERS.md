# Final numbers (read this before the evaluation)

Generated from the results by `src/shapes/make_final_numbers.py`. All errors in GHz, always on shapes the model never saw.

## The five sentences
1. "I want a silicon valley photonic crystal whose band gap sits at a chosen 6G frequency (around 0.3 THz)."
2. "Each hole is described by 5 Fourier numbers, so the AI can create any three-fold shape, not only triangles."
3. "A fast simulator (PWE) made 6,000 training examples; on 3,267 COMSOL cells it agrees to 0.79 / 0.34 GHz,
   and with a correction learned from COMSOL to 0.14 / 0.23 GHz."
4. "The AI turns a target gap into hole shapes: the tandem network hits both edges within 1 GHz for 99% of unseen
   targets, and CVAE + refine gives about 8 different designs per target at 97%."
5. "I simulated 14 AI designs in COMSOL for four targets: 12 hit their target within 1 GHz
   (worst 1.05 GHz)."

## How accuracy is measured (say it like this)
- Forward model (shape -> band edges): mean absolute error in GHz on unseen shapes, plus the 95 % bound.
- Inverse model (target -> shape): every design is re-simulated; its error is the worse of its two edges;
  success = below 1 GHz. Success rate = share of 100 unseen targets that succeed.
- Why GHz: percentages hide real errors at 300 GHz. Why the worse edge: a gap needs both edges right.
  Why 1 GHz: about 0.3 % of 300 GHz, below half the narrowest 6G channel (2.16 GHz), about the PWE-COMSOL difference.

## All final numbers
| What | Final | Before (if it changed) |
|---|---|---|
| COMSOL cells simulated | 4,198 (3,281 with a gap) | 474 |
| Freeform COMSOL cells | 1,007 | 97 |
| PWE vs COMSOL, all gapped cells | 0.79 / 0.34 GHz | ~0.5 GHz (on 274 cells) |
| PWE + COMSOL correction (10-fold CV, all families) | 0.14 / 0.23 GHz | |
| Freeform: PWE solver alone | 0.45 / 0.43 GHz | 0.48 / 0.5 |
| Freeform: best PWE + correction | 0.21 / 0.30 GHz | 0.31 / 0.36 |
| Freeform: best COMSOL-only model | 0.32 / 0.23 GHz (Network, COMSOL freeform only) | 2.61 / 0.98 |
| Forward network (600 unseen PWE cells) | 0.13 / 0.17 GHz; 95 % within 0.41 / 0.62 | unchanged |
| Tandem success (both edges < 1 GHz) | 99 ± 1 % | unchanged |
| CVAE + refine success, designs per target | 97 ± 1 %, ~8 | unchanged |
| AI designs verified in COMSOL | 12 of 14 within 1 GHz (worst 1.05) | 2 designs |

## The answer to "do we still need PWE?"
With 1,007 freeform COMSOL cells, a model trained on COMSOL data alone has nearly caught up (a network on the freeform COMSOL cells: 0.27 GHz, mean of both edges, down from 1.79 GHz with 97 cells); PWE plus correction is still slightly more accurate (0.25 GHz) and needs no freeform COMSOL cells to start (0.36 GHz with none). PWE is the cheap route when COMSOL runs are scarce; with about 1,000 runs of the shapes being designed, COMSOL data alone become a real alternative.

Why the 1,000 new circles did not help the freeform shapes: the PWE error at the upper band edge depends on the shape. PWE's upper edge is 0.40 GHz too high for circles but 0.50 GHz too low for freeform holes. The circles make the correction almost perfect for circles, but they pull its freeform upper edge the wrong way. Lesson: a correction needs COMSOL cells of the shapes being designed.

## The three forward models (in the app and design.py)
- **PWE-trained network**: learned from 6,000 PWE cells; answers like PWE (0.13 / 0.17 GHz from PWE).
- **PWE network + COMSOL correction**: adds the learned COMSOL - PWE difference; answers like COMSOL.
- **COMSOL-trained network**: learned directly from the COMSOL cells; answers like COMSOL, only for shapes like those simulated.
Designs made with the correction land within about 0.2 GHz of the target in the COMSOL estimate (12 such designs are in
`data/comsol/FILES_FOR_COMSOL/5_OPTIONAL_verify_corrected_designs`).

## Where everything is
- Report: `Report/final/254161023_MTP1_Report.pdf` (fill in the red items: supervisor departments, emails, repository link)
- Poster: `Report/final/254161023_MTP1_Poster.pdf`
- Website: `streamlit run app.py` (in the Project folder); course online and offline (`docs/Valley_Crystal_Study_Lab.html`)
- Study material: `docs/PWE_Cheat_Sheet.pdf`, `docs/OptionB_Study_Guide.pdf`
