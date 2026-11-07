# How to run the project

Open PowerShell in this folder (`E:\MTech\MTP project\Project`) and turn on the project's Python once per terminal:

```
.venv\Scripts\Activate.ps1
```

(If PowerShell refuses, write `.venv\Scripts\python.exe` instead of `python` below.)

## The commands you will use

| What | Command | Time |
|---|---|---|
| **The project website** (opens in the browser) | `streamlit run app.py` | starts in seconds |
| Designs for a target, instantly | `python design.py 300 330` | seconds |
| Designs aimed at COMSOL | `python design.py 300 330 --model corr` | seconds |
| Compare all four methods | `python design.py 310 340 --method all --n 3` | ~10 s |
| After new COMSOL exports: import + rebuild everything | `python update_all.py` | ~1 h |

The website has these pages: **Home**, **Learn** (the 23-chapter course), **Inverse designer** (target → shapes, with a
choice of model, PWE checks, band diagrams and a COMSOL file download), **Forward predictor** (shape → band edges from
three models, with their accuracy), **Results**, **How accuracy is measured**, **Data**, **About**.

`design.py` options: `--model pwe | corr | comsol` (which forward model picks the designs), `--allow-big` (allow holes
that cross the COMSOL cell edge), `--no-verify` (skip the PWE check), `--seed 1` (other samples). Outputs go to
`results/designs/` (picture, CSV, COMSOL sweep file).

## Final documents
- Report: `..\Report\final\254161023_MTP1_Report.pdf` (official template; source `254161023_MTP1_Report.tex`)
- Poster: `..\Report\final\254161023_MTP1_Poster.pdf`; three design options in `..\Report\final\poster_options\`
  (A classic, B story, C visual; `compare_posters.png` shows them side by side)
- Presentation: `..\Report\final\254161023_MTP1_Presentation.pptx` (25 slides + backup slides, speaker notes;
  built by `..\Report\final\slides\build_deck.js`)
- One-page summary of every final number: `FINAL_NUMBERS.md`; reference check: `..\Report\final\REFERENCE_CHECK.md`
- Study course offline (double-click): `docs\Valley_Crystal_Study_Lab.html` (30 chapters, incl. network diagrams,
  why-these-choices, why-these-metrics, viva Q&A with flashcards, formula sheet, interview guide, viva-day plan);
  guide: `docs\OptionB_Study_Guide.pdf`; PWE cheat sheet: `docs\PWE_Cheat_Sheet.pdf`
- Viva and career PDFs in `docs\`: `Viva_QA.pdf`, `Viva_Card.pdf` (one page), `Formula_Sheet.pdf`,
  `Architecture_and_Choices.pdf`, `Presenter_Script.pdf`, `Interview_Guide.pdf`

## Viva material and architecture test
- `python src/shapes/ablation_architecture.py` (about 1.5 h): architecture test on the validation split ->
  `results/optionb/ablation.json`, `results/optionb/figs/ablation.png`
- `python src/shapes/make_arch_figures.py`: network diagrams (`results/optionb/figs/arch_*.png`)
- `python learn/build_extras.py`: Q&A / interview / formulas -> study-site data, the viva PDFs and the report's
  appendix table `..\Report\final\ablation.tex`. The text lives in `learn/viva_content.py`.
- `python src/shapes/make_poster_variants.py`: poster options B and C

## New COMSOL data

Put the Global Evaluation exports in `data\comsol\new\` (names in `data\comsol\FILES_FOR_COMSOL\README_FIRST.txt`),
then run `python update_all.py`. It imports only new files, reruns the final COMSOL study, saves the models, and
regenerates the numbers, figures, poster, report PDF, guide, cheat sheet and the offline course. (`--only-import`
just imports and shows the checks.) The online copy of the course on claude.ai is republished by Claude.

## Putting the website online
See `DEPLOY.md`.
