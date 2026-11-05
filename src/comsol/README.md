# Generating a larger band-edge dataset from COMSOL

Goal: hundreds to thousands of (L1, L2) geometries, each giving f_lower and f_upper, exported as a
small table rather than a 4 GB per-mesh-node CSV.

## Why 2 k-points are enough
On the existing k-path (Γ = 0, M = 1, K = 2, Γ = 3):
- band 1 always peaks at **k = 2 (K)**;
- band 2 bottoms out at **k = 1 (M)** for rounded triangles (63 of 64) and at **k = 2 (K)** for sharp ones (63 of 64).

Using only k = 1 and k = 2 reproduces the band edges from the full 281-point path **exactly, for
all 64 rounded and all 64 sharp geometries** (checked against `data/comsol/bands.csv`).
That makes each geometry about 140× cheaper.

**Limit:** this was shown only for this design family (rounded or sharp triangles, L1 140–189 µm,
L2 91–140 µm). If you change the hole shape, add rotation, or free r/L, first run a few geometries
on the full path (`make_samples.py --k 0.1 0.2 ... 2.9`) and check where the extrema sit.

## Step 0: make the geometry list (both routes)
```
python src/comsol/make_samples.py --n 500
```
This writes `data/comsol/new/samples.csv` and `data/comsol/new/sweep_load_file.txt`:
- 3 **check** geometries from the old grid, followed by 500 Latin-hypercube samples;
- each geometry is listed twice, once for k = 1 and once for k = 2;
- L values are in **metres**.

The sampling is seeded, so rerunning with the same `--n` and `--seed` gives the same list.

## Route A: COMSOL GUI only (no extra software)
1. Open your model and **Save As** a copy, for example `data/comsol/new/vpc_light.mph`.
2. **Study 1 > Parametric Sweep:**
   - Sweep type: *Specified combinations*.
   - Delete the old rows, then **Load from File** and choose `sweep_load_file.txt`.
   - The table should now list L1, L2 and k, each with the same number of values.
   - The unit column is ignored when loading a .txt file. Set the unit to **m** for L1 and L2 if
     it's blank.
   - Leave *Keep solutions in memory* at **All**. Global Evaluation needs every solution.
3. **Eigenfrequency step:**
   - Do not change the number of eigenfrequencies or the search shift. The two lowest
     eigenfrequencies must still be bands 1 and 2.
   - Optional: *Values of Dependent Variables > Store fields in output: For selections* with an
     empty selection. This makes the file much smaller. Test it on the check run first, because
     COMSOL documents this setting for fields and doesn't say that `emw.freq` survives it.
4. **Check run first:**
   - Run `python src/comsol/make_samples.py --n 0`, which writes only the 3 check geometries
     (6 solves).
   - Load that file, compute, and **note the run time**.
5. **Results > Derived Values > Global Evaluation:**
   - Dataset: *Study 1/Parametric Solutions 1*.
   - Expression: `emw.freq`, unit **GHz**.
   - Click **Evaluate**, then export the table (Table window > Export, or Results > Export > Table)
     to `data/comsol/new/global_eval.txt`.
6. Parse the table:
   ```
   python src/comsol/parse_table.py data/comsol/new/global_eval.txt
   ```
   - It prints the columns it used and **new minus old** band edges for the check geometries.
     These should be about 0 GHz with the same mesh.
   - If they match, rerun steps 0 to 6 with `--n 500`, or as many geometries as your time allows.

Avoid an *Export to File* node **inside** the sweep. COMSOL forum reports say it can drop
eigenvalues after the first solution. Evaluate after the sweep instead.

## Route B: Python drives COMSOL (MPh), resumable and parallel
Use this for long runs and for active learning later.
1. `.venv\Scripts\pip install mph`. This needs a licensed local COMSOL install.
2. Prepare the model copy as in Route A, but **disable** the Parametric Sweep (right-click >
   Disable). Python sets L1, L2 and k itself.
3. Run the check geometries:
   ```
   python src/comsol/run_mph.py --model data/comsol/new/vpc_light.mph --check
   python src/comsol/run_mph.py --collect
   ```
   This prints the eigenfrequencies per k, the seconds per geometry, and new minus old band edges.
4. Run everything. The run can be stopped and restarted, and it skips finished ids.
   ```
   python src/comsol/run_mph.py --model data/comsol/new/vpc_light.mph
   ```
   For parallel runs, start one process per shard, splitting the cores:
   ```
   python src/comsol/run_mph.py --model ... --shard 0/2 --cores 4
   python src/comsol/run_mph.py --model ... --shard 1/2 --cores 4
   ```
5. `python src/comsol/run_mph.py --collect` writes `bands_new.csv` and `band_edges_new.csv`.

If your model names the parameters differently, pass `--names <L1> <L2> <k>` to both scripts.
The old data shows L1 and L2 as multiples of a/√3, so the original sweep may have used a ratio
parameter. L1 and L2 must be **global parameters** that the geometry uses.

## Output (both routes)
`data/comsol/new/band_edges_new.csv` has the same columns as `data/comsol/band_edges.csv`:
corner, L1_um, L2_um, f_lower, f_upper, gap, plus n_k. The benchmark can read it directly.

## Planning the size
Total time ≈ (seconds per geometry from the check run) × (number of geometries) ÷ (parallel
processes).

Example: 20 s per geometry × 1,000 geometries ≈ 5.5 h, which fits overnight.

## Sources
- COMSOL 6.4 Parametric Sweep (Specified combinations, Load from File format):
  https://doc.comsol.com/6.4/doc/com.comsol.help.comsol/comsol_ref_solver.36.049.html
- COMSOL band-gap photonic crystal tutorial (Floquet k, eigenfrequency sweep):
  https://doc.comsol.com/6.4/doc/com.comsol.help.models.woptics.bandgap_photonic_crystal/bandgap_photonic_crystal.html
- MPh (Python interface to COMSOL): https://mph.readthedocs.io/en/1.3/api/mph.Model.html and
  https://mph.readthedocs.io/en/latest/tutorial.html
- COMSOL batch mode (an alternative for clusters): https://www.comsol.com/blogs/how-to-run-simulations-in-batch-mode-from-the-command-line
- Store fields in output: https://www.comsol.com/blogs/minimize-your-model-file-size-with-storing-solution-techniques
