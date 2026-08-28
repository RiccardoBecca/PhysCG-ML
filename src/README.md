
This README documents the scripts used in the CG protein simulation pipeline, in the order they are typically run (see the SLURM job script for the full workflow).

---

## 1. `coarse_grain.py`

Coarse-grains an all-atom structure into the CG representation (CA+sidechain bead mapping) using the coarse-grain map.

```bash
python src/coarse_grain.py \
    --input data/pdb/1uao.pdb \
    --output data/pdb/1uao_CG.pdb
```

**Arguments**
- `--input` — all-atom input structure
- `--output` — path to write the coarse-grained structure

---

## 2. `evaluate_CG_structure.py`

Evaluates an input CG structure against the intramolecular (bonded) force field statistics.

```bash
python src/evaluate_CG_structure.py \
    --json_file data/potential/cg_statistics.json \
    --pdb data/pdb/1uao_CG.pdb \
    --outdir EXP/1uao/input_analysis
```

**Arguments**
- `--json_file` — path to the bonded-term statistics JSON (equilibrium values, force constants, etc.)
- `--pdb` — folded CG structure to evaluate
- `--outdir` — output directory for the analysis

---

## 3. `unfold_structure.py`

Generates an unfolded starting configuration from a folded CG structure.

```bash
python src/unfold_structure.py \
    --input data/pdb/1uao_CG.pdb \
    --output data/pdb/1uao_CG_unfolded.pdb
```

**Arguments**
- `--input` — folded CG structure
- `--output` — path to write the unfolded structure

---

## 4. `run_cg_stats.py`

Runs the CG MD simulation (bonded + TT+DH non-bonded potential, BAOAB/Langevin integrator), starting from the unfolded structure.

```bash
python src/run_cg_stats.py \
    --folded_pdb data/pdb/1uao_CG.pdb \
    --unfolded_pdb data/pdb/1uao_CG_unfolded.pdb \
    --json data/potential/cg_statistics.json \
    --csv data/potential/tt_dh_joint_params.csv \
    --charges data/potential/bead_charges.json \
    --out EXP/1uao/trajectory.xyz \
    --steps 20000000
```

**Arguments**
- `--folded_pdb` — reference folded CG structure (used for bonded-term equilibrium values)
- `--unfolded_pdb` — starting (unfolded) configuration
- `--json` — bonded statistics JSON
- `--csv` — fitted TT+DH non-bonded parameters
- `--charges` — per-bead charge JSON for the DH electrostatic term
- `--out` — output trajectory (`.xyz`)
- `--steps` — number of integration steps

---

## 5. `calculate_rmsd.py`

Computes RMSD of the trajectory with respect to the folded reference structure.

```bash
python src/calculate_rmsd.py \
    --xyz EXP/1uao/trajectory.xyz \
    --pdb data/pdb/1uao_CG.pdb \
    --out EXP/1uao/analysis_plots/rmsd_trajectory.png
```

**Arguments**
- `--xyz` — input trajectory
- `--pdb` — reference (folded) structure
- `--out` — output RMSD plot

---

## 6. `analyze_cg_folding.py`

Analyses folding behaviour along the trajectory relative to the folded reference.

```bash
python src/analyze_cg_folding.py \
    --xyz EXP/1uao/trajectory.xyz \
    --pdb data/pdb/1uao_CG.pdb \
    --out EXP/1uao/analysis_plots/folding_analysis.png
```

**Arguments**
- `--xyz` — input trajectory
- `--pdb` — reference (folded) structure
- `--out` — output folding-analysis plot

---

## 7. `run_tica_fel.py`

Builds a TICA-based free energy landscape (FEL) from a trajectory, and can reuse a previously fitted TICA model to project new trajectories or extract representative structures from specific minima.

### Standard usage — fit TICA and generate the FEL

```bash
python src/run_tica_fel.py \
    --xyz EXP/1uao/trajectory.xyz \
    --out EXP/1uao/analysis_plots/tica_fel.png \
    --eq_steps 2000000 \
    --save_tica EXP/1uao/tica.npz
```

### Reuse a fitted TICA to evaluate another trajectory

Loads a previously saved TICA model (`--load_tica`) instead of fitting a new one, and projects a different trajectory onto it — e.g. to compare a run without electrostatics against the reference TICA space:

```bash
python src/run_tica_fel.py \
    --xyz EXP_woelec/1uao/trajectory.xyz \
    --out EXP_woelec/1uao/analysis_plots/tica_fel.png \
    --eq_steps 2000000 \
    --load_tica EXP/1uao/tica.npz
```

### Reuse a fitted TICA to extract structures from specific minima

Loads a fitted TICA model and, for each `(TIC1, TIC2)` centre given via `--extract_centers`, extracts `--samples` representative structures from the trajectory nearest to that point, saving them with the given prefix. `--crystal` additionally overlays/compares against a reference crystal structure.

```bash
python src/run_tica_fel.py \
    --xyz EXP/1uao/trajectory.xyz \
    --out EXP/1uao/analysis_plots/tica_fel_crys.png \
    --eq_steps 2000000 \
    --load_tica EXP/1uao/tica.npz \
    --samples 5 \
    --extract_centers "-1.5, -0.6" "0.5, -0.7" "0.3, 1.6" \
    --extract_prefix EXP/1uao/tica_states/tica_states \
    --crystal data/pdb/1uao_CG.pdb
```

In this example, 5 configurations are extracted around `(TIC1, TIC2) = (0.5, -0.7)` and 5 more around `(TIC1, TIC2) = (0.3, 1.6)`.

**Arguments**
- `--xyz` — input trajectory
- `--out` — output FEL plot
- `--eq_steps` — number of initial (equilibration) steps to discard before analysis
- `--save_tica` — path to save the fitted TICA model
- `--load_tica` — path to a previously saved TICA model to reuse instead of fitting a new one
- `--samples` — number of structures to extract per centre (used with `--extract_centers`)
- `--extract_centers` — one or more `"TIC1, TIC2"` points to extract representative structures from
- `--extract_prefix` — output path prefix for extracted structures
- `--crystal` — reference crystal structure for comparison with extracted states

---

## 8. `compute_fold.py`

Studies folding based on end-to-end (first–last bead) distance and related metrics.

```bash
python src/compute_fold.py \
    --xyz EXP/1uao/trajectory.xyz \
    --out EXP/1uao/analysis_plots/fold_prop.png \
    --lag 100
```

**Arguments**
- `--xyz` — input trajectory
- `--out` — output plot
- `--lag` — lag time / window used in the fold-property calculation

---

## Typical pipeline order

1. `coarse_grain.py` — coarse-grain the all-atom input structure
2. `evaluate_CG_structure.py` — sanity-check the CG structure against the force field
3. `unfold_structure.py` — generate an unfolded starting point
4. `run_cg_stats.py` — run the CG MD simulation
5. `calculate_rmsd.py` — RMSD analysis
6. `analyze_cg_folding.py` — folding analysis
7. `run_tica_fel.py` — TICA/FEL analysis
8. `compute_fold.py` — fold-property analysis

See the SLURM submission script at the repo root for the full end-to-end job.



