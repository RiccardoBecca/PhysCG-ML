#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# ================================
# INPUT ARGUMENTS & DYNAMIC NAMES
# ================================
# Use the first command line argument, or default to "default" if omitted
SYSTEM_NAME="${1:-default}"

JOB_ID_NAME="ml_sim_${SYSTEM_NAME}_woel"

# Ensure the logs directory exists
mkdir -p logs
mkdir -p EXP_woelec
mkdir -p "EXP_woelec/${SYSTEM_NAME}"
mkdir -p "EXP_woelec/${SYSTEM_NAME}/analysis_plots"

# Dynamic SLURM log paths
LOG_OUT="logs/${JOB_ID_NAME}.out"
LOG_ERR="logs/${JOB_ID_NAME}.err"

# Override SLURM job name dynamically for squeue tracking
echo "Starting job: ${JOB_ID_NAME}"

FOLDED_PDB="data/pdb/${SYSTEM_NAME}_CG.pdb"
UNFOLDED_PDB="data/pdb/${SYSTEM_NAME}_CG_unfolded.pdb"
JSON_STATS="data/potential/cg_statistics.json"
CSV_PARAMS="data/potential/tt_params.csv"
CHARGES_JSON="data/potential/bead_charges.json"
OUTPUT_XYZ="EXP_woelec/${SYSTEM_NAME}/trajectory.xyz"
STEPS=20000000

source ~/anaconda3/etc/profile.d/conda.sh
conda activate openMM

# Redirect standard output and standard error dynamically to the logs folder
exec > "$LOG_OUT" 2> "$LOG_ERR"

# Analyse input struct wrt intramolecular force field
python src/evaluate_CG_structure.py \
    --json_file "$JSON_STATS" \
    --pdb "$FOLDED_PDB" \
    --outdir "EXP_woelec/${SYSTEM_NAME}/input_analysis"


# Unfold the structure
python src/unfold_structure.py \
    --input "$FOLDED_PDB" \
    --output "$UNFOLDED_PDB"


# run CG MD simulation
python src/run_cg_stats_woelec.py \
    --folded_pdb "$FOLDED_PDB" \
    --unfolded_pdb "$UNFOLDED_PDB" \
    --json "$JSON_STATS" \
    --csv "$CSV_PARAMS" \
    --out "$OUTPUT_XYZ" \
    --steps "$STEPS"


# calculate rmsd
python src/calculate_rmsd.py \
    --xyz "$OUTPUT_XYZ" \
    --pdb "$FOLDED_PDB" \
    --out "EXP_woelec/${SYSTEM_NAME}/analysis_plots/rmsd_trajectory.png"


# analyse folding
python src/analyze_cg_folding.py \
    --xyz "$OUTPUT_XYZ" \
    --pdb "$FOLDED_PDB" \
    --out "EXP_woelec/${SYSTEM_NAME}/analysis_plots/folding_analysis.png"


# get Tica energy landscape
python src/run_tica_fel.py \
    --xyz "$OUTPUT_XYZ" \
    --out "EXP_woelec/${SYSTEM_NAME}/analysis_plots/tica_fel.png" \
    --eq_steps $((STEPS / 10))


# Study fold based on first-last distance and similar
python src/compute_fold.py \
    --xyz "$OUTPUT_XYZ" \
    --out "EXP_woelec/${SYSTEM_NAME}/analysis_plots/fold_prop.png" \
    --lag 100



