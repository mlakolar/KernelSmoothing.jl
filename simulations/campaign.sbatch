#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
# Runs the task list of a simulation campaign on one node: each line of TASKS is
# "STUDY DESIGN N FIRSTREP NREPS", and GNU Parallel runs one single-threaded Julia
# worker per task as an exclusive job step. Tasks whose output exists are skipped, so
# a rerun resumes. Submit from the run directory, which holds the package source in
# src/ (with this folder in src/simulations), with RUN set, for example on USC's
# Discovery cluster:
#
#   RUN=$PWD sbatch --account=ACCOUNT --partition=main --ntasks=64 --mem=120G \
#     --time=02:00:00 --output=$PWD/%x-%j.out --export=ALL src/simulations/campaign.sbatch
#
# JULIA is the Julia binary, TASKS the task list (by default tasks.txt).
set -euo pipefail
RUN=${RUN:?set RUN to the run directory}
JULIA=${JULIA:-julia}
TASKS=${TASKS:-tasks.txt}
export JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1
cd "$RUN"
SIM=src/simulations
$JULIA --startup-file=no --project=$SIM -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using KernelSmoothing, Distributions'
mkdir -p out
MEM=$(( ${SLURM_MEM_PER_NODE:-$((2000 * SLURM_NTASKS))} / SLURM_NTASKS ))M
parallel -j "$SLURM_NTASKS" --colsep ' ' --joblog "joblog-$SLURM_JOB_ID.txt" \
  '[ -s out/{1}-{2}-{3}-{4}.csv ] || srun --exclusive --exact -N1 -n1 -c1 --mem='"$MEM"' '"$JULIA"' --startup-file=no --project='"$SIM"' '"$SIM"'/{1}.jl {2} {3} {4} {5} out/{1}-{2}-{3}-{4}.csv.tmp && mv out/{1}-{2}-{3}-{4}.csv.tmp out/{1}-{2}-{3}-{4}.csv' \
  :::: "$TASKS"
# the results of each study in one file
for s in $(cut -d' ' -f1 "$TASKS" | sort -u); do
  files=(out/$s-*.csv)
  [ -e "${files[0]}" ] || continue
  head -1 "${files[0]}" > "results-$s.csv"
  for f in "${files[@]}"; do tail -n +2 "$f" >> "results-$s.csv"; done
done
wc -l results-*.csv
