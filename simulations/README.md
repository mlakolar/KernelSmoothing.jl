# Simulation studies

This folder holds the Monte Carlo studies behind the notes in [`../notes`](../notes)
(#59). Every table in the notes can be reproduced from the scripts here, from a fixed
seed.

## Layout

- `common.jl`: the shared definitions. These are the arguments of a study script, the
  seeds, and the CSV output.
- One script per study. It is run as

  ```
  julia --project=. STUDY.jl DESIGN N FIRSTREP NREPS OUT
  ```

  and writes the results of replications `FIRSTREP` to `FIRSTREP + NREPS - 1` of one
  design and sample size to the CSV file `OUT`. Replication `r` draws its data from
  `Xoshiro(seed(study, design, n, r))`, so any replication can be redone alone.

  | script | study |
  |---|---|
  | `locallik.jl` | coverage of the conventional and robust bias-corrected intervals of `LocalLikelihood` (#65), with cross-validated and rule-of-thumb bandwidths |
  | `quantreg.jl` | coverage of the pointwise intervals of `LocalQuantileRegression` (#62) |
  | `shape.jl` | error and coverage with and without `rearrange` and `constrain` (#63) |
  | `bounded.jl` | error of density estimates on [0, 1]: `BoundedKDE` (#61) against `KDE` and `LocalPolynomialDensity` |
  | `rdlocrand.jl` | size and power of the randomization tests of `LocalRandomization` (#64) |

- `maketasks.sh`: writes the task list of the campaign, one line
  `STUDY DESIGN N FIRSTREP NREPS` per task.
- `campaign.sbatch`: runs a task list on one node of a Slurm cluster. GNU Parallel
  runs one single-threaded Julia worker per task, and a rerun skips the tasks whose
  output exists.
- `summarize.jl`: prints the tables of the notes from the combined results.

## Running a campaign

On a cluster with Slurm and GNU Parallel, put the package source in `src/` of a run
directory and submit from there:

```
mkdir -p run/src && cd run && git -C /path/to/KernelSmoothing.jl archive HEAD | tar -x -C src
bash src/simulations/maketasks.sh .
julia --project=src/simulations -e 'using Pkg; Pkg.instantiate()'
RUN=$PWD sbatch --account=ACCOUNT --ntasks=64 --mem=120G --time=03:00:00 \
  --export=ALL src/simulations/campaign.sbatch
julia --project=src/simulations src/simulations/summarize.jl
```

With `TASKS=tasks-quick.txt`, the job runs a short canary list instead.

A single task also runs on a laptop:

```
julia --project=src/simulations src/simulations/shape.jl flat 200 1 10 out.csv
```
