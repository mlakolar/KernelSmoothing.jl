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
  | `quantreg.jl` | coverage of conventional and robust bias-corrected pointwise intervals of `LocalQuantileRegression` (#62, #94), including an adaptation of Qu and Yoon (2019, Section 8, equation (21), Model 2) for pointwise checks |
  | `shape.jl` | error and coverage with and without `rearrange` and `constrain` (#63) |
  | `bounded.jl` | error of density estimates on [0, 1]: `BoundedKDE` (#61) against `KDE` and `LocalPolynomialDensity` |
  | `rdlocrand.jl` | size and power of the randomization tests of `LocalRandomization` (#64) |
  | `mvbinning.jl` | binning error of full and diagonal PI, SCV and LSCV matrices (#86), against exact or fine-grid references, with Gaussian-overlap ISE |

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

## Binning error of bandwidth matrices

`mvbinning.jl` uses study code 15. The fixed base design list is `correlated`,
`bimodal`, `skewed`, `trivariate`, in that order; its positions determine the seeds.
The first three targets are exactly those of `mvkde.jl`. The fourth is
0.5 N((-1, 0, 0), 0.25 I) + 0.5 N((1, 0, 0), Σ), with unit variances and all
correlations 0.5. For each target, the task list requests 100 replications at
n = 1000, 50 at 10000 and 10 at 100000.

At n = 10000, `bimodal-g51`, `bimodal-g101`, `bimodal-g151`, `bimodal-g301` and
`trivariate-g21`, `trivariate-g31`, `trivariate-g51` add 50 replications per grid.
These names reuse the seed of the base target, pairing the samples across grids
and with the default-grid study. Keep the base list in order when adding designs.

All six selectors explicitly use binning. Automatic selection bins only 2-D PI
and SCV above 500 observations; LSCV and all 3-D rules default to exact sums.
The grid is otherwise the selector's opt-in default, currently 151 in 2-D and 51
in 3-D, unless the design specifies it. Rows record the grid actually used,
including 31 in earlier 3-D runs. The reference is exact through
n = 10000; at n = 100000 it uses binning at 301 nodes per axis in 2-D or 61 in 3-D.
The generalized eigenvalue error is `maximum(abs.(eigvals(H_bin, H_ref) .- 1))`.
ISE is the exact Gaussian-overlap identity through n = 10000, including the
O(n²) self-overlap sum. ISE and its ratio are NaN at n = 100000 by design. Selection
exceptions are logged and retained as failed rows. Times measure selection only;
they include compilation when a method first runs. The ISE work is not timed.

`summarize.jl mvbinning` reports attempts, failures on either side, the median,
90th percentile and maximum of the eigenvalue error and paired ISE ratio, and the
median selection times. It also reports `abs(ISE ratio - 1)` for the acceptance
criterion. Metric summaries use pairs with finite, converged selections and valid
metrics. Time medians include finite timings of all attempts, including failures.
The n = 100000 ISE omissions do not count as failures.

There are two replications per task at n = 1000 and one at the larger sizes and
in the sweep. This assumes roughly quadratic exact-sum scaling from the 106-second
2-D full SCV time at n = 10000 in `../notes/timings.md`. Full and diagonal rules
together can take several minutes per replication. There is no measured 3-D SCV
baseline, so the single-replication tasks can exceed that estimate. The quick list
checks two replications each of the bimodal and trivariate targets at n = 1000.
Its chunk sizes match the full tasks, so their saved outputs can be reused.
Before the full study, measure a trivariate n = 10000 replication separately:

```
julia --project=simulations simulations/mvbinning.jl trivariate 10000 1 1 canary.csv
```

The optional `ise_selfcheck()` uses five observations in 2-D and a fine integration
grid to check the overlap calculation. It runs only when called explicitly.
`mvkde.jl` keeps `binned = false` for the historical study.
