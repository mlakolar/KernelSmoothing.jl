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
  | `partiallylinear.jl` | coverage of HC0 and HC1 coefficient intervals for `PartiallyLinear` (#96), comparing response, smallest and separate CV bandwidths with an oracle using the true conditional means |

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

## Partially linear coefficient coverage

`partiallylinear.jl` uses study code 14. Keep `sine`, `sine-heteroskedastic` and
`logistic` in their original positions so their seeds and data reproduce. All
designs have u uniform on [0, 1], x₁ = u + v₁, β = [1, -0.5] and independent
standard normal v₁, v₂ and ε. The original designs have x₂ = sin(2πu) + v₂ and
g(u) = sin(2πu), except for the steep logistic design. Errors have σ = 0.5,
except for `sine-heteroskedastic`, where σ = 0.2 + 0.6u.

Four custom designs are appended. `quadratic` changes E[x₂ | u] to u².
`cancellation` changes g(u) to 0.5 sin(2πu), so E[y | u] = u. `g-zero` sets g to
zero. `linear-means` changes E[x₂ | u] to -u. The appended designs have σ = 0.5
and otherwise retain the sine design. These are custom designs, not a published
replication.

Each replication compares `response`, `smallest`, `each` and `oracle`. The first
three use `CrossValidation()` with the corresponding `bandwidth_mode`. HC1 reuses
the full effective bandwidth vector from HC0. The oracle subtracts the true
conditional means of y and X, then uses OLS with HC0 and HC1 sandwiches. Every
method, VCE and coefficient has a row, including failures. A failed HC1 refit
retains its successful HC0 result. The CSV columns are `design,n,rep,method,vce,`
`coefficient,truth,estimate,se,h_y,h_1,h_2,cv_boundary_y,cv_boundary_1,cv_boundary_2,error`;
VCE is `hc0` or `hc1`. Bandwidths are the effective values, ordered y, x₁, x₂.
Each boundary flag is its selection's `details.atbound`, before imposing a common
bandwidth, including covariate selections whose bandwidths are replaced under
`response` or `smallest`. HC1 retains the flags from the HC0 selections.
Unknown flags, flags for numeric bandwidths, and oracle bandwidths and flags are
NaN. Exception messages replace commas, quotes and line breaks so they remain
single unquoted CSV fields.

The default tasks request n = 200 and 1000 with replications 1–2000, in chunks of
1000. At the preliminary estimate of 0.1–0.2 seconds per replication at n = 1000,
a task takes a few minutes; check this with the two-replication n = 1000 canary.
Use a separate output directory for the canary: saved-task names include the
first replication but omit the chunk length, so a canary output could otherwise
cause the first full task to be skipped.

For confirmation on fresh seeds, generate the optional task list:

```sh
PARTIALLYLINEAR_CONFIRMATION=1 bash simulations/maketasks.sh CONFIRMATION_DIR partiallylinear
```

The directory must already exist. This also writes
`tasks-partiallylinear-confirmation.txt`, containing replications 2001–4000 for
every design and n, with the same chunk size. The normal task lists stay the
same. Run the confirmation list in a separate output directory, selecting it
with `TASKS=tasks-partiallylinear-confirmation.txt`, so the combined results do
not mix the original and confirmation replications. No seed offset is needed:
`seed` includes the replication index.

`summarize.jl partiallylinear` reads `results-partiallylinear.csv` in the current
directory, including a confirmation file with that name. An explicit file is
also accepted as `summarize.jl partiallylinear=PATH.csv`. Both runs use the same
schema and summary. Tables report attempts, failures and the common valid set
across all four methods, separately for each design, n, VCE and coefficient.
Coverage, its Monte Carlo SE, paired coverage differences against `response`,
bias, SD, mean and RMS SE, and mean interval length all use this common set.
Bandwidth medians use all finite attempted bandwidths. Boundary percentages are
reported separately for y, x₁ and x₂. They use attempts with known CV flags for all
three smooths and report that shared denominator. Empty valid sets produce NaN
summaries; missing methods have zero attempts. SD and paired
Monte Carlo SE are NaN with fewer than two common valid replications. Files in
the old schema must be regenerated before using this summary.

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
