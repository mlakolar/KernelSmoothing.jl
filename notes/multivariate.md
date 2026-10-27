# Multivariate densities: bandwidth matrices

In this note, we ask which bandwidth matrix of `MultivariateKDE` (#60) to use in two
dimensions: the normal scale matrix, the plug-in matrix of Wand and Jones (1994) and
Duong and Hazelton (2003), smoothed cross-validation (Duong and Hazelton 2005) or
least-squares cross-validation, and whether a full matrix is worth its extra
parameter.

## Design

We draw samples of n = 200 and 1000 from three bivariate targets, in the spirit of
those of Wand and Jones (1993): a normal with correlation 0.7; a bimodal mixture of
two normals with standard deviation 0.5 centred at (±1, 0); and a skewed mixture with
weights 0.2, 0.2 and 0.6 of normals centred at (0, 0), (0.5, 0.5) and (1, 1) with
standard deviations 1, 0.4 and 0.2. For each of the eight rules (four rules, full and diagonal), we
compute the integrated squared error on a 101 × 101 grid over [-3.5, 3.5]², with 500
replications at n = 200 and 200 at n = 1000 (`simulations/mvkde.jl`, run
`20261026T0215Z-ks-sim59c`).

## Results

Mean integrated squared error × 1000:

| target | n | full normal scale | full plug-in | full SCV | full LSCV | diagonal normal scale | diagonal plug-in | diagonal SCV | diagonal LSCV |
|---|---|---|---|---|---|---|---|---|---|
| correlated | 200 | 4.01 | 4.358 | 4.104 | 6.814 | 5.434 | 5.829 | 5.375 | 6.492 |
| correlated | 1000 | 1.487 | 1.534 | 1.519 | 1.944 | 1.99 | 2.033 | 2.007 | 2.244 |
| bimodal | 200 | 14.772 | 10.495 | 11.578 | 13.048 | 14.686 | 10.408 | 11.471 | 11.054 |
| bimodal | 1000 | 6.649 | 3.741 | 3.744 | 4.077 | 6.638 | 3.723 | 3.727 | 3.746 |
| skewed | 200 | 208.763 | 64.22 | 76.175 | 72.898 | 187.75 | 59.663 | 69.425 | 62.062 |
| skewed | 1000 | 121.573 | 21.496 | 23.32 | 23.281 | 103.524 | 20.581 | 21.988 | 21.376 |

Mean time of the selection and the fit, in seconds, averaged over the targets:

| n | normal scale | plug-in | SCV | LSCV |
|---|---|---|---|---|
| 200 | 0.0259 | 0.0031 | 0.174 | 0.065 |
| 1000 | 0.0656 | 0.059 | 1.02 | 0.656 |

The plug-in matrix is within 10% of the best rule for every target, and it is
fast. The normal scale matrix is the best for the normal target, as it should be, but
its error for the mixtures is 1.4 to 1.8 times that of the plug-in matrix for the
bimodal target and 3 to 6 times for the skewed one. The two cross-validation rules
come close to the plug-in matrix at n = 1000, except full least-squares
cross-validation on the correlated target, whose error is 27% larger; at n = 200,
least-squares cross-validation is the least accurate for the correlated target. The
full matrices pay off for the correlated target, by about a quarter of the error with
the plug-in and SCV rules. For the two mixtures, whose components are spherical, the
diagonal matrices are as good or better, by up to 15%. Note that the time of the
normal scale matrix at n = 200 includes compilation, since it is the first rule each
worker runs.

## Recommendations

- Use the default, the plug-in matrix. Use the normal scale matrix only for data that
  are close to normal.
- Use a full matrix when the coordinates are correlated, and consider a diagonal one
  for a mixture of spherical components.

## Binned selection (#86)

### Question

How much does linear binning change the selected bandwidth matrix and the density
error? Can it replace exact pair sums for PI and SCV, and is it suitable for LSCV?

### Design

We use the three bivariate targets above and the trivariate mixture
0.5 N((−1, 0, 0), 0.5² I) + 0.5 N((1, 0, 0), Σ), where Σ has unit variances and
correlation 0.5 between each pair of coordinates. Samples have n = 10³, 10⁴ and
10⁵. Each target has 100, 50 and 10 Monte Carlo replications, respectively. Each
sample uses PI, SCV and LSCV with full and diagonal matrices. Grid sweeps at
n = 10⁴ reuse the same samples across resolutions, with 50 replications per grid.
There are 990 sample-and-grid replications and six rules per replication
(`simulations/mvbinning.jl`, commit `066994e`). The base grids of this study have
151 points per axis in 2-D and 31 in 3-D. The latter differs from the final opt-in
default of 51; the binned sums are unchanged.

The reference selection is exact for n ≤ 10⁴. At n = 10⁵ it uses a finer binned
grid, with 301 points per axis in 2-D and 61 in 3-D. Matrix error is max |λ − 1|,
where λ ranges over the eigenvalues of H_ref⁻¹ H_binned. The ISE ratio is
ISE(H_binned) / ISE(H_ref), evaluated exactly by Gaussian overlap identities,
including self-pairs, for n ≤ 10⁴. ISE is not evaluated at n = 10⁵.
Absolute ISE change below means |ISE ratio − 1|.

Medians and 90th percentiles (p90) are empirical Monte Carlo summaries of finite,
converged pairs. Percentiles use linear interpolation between adjacent ordered
values. All errors below are percentages. Each table cell gives full / diagonal
matrices. PI and SCV have no failed pairs in these designs. Failed LSCV selections
are excluded from quantiles, so a reported quantile alone does not establish
reliability. These are binning errors relative to the reference selection, not
errors relative to an optimal bandwidth.

### Results

At n = 10⁴:

| target | grid per axis | rule | matrix error, median | matrix error, p90 | absolute ISE change, p90 |
|---|---|---|---|---|---|
| correlated (2-D) | 151 | PI | 0.06 / 0.13 | 0.12 / 0.27 | 0.042 / 0.075 |
| correlated (2-D) | 151 | SCV | 0.05 / 0.12 | 0.08 / 0.19 | 0.023 / 0.041 |
| bimodal (2-D) | 151 | PI | 0.14 / 0.13 | 0.24 / 0.22 | 0.059 / 0.047 |
| bimodal (2-D) | 151 | SCV | 0.10 / 0.10 | 0.11 / 0.11 | 0.033 / 0.031 |
| skewed (2-D) | 151 | PI | 1.21 / 0.87 | 1.40 / 1.01 | 0.442 / 0.327 |
| skewed (2-D) | 151 | SCV | 1.19 / 0.86 | 1.38 / 1.00 | 0.553 / 0.423 |
| trivariate (3-D) | 21 | PI | 20.44 / 35.03 | 34.91 / 46.97 | 9.727 / 16.734 |
| trivariate (3-D) | 21 | SCV | 10.76 / 9.17 | 11.62 / 10.93 | 5.111 / 4.500 |
| trivariate (3-D) | 31 | PI | 5.05 / 4.29 | 5.63 / 5.24 | 1.032 / 0.814 |
| trivariate (3-D) | 31 | SCV | 5.09 / 4.30 | 5.59 / 5.06 | 2.336 / 2.202 |
| trivariate (3-D) | 51 | PI | 1.85 / 1.59 | 2.14 / 1.98 | 0.425 / 0.327 |
| trivariate (3-D) | 51 | SCV | 1.82 / 1.55 | 2.03 / 1.79 | 0.823 / 0.755 |

The pre-registered gate requires p90 matrix error ≤ 2% and p90 absolute ISE change
≤ 1% for PI and SCV at n = 10⁴, for each target and matrix form. Both rules pass
in 2-D at 151 points per axis. The skewed target has the largest errors. In 3-D,
31 points fail the gate. At 51 points, both diagonal rules pass, but full PI and
SCV still miss the matrix-error target, at 2.14% and 2.03%.

The 2-D grid sweep on the bimodal target at n = 10⁴ is:

| target | grid per axis | rule | matrix error, median | matrix error, p90 | absolute ISE change, p90 |
|---|---|---|---|---|---|
| bimodal | 51 | PI | 1.22 / 1.10 | 1.98 / 1.77 | 0.432 / 0.458 |
| bimodal | 51 | SCV | 0.92 / 0.90 | 1.04 / 1.00 | 0.298 / 0.298 |
| bimodal | 101 | PI | 0.33 / 0.29 | 0.47 / 0.53 | 0.109 / 0.110 |
| bimodal | 101 | SCV | 0.23 / 0.22 | 0.26 / 0.26 | 0.073 / 0.074 |
| bimodal | 151 | PI | 0.14 / 0.13 | 0.24 / 0.22 | 0.059 / 0.047 |
| bimodal | 151 | SCV | 0.10 / 0.10 | 0.11 / 0.11 | 0.033 / 0.031 |
| bimodal | 301 | PI | 0.04 / 0.03 | 0.06 / 0.05 | 0.015 / 0.013 |
| bimodal | 301 | SCV | 0.03 / 0.03 | 0.03 / 0.03 | 0.008 / 0.008 |

Even 51 points pass the gate for this target. The finer grid of 151 provides more
margin and also passes for the skewed target. This supports automatic binning for
`PlugInMatrix` and `SCVMatrix` in 2-D above 500 observations with positive weight,
with 151 points per axis. The threshold does not use the effective sample size.
All selectors remain exact by default in 3-D. `LSCVMatrix` remains exact by default
in every dimension. `binned = true` opts into binning in 2-D or 3-D, with default
grids of 151 or 51 points per axis, respectively; `binned = false` forces exact
sums. These defaults apply to both full and diagonal matrices.

At n = 10⁵, all 2-D PI and SCV matrix errors are within 1.5% of the 301-point
reference. In 3-D at 31 points, PI has median errors of 51% (full) and 63%
(diagonal), reaching 79% against the 61-point reference. SCV has p90 errors of
4.4% (full) and 61% (diagonal). Agreement with a finer grid at this size does not
measure agreement with exact selection.

LSCV is less stable. In 2-D at 151 points, the correlated and bimodal targets have
median matrix errors of 0.3% to 1.4% at n = 10³–10⁴, with p90 errors of 2.6% to
9.6%. For the skewed target, full LSCV has p90 errors of 40% to 43% and collapses
at n = 10⁵. Only two of ten full selections there are finite and converged; the
diagonal selections have 100% matrix error. In 3-D, grids of 21 and 31 points
collapse at n = 10⁴. A grid of 51 avoids this collapse at that size, but its p90
matrix error is still 28% (full) and 16% (diagonal).

When observations share grid cells, binned LSCV can develop a spurious minimum as
H → 0, even with exact self-pair corrections. Issue #101 tracks a hybrid criterion
that keeps near-field pairs exact.

The problem also occurs in ks. On the shared n = 10⁴ benchmark samples
(`benchmark/binned.jl` and `benchmark/binned.R`, commit `1d86543`), its binned
`Hlscv` differs from exact LSCV by 7.0% in 2-D and 650% in 3-D. The 3-D result is
degenerate, with eigenvalue ratios 0.0, 3.5 and 7.5. The package's binned 2-D LSCV
differs by 0.59% on the same sample. These are single-sample comparisons.

### Recommendations

- Rely on the defaults: binned PI and SCV in 2-D above 500 observations with
  positive weight, exact selection in 3-D, and exact LSCV in every dimension.
- Use `binned = true` in 3-D only with the default grid of 51 points per axis or
  finer, and check `details.resolution`. Its selected and pilot values measure
  kernel standard deviations in units of grid spacing. A value below 1 calls for
  a finer grid or exact sums. This diagnostic does not guarantee the error gate.
- Do not use binned LSCV until the near-field correction of #101 is available.

## Tied data and collapsed LSCV selections (#102)

### Question

Rounded or duplicated observations can give LSCV spurious small-bandwidth minima.
With enough coincident pairs, its criterion can decrease without bound as the
bandwidth becomes singular. Previously, the search could stop there and report
convergence. Can separate diagnostics identify ties and numerical collapse without
flagging the small bandwidths needed by continuous multimodal densities?

### Design

We use six continuous designs: `normal`, N(0, I); `corr09`, a normal with unit
variances and equicorrelation 0.9; `bimodal6` and `bimodal10`, mixtures
0.5 N(0, I) + 0.5 N(s e₁, I), with s = 6 and 10; `lattice`, clusters with
standard deviation 0.05 at uniformly chosen nodes of a unit-spaced 5 × 5 lattice
in 2-D or 3 × 3 × 3 lattice in 3-D; and `lognormal`, the coordinatewise
exponential of N(0, I). Six tied designs reuse the normal sample at each
(d, n, seed): `round1`, `round2` and `round3` round to 1, 2 and 3 decimal places;
`dup5`, `dup10` and `dup25` replace the last k observations by copies of the first
k, with k = round(f n) and f = 0.05, 0.10 and 0.25.

Exact selections cross these designs with d = 2 and 3, n = 100, 400 and 2000,
full and diagonal matrices, and seeds 1–30, giving 4,320 cases. Another 40 cases
use binned full LSCV on `normal` and its `round2` twin, with d = 2 and 3,
n = 10⁴ and 10⁵, seeds 1–5, and the default grids. Samples use `Xoshiro` with a
fixed seed determined by design family, d, n and replicate. The results are from
commit `f6f9b09`, dated 2026-10-27. To reproduce the manifest and selections:

```sh
julia --project=benchmark benchmark/lscv_ties.jl manifest
julia --project=benchmark benchmark/lscv_ties.jl run 1 4360 lscv-ties.csv
```

The manifest command prints the number of cases. The run command accepts any
inclusive range of case ids and uses one Julia thread and one BLAS thread. Each
CSV row records the design and settings, status and error message, selection time,
convergence and collapse flags, relative eigenvalue extremes, duplicates and
coordinate ties, and separate tie, collapse, coarse-grid and other warning flags.
It also records H and independently computed S, A, H₀ and Hy, using padded lower
triangles. Failed selections remain in the output.

Here S is the sample covariance with divisor n − 1. The transformation is
A = S⁻¹ᐟ² for full matrices and diag(1 / sqrt(diag(S))) for diagonal ones.
With Σy = A S Aᵀ, the starting matrix is
H₀ = (4 / ((d + 2)n))^(2 / (d + 4)) Σy, using its diagonal for diagonal
selection, and Hy = A H Aᵀ. From the stored H₀ and Hy, Python independently
computes the extreme eigenvalues (ρ_min, ρ_max) of Symmetric(L₀⁻¹ Hy L₀⁻ᵀ),
where H₀ = L₀L₀ᵀ. Collapse means
ρ_min ≤ 10⁻⁸ or ρ_max ≥ 10⁸, a noncomputable ratio, a nonfinite H or Hy, or
a non-positive-definite H.

Separate computations on the original samples count observations equal to an
earlier one and coordinate tie fractions 1 − distinct/n, treating signed zeros
equally. The tie warning uses duplicates > 0 or max(ties) > 0.1. Selection times
exclude sample generation and these computations, but include compilation on the
first selection. Total selection time was 9,936 s (2.76 hours). The slowest case
took 59.6 s: exact diagonal `dup25`, d = 2, n = 2000, seed 26.

### Results

All 4,360 cases have rows, with no missing cases, duplicate case ids or selection
errors. None of the 2,160 continuous exact selections collapsed. Their smallest
relative eigenvalue was 2.33 × 10⁻⁴, on full `bimodal6` with d = 3, n = 100,
seed 13. This is 2.33 × 10⁴ times the lower collapse limit. The largest relative
eigenvalue was 6.12, on the same case. These are empirical margins for the designs
studied, not guarantees for every continuous density.

The package's duplicate counts and coordinate tie fractions agree with the
independent counts in every case. The tie warning appears exactly when those
counts meet its condition, and never on the continuous designs, including their
binned cases. All 335 collapsed selections emit the collapse warning and report
`converged = false`. The collapse flags agree with the independent matrix checks.

Counts pooled over dimensions, sample sizes, forms and seeds are:

| design | sums | cases | ρ_min < 0.2 | collapsed | tie-warned |
|---|---|---|---|---|---|
| normal | exact | 360 | 46 | 0 | 0 |
| corr09 | exact | 360 | 138 | 0 | 0 |
| bimodal6 | exact | 360 | 324 | 0 | 0 |
| bimodal10 | exact | 360 | 360 | 0 | 0 |
| lattice | exact | 360 | 360 | 0 | 0 |
| lognormal | exact | 360 | 359 | 0 | 0 |
| round1 | exact | 360 | 48 | 3 | 360 |
| round2 | exact | 360 | 48 | 0 | 356 |
| round3 | exact | 360 | 46 | 0 | 122 |
| dup5 | exact | 360 | 117 | 0 | 360 |
| dup10 | exact | 360 | 226 | 41 | 360 |
| dup25 | exact | 360 | 318 | 291 | 360 |
| normal | binned | 20 | 0 | 0 | 0 |
| round2 | binned | 20 | 0 | 0 | 20 |

The small-scale column uses independently recomputed ratios and includes collapsed
selections. It describes the selected scale, not density error. No binned case
collapsed in these normal and rounded-normal samples; this does not resolve the
binned LSCV failures on other targets described above.

For `dup25`, collapse counts by dimension and sample size are:

| d | n | full, out of 30 | diagonal, out of 30 | total, out of 60 |
|---|---|---|---|---|
| 2 | 100 | 28 | 25 | 53 |
| 2 | 400 | 29 | 27 | 56 |
| 2 | 2000 | 29 | 24 | 53 |
| 3 | 100 | 30 | 30 | 60 |
| 3 | 400 | 30 | 30 | 60 |
| 3 | 2000 | 6 | 3 | 9 |

In the 360 exact normal-versus-`round2` pairs, both selections have ρ_min < 0.2
in 46 pairs. Only `round2` is below this level in 2 pairs, only normal in none,
and neither in 312. Neither design collapses. Thus the tie warning identifies the
sample's discreteness even when rounding changes the small-scale count little.

Among the 1,825 noncollapsed exact selections from tied designs, the smallest
ρ_min was 5.07 × 10⁻⁷, on full `dup10` with d = 3, n = 400, seed 8. This selection
emits the tie warning but stays above the collapse limit. The two diagnostics
therefore address different problems: ties can matter well before numerical
singularity, while continuous mixtures and lattices can need small bandwidths.

### Recommendations

- Prefer `SCVMatrix` or `PlugInMatrix` for tied or rounded data. A tie warning
  identifies a risk of spurious LSCV minima even when the selected matrix has not
  collapsed; it does not establish that a particular selection is inaccurate.
- Treat a collapse warning as an unusable LSCV selection. Inspect
  `details.collapsed` and `details.converged`, and choose another rule. The returned
  matrix is the one found by the search; it is not replaced by a fallback.
- Do not reject a selection on continuous multimodal data solely because its
  relative eigenvalues are small. Compare the data's structure and other bandwidth
  rules; the collapse diagnostic detects degeneracy, not density-estimation error.
