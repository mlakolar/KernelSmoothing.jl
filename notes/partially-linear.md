# The partially linear model: coverage

In this note, we ask whether the sandwich intervals for the coefficients of
`PartiallyLinear` (Robinson 1988) cover when the covariates depend on the variable
of the smooth part. We compare three cross-validated bandwidth rules, including
`bandwidth_mode = :smallest`, the new default (#96).

## Design

y = x₁ - 0.5 x₂ + g(u) + σ ε with u uniform on [0, 1], x₁ = u + v₁ and
x₂ = m₂(u) + v₂. The variables v₁, v₂ and ε are independent standard normals,
independent of u. The seven designs are:

| design | m₂(u) = E[x₂ ∣ u] | g(u) | σ |
|---|---|---|---|
| sine | sin(2πu) | sin(2πu) | 0.5 |
| sine-heteroskedastic | sin(2πu) | sin(2πu) | 0.2 + 0.6u |
| logistic | sin(2πu) | 2/(1 + exp(-(u - 0.5)/0.05)) | 0.5 |
| quadratic | u² | sin(2πu) | 0.5 |
| cancellation | sin(2πu) | 0.5 sin(2πu) | 0.5 |
| g-zero | sin(2πu) | 0 | 0.5 |
| linear-means | -u | sin(2πu) | 0.5 |

In cancellation, E[y ∣ u] = u is linear. In linear-means, both covariate means are
linear. These are custom designs, not a replication of a published study.

The earlier run used the first three designs.
The intervals use the
HC0 and HC1 sandwich variances at the bandwidth of `CrossValidation()`, for n = 200
and 1000, with 2000 replications (`simulations/partiallylinear.jl`; run in the
development allocation of `20261025T2000Z-ks-overnight`).

The expanded study uses `simulations/partiallylinear.jl` at commit `5d6d143`.
The main run uses replications 1–2000. It keeps the original designs' seeds.
The confirmation run uses fresh seeds, replications 2001–4000, at the same sample
sizes and commit. All smooths are local linear fits with the Epanechnikov kernel.

Each cross-validated fit selects a bandwidth for y and for each column of X.
The rules differ in how they use those selections:

- `:response`, the earlier default, uses the bandwidth selected for y for all smooths.
- `:smallest`, the new default, uses the smallest selected bandwidth for all smooths.
- `:each` uses each smooth's own selected bandwidth.

The fitted smooth part ĝ uses the effective bandwidth of the smooth of y.
The oracle regresses y − E[y ∣ u] on X − E[X ∣ u]. It uses the true conditional
means and the same sandwich formula. It has no estimated bandwidth.

## Results

Coverage of 95% HC0 intervals in the main run, in percent (Monte Carlo standard
error in percentage points). Each comparison uses the common valid replications
across all four methods. There are 2000 per cell, except quadratic and linear-means
at n = 200, which have 1999. The `response` columns reproduce the earlier run's
coverage with the same seeds. Its interval lengths and median bandwidths also
reproduce the earlier results.

Coefficient of x₁:

| design | n | response | smallest | each | oracle |
|---|---|---|---|---|---|
| sine | 200 | 93.2 (0.6) | 93.1 (0.6) | 90.4 (0.7) | 93.4 (0.6) |
| sine | 1000 | 96.1 (0.4) | 95.9 (0.4) | 94.4 (0.5) | 96.0 (0.4) |
| sine-heteroskedastic | 200 | 94.7 (0.5) | 94.4 (0.5) | 91.2 (0.6) | 94.8 (0.5) |
| sine-heteroskedastic | 1000 | 94.7 (0.5) | 94.8 (0.5) | 93.4 (0.6) | 94.9 (0.5) |
| logistic | 200 | 94.1 (0.5) | 93.9 (0.5) | 89.5 (0.7) | 95.2 (0.5) |
| logistic | 1000 | 94.2 (0.5) | 94.2 (0.5) | 92.1 (0.6) | 94.2 (0.5) |
| quadratic | 200 | 93.7 (0.5) | 93.6 (0.5) | 87.5 (0.7) | 94.0 (0.5) |
| quadratic | 1000 | 95.2 (0.5) | 95.2 (0.5) | 92.8 (0.6) | 95.1 (0.5) |
| cancellation | 200 | 94.6 (0.5) | 93.9 (0.5) | 92.7 (0.6) | 94.8 (0.5) |
| cancellation | 1000 | 93.8 (0.5) | 93.4 (0.6) | 92.8 (0.6) | 93.6 (0.5) |
| g-zero | 200 | 93.4 (0.6) | 93.5 (0.6) | 91.0 (0.6) | 94.2 (0.5) |
| g-zero | 1000 | 94.8 (0.5) | 94.8 (0.5) | 93.8 (0.5) | 94.8 (0.5) |
| linear-means | 200 | 93.6 (0.5) | 93.6 (0.5) | 88.5 (0.7) | 93.8 (0.5) |
| linear-means | 1000 | 94.2 (0.5) | 94.3 (0.5) | 92.1 (0.6) | 94.4 (0.5) |

Coefficient of x₂:

| design | n | response | smallest | each | oracle |
|---|---|---|---|---|---|
| sine | 200 | 73.0 (1.0) | 93.4 (0.6) | 92.6 (0.6) | 93.4 (0.6) |
| sine | 1000 | 93.0 (0.6) | 95.4 (0.5) | 95.2 (0.5) | 95.6 (0.5) |
| sine-heteroskedastic | 200 | 70.9 (1.0) | 94.9 (0.5) | 94.0 (0.5) | 95.4 (0.5) |
| sine-heteroskedastic | 1000 | 92.2 (0.6) | 95.2 (0.5) | 94.9 (0.5) | 95.1 (0.5) |
| logistic | 200 | 93.4 (0.6) | 93.2 (0.6) | 93.0 (0.6) | 94.1 (0.5) |
| logistic | 1000 | 94.3 (0.5) | 94.2 (0.5) | 94.2 (0.5) | 94.4 (0.5) |
| quadratic | 200 | 93.9 (0.5) | 94.0 (0.5) | 92.5 (0.6) | 94.2 (0.5) |
| quadratic | 1000 | 94.6 (0.5) | 94.6 (0.5) | 94.0 (0.5) | 94.6 (0.5) |
| cancellation | 200 | 50.2 (1.1) | 93.4 (0.6) | 92.8 (0.6) | 94.2 (0.5) |
| cancellation | 1000 | 20.7 (0.9) | 94.4 (0.5) | 94.2 (0.5) | 94.6 (0.5) |
| g-zero | 200 | 93.8 (0.5) | 93.2 (0.6) | 93.3 (0.6) | 94.0 (0.5) |
| g-zero | 1000 | 95.4 (0.5) | 95.3 (0.5) | 94.8 (0.5) | 95.2 (0.5) |
| linear-means | 200 | 94.1 (0.5) | 93.9 (0.5) | 92.3 (0.6) | 93.8 (0.5) |
| linear-means | 1000 | 94.7 (0.5) | 94.6 (0.5) | 94.1 (0.5) | 94.6 (0.5) |

The pre-registered rule requires coverage of at least 92.5% in every primary cell:
the coefficient of x₂ in sine, sine-heteroskedastic and cancellation, at both
sample sizes. The no-harm condition requires a paired coverage change against
`:response` of at least −1.5 percentage points in every other cell. Both conditions
use HC0 on the common valid replications, before rounding.

`:smallest` is eligible. It passes both conditions. `:each` passes the primary
condition but fails no-harm. Its largest loss is 6.2 points (Monte Carlo SE 0.7)
for x₁ in quadratic at n = 200. The confirmation run gives the same eligibility
decisions. The lowest primary coverage of `:smallest` there is 93.2% (0.6), for x₂
in sine at n = 200. The largest loss of `:each` is 6.6 points (0.7), for x₁ in
logistic at n = 200.

Separate smoothers introduce the residual term Σⱼ βⱼ (S_j − S_y) x_j, where S_y
and S_j are the smoothing matrices for y and x_j. This term vanishes with a common
smoother. Choosing each bandwidth for its own conditional mean can therefore harm
the coefficient regression. A linear conditional mean for x₁ does not prevent
this problem. In cancellation, the response mean is linear, but its bandwidth is
too large for the sine mean of x₂. The common smallest bandwidth resolves that
mean and improves coverage.

Calibration of β̂₂ in sine at n = 200, in the main run:

| rule | bias | SD | mean se | RMS se | mean length |
|---|---|---|---|---|---|
| response | 0.046 | 0.064 | 0.037 | 0.037 | 0.143 |
| smallest | 0.005 | 0.037 | 0.035 | 0.036 | 0.139 |

The response rule has both bias and an empirical SD well above the reported
standard errors. `:smallest` reduces the bias and brings the SD and standard
errors closer together. The gain in coverage does not come from longer intervals
in this cell.

Across the design, sample size and coefficient cells, the ratio of mean HC0
interval lengths, `smallest / response`, has median 0.999 and range 0.937–1.024.
In the confirmation run, the median is 0.999 and the range is 0.934–1.025.

Median effective bandwidths in the main run are below. The response and smallest
columns give their common bandwidths. For `:each`, h_y equals the response column;
h₁ and h₂ are shown separately. Medians use all finite attempted bandwidths.
CV boundary shares use attempts with known flags for all smooths, before a common
bandwidth is imposed; the shares below are for the selections reported by `:each`.

| design | n | response | smallest | each: h₁, h₂ | x₁ CV boundary (%) |
|---|---|---|---|---|---|
| sine | 200 | 0.252 | 0.148 | 0.982, 0.162 | 57.4 |
| sine | 1000 | 0.169 | 0.108 | 0.997, 0.115 | 58.5 |
| sine-heteroskedastic | 200 | 0.263 | 0.147 | 0.982, 0.162 | 57.6 |
| sine-heteroskedastic | 1000 | 0.170 | 0.109 | 0.997, 0.116 | 60.5 |
| logistic | 200 | 0.160 | 0.135 | 0.982, 0.165 | 55.6 |
| logistic | 1000 | 0.110 | 0.098 | 0.997, 0.116 | 58.4 |
| quadratic | 200 | 0.178 | 0.168 | 0.979, 0.811 | 55.0 |
| quadratic | 1000 | 0.125 | 0.121 | 0.996, 0.388 | 57.6 |
| cancellation | 200 | 0.981 | 0.156 | 0.981, 0.165 | 55.2 |
| cancellation | 1000 | 0.996 | 0.112 | 0.996, 0.115 | 56.8 |
| g-zero | 200 | 0.259 | 0.150 | 0.982, 0.163 | 56.6 |
| g-zero | 1000 | 0.169 | 0.110 | 0.997, 0.116 | 59.8 |
| linear-means | 200 | 0.177 | 0.169 | 0.981, 0.982 | 57.0 |
| linear-means | 1000 | 0.124 | 0.121 | 0.997, 0.997 | 60.3 |

The conditional mean of x₁ is linear in every design. Its CV bandwidth often hits
the upper end of the search range. This does not by itself indicate a poor fit to
that mean. It does explain the large difference between its own selected bandwidth
and the common smallest bandwidth. In cancellation, response CV also hits a
boundary in 55.8% of attempts at n = 200 and 57.9% at n = 1000.

Failures are rare and occur only at n = 200. In the main run, `:smallest` and
`:each` each fail in one of 2000 replications in quadratic and in linear-means.
In confirmation, they each fail once in sine-heteroskedastic and once in
cancellation; `:response` also fails once in cancellation. These failures report
that a smooth in u is not identified at some observation. They are excluded from
the common valid comparisons. There are no failures at n = 1000 in either run.

## Recommendations

- Use the default `bandwidth_mode = :smallest`. It improves coverage in the primary
  cells and meets the no-harm condition in both runs, with little change in mean
  interval length. These simulations do not guarantee coverage in other designs.
- Pass `bandwidth = h` for a common numeric bandwidth when prior knowledge or
  sensitivity checks call for explicit control of the smoothing. It should resolve
  nonlinear covariate means as well as the response mean. Numeric bandwidths
  override the mode; a vector supplies separate bandwidths for y and the columns
  of X, in that order. Separate smoothing can introduce the residual term above.
- If a smooth is not identified at an observation, use a larger common numeric
  bandwidth. Check sensitivity of the coefficients and intervals, since a larger
  bandwidth can restore identification while leaving more smoothing bias.
