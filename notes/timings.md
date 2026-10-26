# Timings

In this note, we ask how long the default call of each estimator takes, how the time
grows with the sample size, and how much threads help. The question matters most for
the procedures whose cost grows faster than linearly: cross-validation over pairs of
observations, and exact sums in several dimensions.

## Design

The data are x ~ Beta(2, 2), y = sin(2πx) + 0.5 ε, a binary response with
P(b = 1 | x) = 1/(1 + exp(-2 sin(2πx))), and a second coordinate 0.5x + 0.3 ε for the
bivariate estimates, for n = 10³, 10⁴, 10⁵ and 10⁶. Each call runs once to compile and
is then timed at least three times within a budget of a few seconds, or once when it
takes more than 20 seconds; we report the minimum. Each call has a fixed largest
sample size. The times are on one core and on eight cores of
an AMD EPYC 7513 node of Discovery, with Julia 1.13.2 (`benchmark/timings.jl`, run
`20261026T0105Z-ks-timings`). The estimates of regression and of densities at points
use the package's default of 30 points (19 for `LocalPolynomialDensity`).

## Results

Seconds, on one thread / on eight threads:

| call | n = 10^3 | n = 10^4 | n = 10^5 | n = 10^6 |
|---|---|---|---|---|
| `SmoothingData(x, y)` | 2.3e-05 / 2.2e-05 | 0.00031 / 0.0003 | 0.0039 / 0.0039 | 0.048 / 0.049 |
| `fit(KDE, x)` | 0.0016 / 0.0016 | 0.0021 / 0.0021 | 0.0068 / 0.0069 | 0.052 / 0.052 |
| `fit(KDE, x; bandwidth = SheatherJones())` | 0.0016 / 0.0016 | 0.0021 / 0.0021 | 0.0069 / 0.0068 | 0.052 / 0.052 |
| `fit(KDE, x; bandwidth = LSCV())` | 0.024 / 0.0094 | 0.26 / 0.065 | 2.7 / 0.66 |  |
| `fit(PointwiseDensity, x)` | 0.0018 / 0.0021 | 0.0028 / 0.0025 | 0.011 / 0.0098 | 0.08 / 0.071 |
| `fit(LocalPolynomialDensity, x)` | 0.0039 / 0.0035 | 0.0087 / 0.0046 | 0.046 / 0.016 |  |
| `fit(LocalPolynomialRegression, x, y)` | 0.013 / 0.0089 | 0.13 / 0.059 | 1.4 / 0.34 | 14 / 3.3 |
| `fit(LocalPolynomialRegression, x, y; bandwidth = IMSEOptimal())` | 0.013 / 0.009 | 0.13 / 0.059 | 1.4 / 0.34 | 13 / 3.4 |
| `fit(LocalPolynomialRegression, x, y; bandwidth = CrossValidation())` | 0.012 / 0.011 | 0.12 / 0.054 | 1.2 / 0.28 |  |
| `conf_band(fit(LocalPolynomialRegression, x, y))` | 0.015 / 0.01 | 0.14 / 0.061 | 1.4 / 0.36 | 14 / 3.4 |
| `fit(RegressionDiscontinuity, x .- 0.5, y)` | 0.00021 / 0.00026 | 0.0017 / 0.0017 | 0.017 / 0.017 | 0.19 / 0.18 |
| `fit(DensityDiscontinuity, x .- 0.5)` | 9.9e-05 / 9.9e-05 | 0.00053 / 0.00053 | 0.0051 / 0.0051 |  |
| `fit(LocalLikelihood, x, b)` | 3.5 / 0.67 | 270 / 38 |  |  |
| `fit(LocalLikelihood, x, b; bandwidth = RuleOfThumb())` | 0.0032 / 0.0012 | 0.02 / 0.008 | 0.15 / 0.073 | 1.2 / 0.72 |
| `fit(LocalLikelihoodDensity, x)` | 0.0014 / 0.00065 | 0.01 / 0.003 | 0.1 / 0.032 |  |
| `fit(LocalQuantileRegression, x, y)` | 0.027 / 0.012 | 0.33 / 0.098 | 6.9 / 1.1 |  |
| `fit(ConditionalDensity, x, y)` | 1.8 / 1.2 | 182 / 120 |  |  |
| `fit(BoundedKDE, x; support = (0, 1))` | 0.0016 / 0.0016 | 0.003 / 0.0031 | 0.023 / 0.023 | 0.32 / 0.28 |
| `fit(BoundedKDE, x; support = (0, 1), method = BetaKernels())` | 0.68 / 0.68 | 33 / 30 |  |  |
| `fit(MultivariateKDE, X2)` | 0.055 / 0.0075 | 5.5 / 0.7 | 548 / 69 |  |
| `fit(MultivariateKDE, X2; bandwidth = PlugInMatrix())` | 0.055 / 0.0075 | 5.5 / 0.7 |  |  |
| `fit(MultivariateKDE, X2; bandwidth = SCVMatrix())` | 0.53 / 0.1 | 106 / 14 |  |  |
| `fit(PartiallyLinear, x, z, y)` | 0.013 / 0.0079 | 0.13 / 0.065 | 1.4 / 0.38 |  |

Note that preparing the data (`SmoothingData`, which sorts it) takes 0.05 seconds at
n = 10⁶, which is negligible for regression but most of the time of the density
estimates at that size. The default density estimate with the bandwidth of Sheather
and Jones, the pointwise density intervals and regression discontinuity run in a
fraction of a second even at n = 10⁶, and the manipulation test in 0.005 seconds at
n = 10⁵; their sums are taken over windows or binned. Local polynomial regression
with its plug-in bandwidths grows linearly, from 0.13 seconds at n = 10⁴ to 14 seconds
at n = 10⁶ on one thread, and eight threads make it about four times faster from
n = 10⁵ on.

Four defaults grow faster than linearly and become slow for large samples (#93),
and `LocalQuantileRegression` also grows faster than linearly (6.9 seconds at
n = 10⁵):

1. `fit(LocalLikelihood, x, b)` cross-validates over a grid of 40 bandwidths, with a
   local fit at every observation for each, so that its cost grows like n² h. It takes
   270 seconds at n = 10⁴ on one thread and 38 on eight. The rule-of-thumb bandwidth
   (`bandwidth = RuleOfThumb()`) takes 0.02 seconds at the same size.
2. `fit(ConditionalDensity, x, y)` cross-validates two bandwidths over pairs of
   observations: 182 seconds at n = 10⁴, and threads help by only a third.
3. `fit(BoundedKDE, x; method = BetaKernels())` chooses the smoothing parameter of the
   beta kernels by least-squares cross-validation, which does not use the threads: 33
   seconds at n = 10⁴.
4. `fit(MultivariateKDE, X)` evaluates its sums exactly over all pairs: 5.5 seconds at
   n = 10⁴ and 548 at n = 10⁵ on one thread, which threads cut eightfold. A binned
   engine is planned in #86.

## Against other Julia packages

On the same data and one thread (`benchmark/packages.jl`), seconds:

| call | n = 10³ | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|
| KernelDensity.jl: `kde(x)` (Silverman, FFT), then `pdf` on 512 points | 0.00018 | 0.00083 | 0.0074 | 0.081 |
| `fit(KDE, x; bandwidth = SilvermanRule())`, then `pdf` on 512 points | 0.00021 | 0.0015 | 0.010 | 0.085 |
| `fit(KDE, x)` (Sheather and Jones), then `pdf` on 512 points | 0.0018 | 0.0037 | 0.014 | 0.10 |
| `fit(KDE, x; kernel = GaussianKernel(), bandwidth = h)` (Silverman's h), then `pdf` on 512 points | 0.0036 | 0.036 | 0.36 | 3.0 |
| Loess.jl: `loess(x, y; span = 0.2)`, then `predict` at 30 points | 0.00095 | 0.0078 | 0.12 | |
| `local_polynomial(x, y, points; bandwidth = 0.1)` at 30 points | 0.000059 | 0.00045 | 0.0051 | |
| `fit(LocalPolynomialRegression, x, y; bandwidth = 0.1, points)` | 0.00036 | 0.0035 | 0.037 | |

With its default Epanechnikov kernel, the kernel density estimate is comparable with
KernelDensity.jl's binned FFT estimate at n = 10⁶ (1.05 times its time with Silverman's
bandwidth, 1.2 times with Sheather and Jones's), and within milliseconds of it at
smaller n. With the Gaussian kernel, whose support is unbounded, the sums are exact
and cost n times the number of points, about 37 times as much as KernelDensity.jl at
n = 10⁶ (#93). Local linear estimates at a given bandwidth are 20 times as fast as
Loess.jl's fit at a comparable span, and two to three times as fast with the robust
bias-corrected intervals.

## Against the R counterparts

On data from the same distributions, generated in R, and one thread each (R 4.5.3 with
the packages of the golden tests; `benchmark/timings.R`), seconds for
KernelSmoothing.jl / R:

| KernelSmoothing.jl | R | n = 10³ | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|---|
| `fit(KDE, x)` (Sheather and Jones) | `density(x, bw = "SJ")` (binned) | 0.0016 / 0.001 | 0.0021 / 0.001 | 0.0068 / 0.006 | 0.052 / 0.053 |
| `fit(KDE, x; bandwidth = LSCV())` | `bw.ucv(x)` (binned) | 0.024 / 0 | 0.26 / 0.001 | 2.7 / 0.002 |  |
| `fit(PointwiseDensity, x)` | nprobust `kdrobust` | 0.0018 / 0.016 | 0.0028 / 0.057 | 0.011 / 0.47 | 0.08 / 5.7 |
| `fit(LocalPolynomialDensity, x)` (19 points) | `lpdensity` (30 points) | 0.0039 / 0.09 | 0.0087 / 0.4 | 0.046 / 4.5 |  |
| `fit(LocalPolynomialRegression, x, y)` | nprobust `lprobust` | 0.013 / 0.36 | 0.13 / 2.8 | 1.4 / 23 | 14 / 204 |
| `fit(LocalPolynomialRegression, x, y; bandwidth = IMSEOptimal())` | `lprobust(bwselect = "imse-dpi")` | 0.013 / 0.36 | 0.13 / 2.8 | 1.4 / 23 | 13 / 202 |
| `fit(RegressionDiscontinuity, x .- 0.5, y)` | `rdrobust` | 0.00021 / 0.017 | 0.0017 / 0.13 | 0.017 / 0.96 | 0.19 / 7.9 |
| `fit(DensityDiscontinuity, x .- 0.5)` | `rddensity` | 9.9e-05 / 0.004 | 0.00053 / 0.011 | 0.0051 / 0.091 |  |
| `fit(LocalLikelihoodDensity, x)` | locfit density (`nn = 0.7`) | 0.0014 / 0.001 | 0.01 / 0.014 | 0.1 / 0.13 |  |
| `fit(LocalQuantileRegression, x, y)` (with its bandwidth) | quantreg `lprq` (given h, 30 points) | 0.027 / 0.03 | 0.33 / 0.49 | 6.9 / 39 |  |
| `fit(ConditionalDensity, x, y)` | np `npcdens` with `cv.ml` | 1.8 / 1.6 | 182 / 161 |  |  |
| `fit(BoundedKDE, x; support = (0, 1))` | evmix `fbckden` (reflection, likelihood CV) | 0.0016 / 46 | 0.003 / – | 0.023 / – | 0.32 / – |
| `fit(MultivariateKDE, X2)` (plug-in, exact) | ks `Hpi`, exact | 0.055 / 1 | 5.5 / 131 | 548 / – |  |
| (the same) | ks `Hpi`, binned | 0.055 / 1 | 5.5 / 0.77 | 548 / 0.39 | – / 0.54 |
| `fit(MultivariateKDE, X2; bandwidth = SCVMatrix())` | ks `Hscv`, exact | 0.53 / 4.8 | 106 / 563 |  |  |
| (the same) | ks `Hscv`, binned | 0.53 / 6.9 | 106 / 6.8 | – / 4.5 | – / 4.8 |

The calls are not all the same computation: `density` and `bw.ucv` bin the data, the
local polynomial density of lpdensity is evaluated at 30 points against 19, `lprq` is
given its bandwidth while `LocalQuantileRegression` chooses one, evmix's `fbckden`
chooses its bandwidth by likelihood cross-validation, and the exact bandwidth matrices
of ks are compared with the exact ones of the package. evmix's beta kernels are left
out, since `fbckden` with `bcmethod = "beta1"` did not finish in 45 minutes at
n = 1000.

Where the computations agree, the package is faster: 15 to 28 times for local
polynomial regression (nprobust), 40 to 80 times for regression discontinuity
(rdrobust), 9 to 70 times for the pointwise density intervals (nprobust's
`kdrobust`), 18 to 40 times for the manipulation test (rddensity), and 5 to 24 times
for the exact bandwidth matrices (ks). It is about as fast as locfit's density
estimate and as np's conditional density with likelihood cross-validation, both of
which grow quadratically. Two gaps remain, both because R bins the data: least-squares
cross-validation for a density, where `bw.ucv` takes milliseconds at n = 10⁵ against
2.7 seconds, and the binned bandwidth matrices of ks, which take under a second at
n = 10⁶ against minutes for the exact sums (#86).

## Where the time goes

For four defaults, we split the time on one thread into preparing the data
(`SmoothingData`), choosing the bandwidth, and estimating at the chosen bandwidth, on
a skewed sample (x = u^1.5 with u uniform) at 30 points, or 512 points for the
kernel density estimate (`benchmark/shares.jl`). Seconds:

| call | n | prepare | select | estimate | total |
|---|---|---|---|---|---|
| `LocalPolynomialRegression` | 10⁴ | 0.0004 | 0.13 | 0.0018 | 0.13 |
| `LocalPolynomialRegression` | 10⁵ | 0.0067 | 1.3 | 0.013 | 1.4 |
| `KDE`, with `pdf` on 512 points | 10⁴ | 0.0004 | 0.0014 | 0.0001 | 0.0019 |
| `KDE`, with `pdf` on 512 points | 10⁵ | 0.0038 | 0.011 | 0.0006 | 0.018 |
| `PointwiseDensity` | 10⁴ | 0.0004 | 0.0015 | 0.0001 | 0.0019 |
| `PointwiseDensity` | 10⁵ | 0.0038 | 0.012 | 0.0003 | 0.017 |
| `RegressionDiscontinuity` | 10⁴ | 0.0004 | 0.0014 | 0.0005 | 0.0017 |
| `RegressionDiscontinuity` | 10⁵ | 0.0067 | 0.011 | 0.0054 | 0.016 |

The bandwidth is the cost. For local polynomial regression it takes 96% to 97% of the
default call, and the estimates and intervals at the chosen bandwidths take about 1%;
a user who fits many curves with one bandwidth should choose it once and pass it.
For the density estimates, which are fast, the selection takes 63% to 79% of the time
and the sorting of the data about 20%. For regression discontinuity the estimate at
the chosen bandwidths includes the preparation of the data, so that the parts add up
to more than the total; there, the sorting takes 22% to 41% of the time. These
timings ran on the development allocation, a different node from the main table.

## Recommendations

- For samples beyond a few thousand, prefer `RuleOfThumb()` for `LocalLikelihood`,
  and plan for minutes with cross-validation.
- Start Julia with threads (`julia --threads=auto`) for local polynomial regression,
  local likelihood and the multivariate estimates, which use them well.
- When fitting many curves with one bandwidth, choose it once with
  `select_bandwidth` and pass it to `fit`.
