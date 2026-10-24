# KernelSmoothing.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://mlakolar.github.io/KernelSmoothing.jl/dev/)
[![CI](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/KernelSmoothing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/KernelSmoothing.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference. The package is under
development.

- `fit(KDE, x)`: a kernel density estimate that is a distribution of
  Distributions.jl (`pdf`, `cdf`, `quantile`, `rand` and the rest), computed exactly
  from the observations, with derivatives of the density and standard errors.
- `fit(PointwiseDensity, x)`: density estimates at a set of points with robust
  bias-corrected intervals (Calonico, Cattaneo and Farrell 2018), as nprobust's
  `kdrobust`.
- `fit(LocalPolynomialDensity, x)`: the density estimator of Cattaneo, Jansson and
  Ma (2020), which keeps its rate at a boundary, as R's lpdensity.
- `fit(LocalPolynomialRegression, x, y)`: local polynomial estimates of a
  regression function or its derivatives, as nprobust's `lprobust`, with
  conventional, robust bias-corrected and honest intervals (Armstrong and Kolesár),
  nearest-neighbour, HC0–HC3 and cluster-robust CR0–CR3 variances, covariances
  between points and uniform sup-t bands.
- `fit(RegressionDiscontinuity, x, y; cutoff)`: sharp, fuzzy and kink regression
  discontinuity designs, with covariates and rdrobust's bandwidth selectors, as
  rdrobust (Calonico, Cattaneo and Titiunik 2014), with the honest intervals and
  bandwidths of RDHonest, and `fit(DensityDiscontinuity, x; cutoff)`, the
  manipulation test of rddensity (Cattaneo, Jansson and Ma 2020).
- Bandwidth selectors: Sheather and Jones (the default for densities), rules of
  thumb, least-squares and leave-one-out cross-validation, the plug-in bandwidths
  of nprobust and lpdensity, coverage-error-optimal rescalings and the bandwidths
  that make honest intervals shortest.
- The layers underneath: smoothing kernels with their distribution functions,
  moments, derivatives, convolutions and products; `SmoothingData`, a sample sorted
  once with its weights and ties; and `kernelsums`, the sums all estimators are
  built from, with exact and fast engines that run in parallel with several
  threads (`julia --threads=auto`).

With given bandwidths, the estimates and standard errors agree with nprobust,
lpdensity, rdrobust, rddensity and RDHonest to rounding. The default plug-in bandwidths depart from
these packages where their pilot estimates or their squared bias fail, and
`compat = true` reproduces them; the
[documentation](https://mlakolar.github.io/KernelSmoothing.jl/dev/) explains the
differences.

```julia
using KernelSmoothing

x = randn(1000)
f = fit(KDE, x)                      # Epanechnikov kernel, Sheather–Jones bandwidth
pdf(f, 0.0), cdf(f, 0.0), quantile(f, 0.975)
pdf.(f, range(-3, 3; length = 401))  # many points at once

t = rand(2000); y = sin.(2π .* t) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, t, y; points = range(0, 1; length = 21))
coef(m), confint(m)                  # estimates and robust bias-corrected intervals
confband(m)                          # a uniform band over the points
```

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/KernelSmoothing.jl")
```
