# KernelSmoothing.jl

[![CI](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/KernelSmoothing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/KernelSmoothing.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

The package is under development. So far it provides kernel density estimates
and the pieces that all estimators build on:

- `fit(LocalPolynomialRegression, x, y)`, local polynomial estimates of a
  regression function or its derivatives at a set of points, with conventional
  and robust bias-corrected standard errors and intervals that agree with
  nprobust, nearest-neighbour and HC0–HC3 variances, floors on the windows that
  are reported where they bind, and a rule-of-thumb bandwidth;
- `fit(KDE, x)`, a kernel density estimate that is a distribution of
  Distributions.jl: `pdf`, `cdf`, `quantile`, `rand` and the rest, computed
  exactly from the observations, with derivatives of the density, pointwise
  standard errors and rule-of-thumb bandwidths;
- smoothing kernels: the uniform, triangular, Epanechnikov, biweight, triweight
  and Gaussian kernels, with their distribution functions, moments, roughness,
  derivatives, convolutions and products;
- `SmoothingData`, a sample sorted once, with its weights and ties recorded, that
  answers local questions fast: which observations lie within a bandwidth of a
  point, how many distinct values they take, what they weigh, and how far away
  the `k`-th nearest observation is;
- `kernelsums`, the sums `∑ wᵢ zᵢ K(uᵢ) uᵢʳ` that density estimates, local
  polynomial fits and cross-validation are built from, with one bandwidth or one
  per point, exact direct, windowed and fast engines and an approximate binned
  one. They run in parallel when Julia is started with several threads
  (`julia --threads=auto`).

```julia
using KernelSmoothing
f = fit(KDE, randn(1000))            # Epanechnikov kernel, normal-reference bandwidth
pdf(f, 0.0), cdf(f, 0.0), quantile(f, 0.975)
pdf.(f, range(-3, 3; length = 401))   # many points at once

x = rand(2000); y = sin.(2π .* x) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, x, y; points = range(0, 1; length = 21))
coef(m), confint(m)                   # estimates and robust bias-corrected intervals
```

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/KernelSmoothing.jl")
```
