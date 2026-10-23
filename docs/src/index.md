# KernelSmoothing.jl

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference. The package is under
development.

It provides:

- [kernel density estimates](@ref density-page) that are distributions of
  Distributions.jl, robust bias-corrected intervals for densities at interior
  points, and the local polynomial density estimator of Cattaneo, Jansson and Ma
  for densities with a boundary;
- [local polynomial regression](@ref regression-page) of a regression function or
  its derivatives, with conventional, robust bias-corrected and honest intervals,
  covariances between points and uniform bands;
- [bandwidth selectors](@ref bandwidths-page): rules of thumb, Sheather and Jones,
  cross-validation and plug-in bandwidths.

The estimators are built in layers. [Kernels](@ref kernels-page) are the smoothing
kernels with their distribution functions, moments, derivatives, convolutions and
products. [Prepared data](@ref data-page) is a sample sorted once, with its weights
and ties recorded. [Kernel sums](@ref sums-page) are the sums
``\sum_i w_i z_i K(u_i) u_i^r`` that all estimators are built from, computed by
exact direct, windowed and fast engines or approximately from binned data.

```@docs
KernelSmoothing
```

## Example

```julia
using KernelSmoothing

x = randn(1000)
f = fit(KDE, x)                      # Epanechnikov kernel, Sheather–Jones bandwidth
pdf(f, 0.0), cdf(f, 0.0), quantile(f, 0.975)
pdf.(f, range(-3, 3; length = 401))  # many points at once, exactly
derivative(f, 1)(0.5)                # the estimated derivative of the density
confint(fit(PointwiseDensity, x; points = [-1, 0, 1]))   # robust bias-corrected intervals

t = rand(2000); y = sin.(2π .* t) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, t, y; points = range(0, 1; length = 21))
coef(m), confint(m)                  # estimates and robust bias-corrected intervals
confband(m)                          # a uniform band over the 21 points
```

## Defaults and agreement with R

The estimators follow R's nprobust, lpdensity and RDHonest closely. With given
bandwidths, the estimates and standard errors agree with these packages to
rounding, as the tests check against reference values from them. The defaults are
the following.

| estimator | kernel | bandwidth | `compat = true` gives |
|---|---|---|---|
| `fit(KDE, x)` | Epanechnikov | [`SheatherJones`](@ref) | (no counterpart) |
| `fit(PointwiseDensity, x)` | Epanechnikov | plug-in, pilots anchored on Sheather–Jones | nprobust's `kdrobust` |
| `fit(LocalPolynomialRegression, x, y)` | Epanechnikov, degree 1 | plug-in, pilots anchored on cross-validation | nprobust's `lprobust` |
| `fit(LocalPolynomialDensity, x)` | triangular, degree 2 | lpdensity's plug-in, with a bound on the bias | lpdensity |

The plug-in bandwidths are pointwise ([`MSEOptimal`](@ref)) at a single point and
integrated ([`IMSEOptimal`](@ref)) at several, as in nprobust. In four places the
defaults depart from the R packages, which `compat = true` reproduces:

1. The density plug-in of nprobust estimates ``f''`` with a pilot bandwidth that
   assumes a normal density. For a bimodal density this pilot can be far too wide:
   in the valley of an equal mixture of ``N(\pm 1, 1/4)`` with 2000 observations,
   the selected bandwidth was ten times the Sheather–Jones bandwidth. We scale the
   pilot from the Sheather–Jones bandwidth instead, and use the bias constant of
   Calonico, Cattaneo and Farrell (2018), while nprobust 1.0.0 removes only 5/9 of
   the leading bias for the Epanechnikov kernel.
2. The regression plug-in of nprobust estimates the derivative in the bias with
   pilot bandwidths that, at interior points, end at their cap, the distance to the
   farther end of the data. Its bandwidths are then too large wherever the curvature
   is local, about twice the optimal ones at a sharp peak. We scale the pilot from
   the cross-validation bandwidth instead.
3. The plug-in of lpdensity minimizes a mean squared error whose squared bias
   ``(B_1 + a B_2)^2`` vanishes when the estimated terms have opposite signs, and it
   then chooses bandwidths of the order of the range of the data. We bound the bias
   by ``(|B_1| + a |B_2|)^2``.
4. The regression plug-in of nprobust has the same flaw for even ``p - \nu``, whose
   squared bias ``(B_1 + h B_2 + \lambda R)^2`` moreover adds the regularization
   ``R``, which has the units of ``B_1^2``, to ``B_1``, so that the bandwidth depends
   on the units of ``y``. We use ``(|B_1| + h |B_2|)^2 + \lambda R``. Its integrated
   rule also leaves out the regularization for odd ``p - \nu`` and has no upper
   bound, so that an exactly linear regression function on the unit interval gave a
   bandwidth of 966; we keep the regularization and cap the bandwidth at the range of
   the data.

The pages of the estimators give the details.

For all fits, `coef(m)` and `stderror(m)` describe the conventional estimates, and
`stderror(m; method = RobustBiasCorrected())` gives the robust standard errors. The
intervals of `confint(m)` and the bands of `confband(m)` are robust bias-corrected
by default and centred at the bias-corrected estimates, since conventional intervals
cover only for bandwidths well below the ones that minimize the mean squared error.
`coeftable(m)` lists both.

## Bandwidth convention

The compactly supported kernels live on `[-1, 1]`, so the bandwidth `h` is the
half-width of the smoothing window: an observation gets weight zero unless it is
within `h` of the point. The Gaussian kernel is the standard normal density, so
its bandwidth is a standard deviation. KernelDensity.jl, KDEpy and R's `density`
instead scale every kernel to unit variance; their bandwidth `h_sd` corresponds
to the half-width `h_sd / sqrt(moment(k, 2))` here, for example `h_sd √5` for the
Epanechnikov kernel.

## Threads

Kernel sums, fits at several points and cross-validation run in parallel when
Julia is started with several threads, for example with `julia --threads=auto`.
The results do not depend on the number of threads.

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/KernelSmoothing.jl")
```
