# KernelSmoothing.jl

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference. The package is under
development.

It provides:

- [kernel density estimates](@ref density-page) that are distributions of
  Distributions.jl, robust bias-corrected intervals for densities at interior
  points, and the local polynomial density estimator of Cattaneo, Jansson and Ma
  for densities with a boundary;
- [multivariate kernel density estimates](@ref multivariate-page) with a bandwidth
  matrix, chosen by normal scale, plug-in, smoothed or least-squares
  cross-validation rules;
- [density estimates on a bounded support](@ref bounded-page) that are
  distributions there: reflection, beta and gamma kernels, and transformations;
- [local polynomial regression](@ref regression-page) of a regression function or
  its derivatives, with conventional, robust bias-corrected and honest intervals,
  covariances between points and uniform bands;
- [conditional densities and local polynomial quantile regression](@ref conditional-page);
- [shape constraints](@ref shape-page): rearranged estimates and bands, and
  projections onto monotone and convex sequences;
- [regression discontinuity](@ref rd-page) estimates for sharp, fuzzy and kink
  designs, with covariates, and the manipulation test of the density at the cutoff;
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
conf_band(m)                          # a uniform band over the 21 points
```

## Defaults and agreement with R

The estimators follow R's nprobust, lpdensity and RDHonest closely. With given
bandwidths, the estimates and standard errors agree with these packages to
rounding, as the tests check against reference values from them. The defaults are
the following.

| estimator | kernel | bandwidth | `compat = true` gives |
|---|---|---|---|
| `fit(KDE, x)` | Epanechnikov | [`SheatherJones`](@ref) | `bandwidth = SheatherJones(compat = true)`: R's `bw.SJ` |
| `fit(PointwiseDensity, x)` | Epanechnikov | plug-in, pilots anchored on Sheather–Jones | nprobust's `kdrobust` |
| `fit(LocalPolynomialRegression, x, y)` | Epanechnikov, degree `derivative + 1` | plug-in, pilots anchored on cross-validation | nprobust's `lprobust` |
| `fit(LocalPolynomialDensity, x)` | triangular, degree 2 | lpdensity's plug-in, with a bound on the bias | lpdensity |

The page [Comparison with R](r.md) lists what the tests verify against each R package,
explains every place where the defaults depart from it, and translates the argument names.

The pages of the estimators give the details.

For all fits, `coef(m)` and `stderror(m)` describe the conventional estimates, and
`coef(m; method = RobustBiasCorrected())` and `stderror(m; method = RobustBiasCorrected())`
give the bias-corrected estimates and their robust standard errors. The
intervals of `confint(m)` and the bands of `conf_band(m)` are robust bias-corrected
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

## Plotting

Loading Plots.jl, or another package that uses RecipesBase, adds plot recipes.
`plot(m)` draws the estimates of a pointwise fit ([`LocalPolynomialRegression`](@ref),
[`PointwiseDensity`](@ref) or [`LocalPolynomialDensity`](@ref)) over its shaded
robust bias-corrected intervals, and `plot(f)` the density estimate of a
[`KDE`](@ref). The keywords `level` and `method` choose the intervals as for
[`confint`](@ref), `uniform = true` draws the uniform band of [`conf_band`](@ref)
instead, and `data = true` adds the observations of a regression.

```julia
using KernelSmoothing, Plots
m = fit(LocalPolynomialRegression, x, y)
plot(m; data = true)
plot(m; uniform = true, level = 0.9)
```

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

## Development tests

From a master checkout's package root, run:

```sh
julia scripts/test_master.jl
```

This prepares the dedicated master-based test environment, including test extras
and sibling workspace dependencies, and runs the complete suite. Package version
numbers and dependency declarations remain unchanged.

## Acknowledgements

KernelSmoothing.jl is an independent implementation. We wrote its procedures from
the papers that proposed them, and we validated them against the R packages below,
whose authors made these methods available to applied researchers. With
`compat = true`, the results agree with these packages on the configurations in
the test suite; the defaults depart from them where the documentation says so. If
you use the package, please cite the papers of the methods you use.

- [nprobust](https://nppackages.github.io/nprobust/) (Calonico, Cattaneo and
  Farrell): local polynomial regression and kernel density estimation with robust
  bias-corrected inference. Calonico, Cattaneo and Farrell (2018), *Journal of the
  American Statistical Association* 113(522); (2019), *Journal of Statistical
  Software* 91(8); (2022), *Bernoulli* 28(4).
- [rdrobust](https://rdpackages.github.io/rdrobust/) (Calonico, Cattaneo, Farrell
  and Titiunik): regression discontinuity designs. Calonico, Cattaneo and Titiunik
  (2014), *Econometrica* 82(6); Calonico, Cattaneo, Farrell and Titiunik (2017),
  *Stata Journal* 17(2); (2019), *Review of Economics and Statistics* 101(3).
- [rddensity](https://rdpackages.github.io/rddensity/) (Cattaneo, Jansson and Ma):
  manipulation tests. Cattaneo, Jansson and Ma (2018), *Stata Journal* 18(1);
  (2020), *Journal of the American Statistical Association* 115(531).
- [lpdensity](https://nppackages.github.io/lpdensity/) (Cattaneo, Jansson and Ma):
  local polynomial density estimation. Cattaneo, Jansson and Ma (2020), *Journal of
  the American Statistical Association* 115(531); (2022), *Journal of Statistical
  Software* 101(2).
- [RDHonest](https://github.com/kolesarm/RDHonest) (Kolesár): honest confidence
  intervals. Armstrong and Kolesár (2018), *Econometrica* 86(2); (2020),
  *Quantitative Economics* 11(1); Kolesár and Rothe (2018), *American Economic
  Review* 108(8).
- [ks](https://cran.r-project.org/package=ks) (Duong): multivariate kernel density
  estimation. Duong (2007), *Journal of Statistical Software* 21(7).
- [evmix](https://cran.r-project.org/package=evmix) (Hu and Scarrott): boundary
  corrected kernel density estimation. Hu and Scarrott (2018), *Journal of
  Statistical Software* 84(5).
- [np](https://cran.r-project.org/package=np) (Hayfield and Racine): conditional
  density estimation. Hayfield and Racine (2008), *Journal of Statistical Software*
  27(5); Hall, Racine and Li (2004), *Journal of the American Statistical
  Association* 99(468).
- [quantreg](https://cran.r-project.org/package=quantreg) (Koenker): quantile
  regression. Koenker (2005), *Quantile Regression*, Cambridge University Press; Yu
  and Jones (1998), *Journal of the American Statistical Association* 93(441).
- R's `stats` package (R Core Team): `bw.SJ`, the bandwidth of Sheather and Jones
  (1991), *Journal of the Royal Statistical Society, Series B* 53(3), and `glm`,
  against which the local likelihood fits are checked.
