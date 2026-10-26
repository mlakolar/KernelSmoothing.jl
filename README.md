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
- `fit(MultivariateKDE, x)`: a kernel density estimate in several dimensions with
  a bandwidth matrix, a distribution of Distributions.jl, with the normal scale,
  plug-in, smoothed and least-squares cross-validation matrices of R's ks.
- `fit(BoundedKDE, x; support, method)`: a density estimate on `[a, b]`, `[a, ∞)` or `(-∞, b]`
  that is a distribution there, by reflection, the beta and gamma kernels of Chen
  (1999, 2000), or a transformation, as R's evmix.
- `fit(LocalPolynomialRegression, x, y)`: local polynomial estimates of a
  regression function or its derivatives, as nprobust's `lprobust`, with
  conventional, robust bias-corrected and honest intervals (Armstrong and Kolesár),
  nearest-neighbour, HC0–HC3 and cluster-robust CR0–CR3 variances, covariances
  between points and uniform sup-t bands.
- `fit(RegressionDiscontinuity, x, y; cutoff)`: sharp, fuzzy and kink regression
  discontinuity designs, with covariates and rdrobust's bandwidth selectors, as
  rdrobust (Calonico, Cattaneo and Titiunik 2014), with the honest intervals and
  bandwidths of RDHonest, and `fit(DensityDiscontinuity, x; cutoff)`, the
  manipulation test of rddensity (Cattaneo, Jansson and Ma 2020); designs with
  several cutoffs (`MultipleCutoffs`, `CumulativeCutoffs`) as rdmulti, power
  calculations (`rd_power`, `rd_samplesize`) as rdpower, and randomization
  inference in a window around the cutoff (`LocalRandomization`) as rdlocrand.
- `fit(PartiallyLinear, u, X, y)`: partially linear models (Robinson's double
  residuals), with sandwich standard errors and cross-validated bandwidths.
  Varying-coefficient models are in HighDimensionalModels.jl, on the kernels of this
  package.
- `fit(LocalLikelihood, x, y; family)`: local likelihood regression for binary, count
  (Poisson and negative binomial), positive (gamma and inverse Gaussian) and
  Gaussian responses, with robust bias-corrected intervals and the rule-of-thumb
  bandwidth of Fan, Heckman and Wand (1995); `fit(LocalLikelihoodDensity, x)`, the
  local likelihood density estimate of Loader (1996), as R's locfit.
- `fit(ConditionalDensity, x, y)`: kernel estimates of the conditional density of
  `y` given `x` (local constant and local linear), with likelihood and least-squares
  cross-validation, as R's np; `fit(LocalQuantileRegression, x, y; tau)`: local
  polynomial quantile regression with exact local fits, as quantreg's `lprq`.
- `rearrange(m)` and `constrain(m, shape)`: monotone estimates and bands by
  rearrangement (Chernozhukov, Fernández-Val and Galichon 2009), and projections of
  the estimates onto monotone, convex or concave sequences.
- `local_polynomial` and `smoother_matrix`: local polynomial estimates and their
  weights at given points and bandwidths, without inference, for smoothing many
  curves and for building other estimators, also for two covariates (covariance
  surfaces), with leave-one-out or leave-group-out cross-validation.
- `kernel_weights`, `regression_points`, `bandwidth_grid`, `grid_minimize`,
  `sandwich` and `supt_critical`: the building blocks of the estimators (kernel
  weights, default points and bandwidth grids, the search over a grid, robust
  variances of local fits and the critical value of bands), for other packages.
- Bandwidth selectors: Sheather and Jones (the default for densities), rules of
  thumb, least-squares and leave-one-out cross-validation, the plug-in bandwidths
  of nprobust and lpdensity, coverage-error-optimal rescalings and the bandwidths
  that make honest intervals shortest.
- The layers underneath: smoothing kernels with their distribution functions,
  moments, derivatives, convolutions and products; `SmoothingData`, a sample sorted
  once with its weights and ties; and `kernel_sums`, the sums all estimators are
  built from, with exact and fast engines that run in parallel with several
  threads (`julia --threads=auto`).

With given bandwidths, the estimates and standard errors agree with nprobust,
lpdensity, rdrobust, rddensity and RDHonest to rounding. The default plug-in bandwidths depart from
these packages where their pilot estimates or their squared bias fail, and
`compat = true` reproduces them; the
[documentation](https://mlakolar.github.io/KernelSmoothing.jl/dev/) explains the
differences. The [notes](notes/README.md) report how well the intervals cover in
simulations, how long each procedure takes, and which settings we recommend.

```julia
using KernelSmoothing

x = randn(1000)
f = fit(KDE, x)                      # Epanechnikov kernel, Sheather–Jones bandwidth
pdf(f, 0.0), cdf(f, 0.0), quantile(f, 0.975)
pdf.(f, range(-3, 3; length = 401))  # many points at once

t = rand(2000); y = sin.(2π .* t) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, t, y; points = range(0, 1; length = 21))
coef(m), confint(m)                  # estimates and robust bias-corrected intervals
conf_band(m)                          # a uniform band over the points
```

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/KernelSmoothing.jl")
```

## Development and testing

Use a `master` checkout for this package and its workspace dependencies. From
the package root, run the complete test suite with:

```sh
julia scripts/test_master.jl
```

The runner resolves sibling master checkouts and test extras in a dedicated
`.master-test-env` environment, leaving the package dependency declarations
unchanged. Missing workspace dependencies are cloned from their master branches.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the naming and development policy.

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
- [rdmulti](https://rdpackages.github.io/rdmulti/), [rdpower](https://rdpackages.github.io/rdpower/)
  and [rdlocrand](https://rdpackages.github.io/rdlocrand/) (Cattaneo, Titiunik and
  Vazquez-Bare): several cutoffs, power calculations and local randomization.
  Cattaneo, Keele, Titiunik and Vazquez-Bare (2016), *Journal of Politics* 78(4);
  Cattaneo, Titiunik and Vazquez-Bare (2020), *Stata Journal* 20(4); (2019), *Stata
  Journal* 19(1); Cattaneo, Frandsen and Titiunik (2015), *Journal of Causal
  Inference* 3(1); Cattaneo, Titiunik and Vazquez-Bare (2016), *Stata Journal* 16(2).
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
- [locfit](https://cran.r-project.org/package=locfit) (Loader): local regression,
  likelihood and density estimation. Loader (1996), *Annals of Statistics* 24(4);
  Loader (1999), *Local Regression and Likelihood*, Springer.
- [MASS](https://cran.r-project.org/package=MASS) (Venables and Ripley): the negative
  binomial family. Venables and Ripley (2002), *Modern Applied Statistics with S*,
  Springer.
- R's `stats` package (R Core Team): `bw.SJ`, the bandwidth of Sheather and Jones
  (1991), *Journal of the Royal Statistical Society, Series B* 53(3), and `glm`,
  against which the local likelihood fits are checked.
