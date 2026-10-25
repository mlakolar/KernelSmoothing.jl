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

The plug-in bandwidths are pointwise ([`MSEOptimal`](@ref)) at a single point and
integrated ([`IMSEOptimal`](@ref)) at several, as in nprobust. In the following
places the defaults depart from the R packages, which `compat = true` reproduces:

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
5. The floor of nprobust raises the bandwidths at each point to the distance of the
   21st nearest observation. With heavy ties, such a window can hold fewer distinct
   values than the fit has coefficients, and the estimate is then undefined: with
   four distinct values of ``x``, the default fit left 12 of its 30 estimates and
   all bias corrections `NaN`. We raise the bandwidths further where needed, so
   that the window of ``h`` holds ``p + 1`` distinct values with positive weight
   and that of the pilot ``b`` holds ``p + 2``.
6. The nearest-neighbour variance estimator of nprobust takes the neighbours on both
   sides when they are at the same distance, and compares the two distances exactly.
   On an equally spaced design, rounding then decides which neighbours count, and a
   shift of the design changed the standard errors by up to 9 percent. We treat
   distances that agree to about eight digits as equal.
7. The coverage-error optimal bandwidths of nprobust rescale the plug-in bandwidths
   to the rate that is optimal at the boundary, also at interior points, which
   shrinks local linear bandwidths by ``n^{-1/20}``. For odd ``p - \nu`` the optimal
   rate at interior points is that of the mean squared error, and
   [`CoverageOptimal`](@ref) uses it by default, so that it leaves local linear
   bandwidths unchanged.
8. The bandwidth rules of lpdensity standardize the data by their mean and standard
   deviation: one outlier at ``10^4`` among 2000 normal observations moved its rule
   of thumb at 0 from 0.55 to 124. We standardize by the median and a robust scale.
   Its pilot bandwidths are integrated over the evaluation points, so that one far
   point changes the bandwidths at all the others, and ours over the quantiles of
   the data. Its rule of thumb jumps towards the range of the data at the inflection
   points of the normal density, where the leading bias vanishes, and ours adds the
   next term of the bias.
9. nprobust gives a standard error of zero where no observation has positive weight,
   lpdensity extrapolates to points far outside the data, and both let intervals for
   a density reach below zero. We return `NaN` in the first two cases and clamp the
   lower ends at zero.
10. R's `bw.SJ` takes the smallest root of the equation of Sheather and Jones. With
    rounded data, pairs of equal values create roots at bandwidths that resolve the
    grid of the values: for `10^5` normal observations rounded to 0.1 it gave 0.0026,
    at which the estimate is a comb of spikes. [`SheatherJones`](@ref) searches only
    bandwidths that do not resolve the grid, and gives 0.106 there.
11. nprobust evaluates regression by default at 30 equally spaced points between the
    smallest and the largest observation, and its integrated bandwidth averages over
    such points. For a covariate with heavy tails, most of them lie where few
    observations are left. We evaluate between the quantiles of order 0.025 and
    0.975, and average over 30 quantiles of order 0.05 to 0.95.
12. For clustered data, nprobust adjusts the residuals of the bias-corrected
    estimate for CR2 and CR3 with a matrix that is not a block of a hat matrix and
    changes with the units of ``x``: at the boundary, measuring ``x`` in other units
    changed these standard errors by five orders of magnitude. We adjust them with
    the hat matrix of the fit of order ``p``, as for HC2 and HC3. Furthermore,
    nprobust's covariances between points ignore the clusters off the diagonal;
    ours sum over clusters also with `compat = true`.

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
- R's `stats` package (R Core Team): `bw.SJ`, the bandwidth of Sheather and Jones
  (1991), *Journal of the Royal Statistical Society, Series B* 53(3), and `glm`,
  against which the local likelihood fits are checked.
