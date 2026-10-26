# Comparison with R

Most procedures of this package have a counterpart in an R package, and many users
will come from one of them. In this page, we answer two questions: what agrees with
R, and where the defaults differ and why. A table at the end translates the argument
names.

Note that agreement with R is a property we test, not one we assume. For every
counterpart below, the test suite compares the package with reference values that
an R script in `test/data` generates. The script records the version of the R
package it used, and the tolerances are those of the tests.

## What agrees with R

With `compat = true`, where the keyword exists, the procedures reproduce the R
packages, including their conventions for ties, mass points, leverages and
clusters. `DensityDiscontinuity` and the honest intervals have no `compat` keyword,
since they reproduce rddensity and RDHonest by default, with the exceptions listed
in the next section.

| R | Julia | configurations | agreement |
|---|---|---|---|
| nprobust 1.0.0, `lprobust` | `fit(LocalPolynomialRegression, x, y; compat = true)` | 9 | estimates and standard errors to 1e-10, bandwidths to 1e-14 |
| nprobust, `lpbwselect` | `MSEOptimal()`, `IMSEOptimal()`, `CoverageOptimal()` with `compat = true` | 12 | bandwidths to 1e-7, the accuracy of R's numerical minimization |
| nprobust, `lprobust` with `cluster` | `ClusterVariance(type)` | 19 | estimates and standard errors to 1e-10, bandwidths to 1e-7 |
| nprobust, `lprobust.vce` covariances | `vcov(m)` of `LocalPolynomialRegression` | 2 matrices | 1e-10 |
| nprobust, `kdrobust` and `kdbwselect` | `fit(PointwiseDensity, x; compat = true)` | 8 | 1e-12, bias-corrected estimates to 1e-11 |
| lpdensity 3.0.1, `lpdensity` and `lpbwdensity` | `fit(LocalPolynomialDensity, x; compat = true)` | 15 | 1e-12 at given bandwidths, 1e-5 with selected ones, since R integrates the kernel constants numerically |
| rdrobust 4.1.1, `rdrobust` | `fit(RegressionDiscontinuity, x, y; compat = true)` | 18 | 1e-10 |
| rdrobust, `rdbwselect` | `RDOptimal(bwselect)` | 30 | 1e-8 |
| rddensity 3.0, `rddensity` and `rdbwdensity` | `fit(DensityDiscontinuity, x)` | 19 | 1e-8, binomial tests to 1e-10 |
| RDHonest 1.0.2, `RDHonest` | `RDHonestOptimal(M)` and `confint(m; method = Honest(M))` | 17, and 24 for the rule of thumb for M | 1e-10 at given bandwidths, 1e-6 for optimized ones |
| RDHonest, local linear regression | `confint(m; method = Honest(M))` of `LocalPolynomialRegression` | 4 | 1e-10 |
| R 4.5, `stats::bw.SJ` | `SheatherJones(compat = true)` | 13 samples | 1e-10 to 1e-12 |
| R 4.5, `stats::glm` with kernel prior weights | `fit(LocalLikelihood, x, y; family, bandwidth)` | 6 | 1e-10 |
| coneproj 1.23, `coneA` with and without weights; R 4.5, `isoreg` | `shape_projection(values, points; shape, weights)` | 8 shapes | 1e-9 |
| cgam (`incr`) and scam (`bs = "mpi"`), constrained spline fits | `constrain(fit(LocalPolynomialRegression, x, y), :increasing)` | 1 | within 0.05 of cgam's step function and 0.03 of scam's fit, for a sample with noise of 0.2; a different smoother |
| ks 1.15.3, `kde` with `binned = FALSE` | `pdf(fit(MultivariateKDE, x; bandwidth, weights), points)` | d = 2, 3, with and without weights | 1e-12 |
| ks, `Hns`, `Hns.diag` (d = 2), `kfe`, `scv.mat`, `lscv.mat` | `NormalScaleMatrix`, the estimates of the functionals and the criteria | d = 2, 3 | 1e-12; `Hns.diag` to 1e-7 |
| ks, `Hpi` and `Hpi.diag` with `pilot = "samse"`, `binned = FALSE` | `PlugInMatrix(; stages, compat = true)` | d = 2, 3, one and two stages | 5e-4, the accuracy of R's numerical minimization |
| ks, `Hlscv` and `Hlscv.diag` with `binned = FALSE` | `LSCVMatrix()` | d = 2 (full), 2 and 3 (diagonal) | 2e-3, with a criterion at least as low as R's |
| evmix 2.12, `dbckden` and `pbckden` with `bcmethod = "reflect"`, `"beta1"`, `"beta2"`, `"gamma1"`, `"gamma2"`, `"logtrans"` | `fit(BoundedKDE, x; support, method, kernel, bandwidth)` | 7 | densities to 1e-12 before normalization; after it, and distribution functions, to 1e-9 to 1e-6, the accuracy of R's numerical integration |
| np 0.70.5, `npcdens` at given bandwidths | `pdf(fit(ConditionalDensity, x, y; bandwidth), y, x)` | 2 | 1e-12; np orders the bandwidths (y, x) |
| np, `npcdensbw(bwmethod = "cv.ml")` | `ConditionalCV()` | 1 | 1e-3, the accuracy of R's numerical minimization, with a criterion at least as low |
| quantreg 6.1, `lprq` and `rq` with weights | `fit(LocalQuantileRegression, x, y; tau, kernel = GaussianKernel())` and the local fits | 3 and 3 | 1e-12 |

Where agreement is not to rounding, the reason is the R package's own numerical
accuracy, such as a numerical minimization or integration with a coarse tolerance,
not a difference of method.

## Where the defaults differ

The plug-in bandwidths are pointwise ([`MSEOptimal`](@ref)) at a single point and
integrated ([`IMSEOptimal`](@ref)) at several, as in nprobust. In the following
places the defaults depart from the R packages. Each departure fixes a behavior of
the R package that we found to fail in a concrete case, and `compat = true`
reproduces the R package where the keyword exists.

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
13. For regression discontinuity, rdrobust floors ``1 - h_{ii}`` at a small
    constant for leverages near one, uses a factor of one for CR3, and falls back on
    other estimates for clusters whose block of the hat matrix is singular. By
    default, as for local polynomial regression, a leverage of one or a singular
    block leaves no residual to estimate the variance from, and the standard error
    is `NaN`, while CR3 has the factor ``(G - 1)/G``.
14. For the manipulation test, the results are `NaN` where the window of a side
    holds too few distinct values for the fit or its normal matrix is singular to
    working precision. The R package can return numbers that depend on rounding
    there. Furthermore, `-0.0` and `0.0` count as one distinct value, as in R's
    `unique()`. The pilot fits of order ``p + 2`` have ill-conditioned normal
    equations, which we sum with compensated arithmetic, so that the results agree
    with rddensity to about 1e-9, its own rounding error.
15. The honest intervals agree with RDHonest by default. Its rule of thumb for
    ``M`` needs 5 distinct values of ``x`` on each side of the cutoff, and like
    RDHonest we throw an error below that.
16. ks's two-stage plug-in matrix (`Hpi`, and `Hpi.diag`, with the pilot
    `"samse"`) reads the estimated functionals of order 6 in the wrong order when it
    chooses the pilot of order 4: it passes the whole tensor of derivatives where the
    pilot expects the distinct functionals, so that, for example, ``\psi_{51}`` takes
    the place of ``\psi_{42}`` in two dimensions. For the sample of the tests, its
    pilot was 0.723 instead of 0.653, and the matrix differed by 5%.
    [`PlugInMatrix`](@ref) reads them in the right order, and `compat = true`
    reproduces ks.
17. For ``d \geq 3``, ks's plug-in and smoothed cross-validation matrices default to
    the pilot `"dscalar"` of Chacón and Duong (2010); [`PlugInMatrix`](@ref) uses the
    SAMSE pilot of Duong and Hazelton (2003) in every dimension, which is ks's default
    for ``d = 2``. Furthermore, ks bins the data by default above 500 observations in
    two dimensions and above 1000 in three; our selectors sum over the pairs exactly.
18. [`SCVMatrix`](@ref) agrees with ks's `Hscv` in its criterion, to rounding at a
    given pilot, but its pilot follows Theorem 4 of Duong (2004), from which ks's
    differs, by about 5% for the samples of the tests; the selected matrices
    differed by 2% to 3%. Least-squares cross-validation agrees with ks's `Hlscv` in two
dimensions; in three, for the sample of the tests, ks stopped at a matrix whose
criterion was higher than at ours. For ``d \geq 3`` ks's `Hns.diag` is not the diagonal
    matrix that minimizes the normal-reference error; `NormalScaleMatrix(diagonal =
    true)` is, and agrees with ks for ``d = 2``.

19. np's least-squares cross-validation for conditional densities (`bwmethod =
    "cv.ls"`) reports a criterion that differs by about 0.6% from that of Hall, Racine
    and Li (2004) at its bandwidths, and its selection is not a minimum of that
    criterion. `ConditionalCV(criterion = :leastsquares)` minimizes the criterion of
    the paper.

## Translating the arguments

The names of the arguments differ between the R packages and the Julia keywords.
Note that the Julia functions take the running variable or covariate first, so
that rdrobust's `rdrobust(y, x)` becomes `fit(RegressionDiscontinuity, x, y)`.

**nprobust** (`lprobust`, `lpbwselect`, `kdrobust`, `kdbwselect`)

| R | Julia |
|---|---|
| `eval` | `points` |
| `p`, `deriv` | `degree`, `derivative` |
| `h`, `b`, `rho` | `bandwidth`, `pilot`, `rho` |
| `kernel = "epa"`, `"uni"`, `"tri"`, `"gau"` | `EpanechnikovKernel()`, `UniformKernel()`, `TriangularKernel()`, `GaussianKernel()` |
| `bwselect = "mse-dpi"`, `"imse-dpi"` | `MSEOptimal()`, `IMSEOptimal()` |
| `bwselect = "ce-rot"` | `CoverageOptimal(MSEOptimal())` or `CoverageOptimal(IMSEOptimal())` |
| `bwregul`, `interior` | `MSEOptimal(; regularization, interior)` |
| `vce = "nn"`, `nnmatch` | `vce = NNVariance(matches)` |
| `vce = "hc0"`, …, `"hc3"` | `vce = HCVariance(0)`, …, `HCVariance(3)` |
| `cluster` | `cluster`, with `vce = ClusterVariance(type)` |
| `level` | `confint(m; level)` |

**lpdensity** (`lpdensity`, `lpbwdensity`)

| R | Julia |
|---|---|
| `data`, `grid` | `x`, `points` |
| `p` | `degree` |
| `v` (1 for the density) | `derivative = v - 1` |
| `bw` | `bandwidth` |
| `kernel = "triangular"`, `"epanechnikov"`, `"uniform"` | `TriangularKernel()`, `EpanechnikovKernel()`, `UniformKernel()` |
| `bwselect = "mse-dpi"`, `"imse-dpi"`, `"mse-rot"` | `MSEOptimal()`, `IMSEOptimal(grid = :points)`, `NormalReference()` |
| `Pweights` | `weights` |

**rdrobust** (`rdrobust`, `rdbwselect`)

| R | Julia |
|---|---|
| `y`, `x`, `c` | `x`, `y`, `cutoff` |
| `fuzzy`, `covs` | `treatment`, `covariates` |
| `p`, `deriv` | `degree`, `derivative` |
| `h`, `b`, `rho` | `bandwidth` (a number or a pair), `pilot`, `rho` |
| `kernel = "triangular"`, `"epanechnikov"`, `"uniform"` | `TriangularKernel()`, `EpanechnikovKernel()`, `UniformKernel()` |
| `bwselect = "mserd"`, `"msetwo"`, … | `RDOptimal(:mserd)`, `RDOptimal(:msetwo)`, … |
| `vce`, `nnmatch`, `cluster` | `vce` and `cluster` as for nprobust |
| `masspoints = "adjust"`, `"check"`, `"off"` | `masspoints = :adjust`, `:check`, `:off` |
| `bwcheck`, `weights` | `bwcheck`, `weights` |

**rddensity** (`rddensity`, `rdbwdensity`)

| R | Julia |
|---|---|
| `X`, `c` | `x`, `cutoff` |
| `p`, `kernel` | `degree`, `kernel` |
| `fitselect = "unrestricted"`, `"restricted"` | `restricted = false`, `true` |
| `vce = "jackknife"`, `"plugin"` | `vce = :jackknife`, `:plugin` |
| `h` | `bandwidth`, a number or a pair |
| `bwselect = "each"`, `"diff"`, `"sum"`, `"comb"` | `RDDensityOptimal(:each)`, …, `RDDensityOptimal(:comb)` |
| `massPoints`, `regularize` | `masspoints`, `regularize` |
| `nLocalMin`, `nUniqueMin` | `minobs`, `minunique` |
| `binoN`, `binoNW` | `binomialn`, `binomialwindows` |

**RDHonest** (`RDHonest`)

| R | Julia |
|---|---|
| `y ~ x`, `y | d ~ x` | `fit(RegressionDiscontinuity, x, y)`, with `treatment = d` |
| `cutoff`, `h`, `kern` | `cutoff`, `bandwidth`, `kernel` |
| `M` | `Honest(M)` and `RDHonestOptimal(M)`; omitted, the rule of thumb `:rot` |
| `opt.criterion = "MSE"`, `"FLCI"`, `"OCI"` | `RDHonestOptimal(M; criterion = :mse)`, `:flci`, `:oci` |
| `sclass = "H"`, `"T"` | `class = :holder`, `:taylor` |
| `se.method = "nn"`, `"EHW"` | `vce = NNVariance()`, `HCVariance(0)` |
| `T0` | `RDHonestOptimal(M; T0)` |

**np** (`npcdens`, `npcdensbw`) and **quantreg** (`lprq`)

| R | Julia |
|---|---|
| `npcdens(txdat = x, tydat = y, bws = c(b, h))`, the bandwidths ordered (y, x) | `fit(ConditionalDensity, x, y; bandwidth = (h, b))`, ordered (x, y) |
| `exdat`, `eydat` | `pdf(m, y, x)` at each pair |
| `npcdensbw(bwmethod = "cv.ml")`, `"cv.ls"` | `ConditionalCV()`, `ConditionalCV(criterion = :leastsquares)` |
| `lprq(x, y, h, tau, m)` | `fit(LocalQuantileRegression, x, y; bandwidth = h, tau, kernel = GaussianKernel(), points = range(minimum(x), maximum(x); length = m))` |
| `lprq(...)$dv` | the same with `derivative = 1, degree = 1` |
