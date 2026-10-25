# [Local polynomial regression](@id regression-page)

`fit(LocalPolynomialRegression, x, y)` fits local polynomials at a set of points
and estimates the regression function or one of its derivatives, with
conventional and robust bias-corrected standard errors and confidence intervals
(Calonico, Cattaneo and Farrell 2018). With given bandwidths, the results agree
with nprobust to rounding.

```julia
using KernelSmoothing

x = rand(2000); y = sin.(2π .* x) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, x, y; points = range(0, 1; length = 21))
coef(m)                                  # the estimates at the points
confint(m)                               # robust bias-corrected 95% intervals
confint(m; method = Conventional())      # intervals that ignore the bias
predict(m, [0.25, 0.5])                  # estimates at new points
coeftable(m)                             # everything, as a table of columns
fit(LocalPolynomialRegression, x, y; derivative = 1)              # the derivative, by local quadratics
```

## The estimator

Let ``(x_i, y_i)`` be the observations with weights ``w_i``, ``K`` a kernel and
``h`` a bandwidth. At a point ``x_0``, the fit of order ``p`` minimizes

```math
\sum_{i=1}^n w_i K\Big(\frac{x_i - x_0}{h}\Big) \Big( y_i - \sum_{j=0}^p \beta_j (x_i - x_0)^j \Big)^2 ,
```

and ``\nu!\, \hat\beta_\nu`` estimates the derivative ``m^{(\nu)}(x_0)`` of the
regression function for ``0 \leq \nu \leq p``. In the scaled variable
``u = (x - x_0)/h``, with ``r_p(u) = (1, u, \dots, u^p)'`` and the Gram matrix
``\Gamma_p = \sum_i w_i K(u_i) r_p(u_i) r_p(u_i)'``, the estimate is linear in the
responses, ``\sum_i \ell_i y_i`` with
``\ell_i = (\nu!/h^\nu)\, w_i K(u_i)\, e_\nu' \Gamma_p^{-1} r_p(u_i)``.

The leading bias of the fit is ``\Gamma_p^{-1} \Lambda\, \beta_{p+1}``, where
``\Lambda = \sum_i w_i K(u_i) u_i^{p+1} r_p(u_i)`` and ``\beta_{p+1}`` is the next
Taylor coefficient (Calonico, Cattaneo and Titiunik 2014). We estimate
``\beta_{p+1}`` by a fit of order ``q = p + 1`` with the pilot bandwidth ``b``, in
the variable ``v = (x - x_0)/b``. The bias-corrected estimate is then
``\sum_i \ell_i^{\mathrm{bc}} y_i`` with

```math
\ell_i^{\mathrm{bc}} = \ell_i - \frac{\nu!}{h^\nu} \Big(\frac{h}{b}\Big)^{p+1} (g' \Lambda)\, w_i K(v_i)\, c' r_q(v_i),
\qquad g = \Gamma_p^{-1} e_\nu, \qquad c = \Gamma_q^{-1} e_{p+1} ,
```

where ``\Gamma_q`` is the Gram matrix of the second fit. The standard errors are
``(\sum_i \ell_i^2 \hat\sigma_i^2)^{1/2}`` and
``(\sum_i (\ell_i^{\mathrm{bc}})^2 \hat\sigma_i^2)^{1/2}``, so that the robust one
includes the variability of the bias estimate and its covariance with the estimate.
By default ``b = h``, as in nprobust, and then the bias-corrected estimate is the
fit of order ``p + 1`` at ``h``.

The conditional variances ``\hat\sigma_i^2`` come from [`NNVariance`](@ref), the
default, which compares each response with the mean of its nearest neighbours in
the window, or from [`HCVariance`](@ref), which uses the residuals of the local fits
with the corrections HC0 to HC3. In small samples at the boundary, HC0 understates
the variance more than the nearest-neighbour estimator and HC3.

For observations that are independent across clusters but may be dependent within
them, pass the cluster of each observation as `cluster`. The default variance
estimator is then [`ClusterVariance`](@ref)`(1)`, the CR1 estimator of nprobust,
which squares the sums of ``\ell_i \hat\varepsilon_i`` over the clusters in the
window instead of the single products, with the factor
``(n - 1) / (n - k) \cdot G / (G - 1)`` for ``n`` observations and ``G`` clusters in
the window and ``k`` coefficients. Types 2 and 3 adjust the residuals of each
cluster with its block of the hat matrix, as Bell and McCaffrey (2002) do for CR2,
and type 0 leaves them as they are. The covariances of [`vcov`](@ref) and the bands
of [`confband`](@ref) sum over clusters in the same way, and so do the variance
constants of the plug-in bandwidths. With one observation per cluster, types 0 to 2
reduce to [`HCVariance`](@ref) of the same type.

At each point,
the bandwidths are raised where needed to the distance of the `minobs`-th nearest
observation, 21 by default as nprobust's `bwcheck`. With ties, such a window can
still hold too few distinct values: the fit of order ``p`` needs ``p + 1`` distinct
values of positive weight and the bias correction ``p + 2``. Where they are
missing, the bandwidth is raised further, to 1.5 times the distance of the farthest
distinct value it needs, which puts that value well inside the window; with
`compat = true`, as in nprobust, it is not. The fit records the points where these
floors bind. Estimates that still lack distinct values, as with `minobs = 0` or
with too few distinct values in the data, are `NaN`, so that the conventional
estimate survives a missing bias correction, and `coeftable` reports the number of
distinct values at each point. The default degree is `derivative + 1`, as in
nprobust, since an odd ``p - \nu`` has the same order of bias at the boundary as
in the interior.

The default bandwidths follow nprobust: the plug-in bandwidth
[`MSEOptimal`](@ref) at a single point and [`IMSEOptimal`](@ref) at several, with
pilot bandwidths anchored on cross-validation; see
[Bandwidth selection](@ref bandwidths-page).

## Local polynomial smoothing

For smoothing many curves, or for building other estimators on local fits,
[`localpolynomial`](@ref) gives the estimates of the local polynomial fits at a set
of points with a given bandwidth, without the floors on the windows and without
standard errors, from [`kernelsums`](@ref) for all points at once.
[`smoothermatrix`](@ref) gives their weights: each estimate is a weighted sum of
the responses, `L * y`, with columns in the input order of the observations, so
that linear functionals of the fit, hat matrices and their traces follow directly.

```julia
e = localpolynomial(t, y, grid; bandwidth = 0.1)                 # a smoothed curve on a grid
L = smoothermatrix(t, t; bandwidth = 0.1)                        # the hat matrix of the observations
gcv = length(y) * sum(abs2, y - L * y) / (length(y) - sum(LinearAlgebra.diag(L)))^2
```

For a regression function of two covariates, as the covariance surfaces of
functional data, `localpolynomial((x1, x2), y, (g1, g2); bandwidth)` gives the
bivariate local constant or local linear estimates on a grid, with a product kernel
and a bandwidth for each covariate, and `smoothermatrix((x1, x2), points)` their
weights. `selectbandwidth(CrossValidation(), (x1, x2), y)` chooses the bandwidth by
leave-one-out cross-validation, or with `groups`, as the curves of functional data,
by leaving out whole groups, so that the dependence within a curve does not lead to
too small bandwidths.

```julia
s = selectbandwidth(CrossValidation(), (t1, t2), products; groups = curve)
C = localpolynomial((t1, t2), products, (grid, grid); bandwidth = s.h)   # a covariance surface
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["locpoly.jl", "surface.jl"]
```

## Partially linear models

[`PartiallyLinear`](@ref) fits `y = x'β + g(u) + ε` by the double residuals of
Robinson (1988): the regression of `y` on `X` after removing from both their local
polynomial smooths in `u`, which removes `g` without estimating it first, so that
`β̂` is `√n`-consistent with sandwich standard errors; `ĝ` is then the smooth of
`y - x'β̂`.

Varying-coefficient models, with or without penalties, are in [CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl),
which uses the kernels of this package.

```julia
p = fit(PartiallyLinear, u, X, y)
coeftable(p)                                           # β̂ with standard errors; p.smooth is ĝ at p.points
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["semiparametric.jl"]
```

## Local likelihood

For binary or count responses, [`LocalLikelihood`](@ref) fits a local polynomial in
the linear predictor of a logistic (`family = :logit`) or Poisson (`:poisson`)
model by maximizing the kernel-weighted log likelihood at each point (Tibshirani and
Hastie 1987; Fan, Heckman and Wand 1995), which keeps the estimates of
probabilities in ``(0, 1)`` and of means positive. The standard errors are the
sandwich ones of the local fit, intervals for the mean transform those for the
linear predictor, and [`CrossValidation`](@ref) maximizes the leave-one-out log
likelihood, from one Newton step without each observation. At given bandwidths the
estimates and standard errors are those of R's `glm` with the kernel weights as
prior weights.

```julia
m = fit(LocalLikelihood, x, y; family = :logit)    # y in {0, 1}
predict(m), confint(m)                             # probabilities at the points, with intervals
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["locallik.jl"]
```

## Honest intervals

Robust bias correction removes an estimate of the bias, while [`Honest`](@ref)
intervals (Armstrong and Kolesár 2018, 2020) bound it. The Taylor class bounds the
remainder of the order-``p`` Taylor polynomial at ``x_0`` by
``M |x - x_0|^{p+1}/(p+1)!`` throughout the smoothing windows. Its worst-case bias
for the linear estimate ``\sum_i \ell_i y_i`` is

```math
B = \frac{M}{(p+1)!} \sum_{i=1}^n |\ell_i|\, |x_i - x_0|^{p+1},
```

since the estimate reproduces polynomials of order ``p``. The interval is
``\hat\theta \pm \mathrm{cv}(B/\widehat{\mathrm{se}})\, \widehat{\mathrm{se}}``,
where ``\mathrm{cv}(t)`` is the quantile of ``|N(t, 1)|`` at the level of the
interval. With estimated standard errors, coverage is asymptotic under the normal
approximation and its regularity conditions. With fixed weights, Gaussian errors
and the true standard error, the coverage bound is finite-sample. A bound ``M`` on
the derivative of order ``p + 1`` throughout the windows implies the Taylor
remainder bound; over this smaller class, the displayed bias bound can be
conservative. The intervals need
``M``, chosen before seeing the estimates, since adapting to unknown smoothness is
impossible over such classes, and [`Honest`](@ref) has no default for it.
`Honest(:rot)` takes ``M`` from a global polynomial of order ``p + 3``, the rule of
thumb of Armstrong and Kolesár, which they offer as a starting point for a
sensitivity analysis. [`FLCIOptimal`](@ref) chooses the bandwidth that makes the
interval shortest. The plug-in bandwidths of [`MSEOptimal`](@ref) estimate the bias
rather than bound it, and where it is small, as where the curvature vanishes, they
are much larger than the honest intervals can afford, which then become much longer.
The bandwidth that minimizes the worst-case mean squared error over the class, in
contrast, gives intervals only slightly longer than the shortest ones (Armstrong
and Kolesár 2020).

```julia
m = fit(LocalPolynomialRegression, x, y; points = [0.5], bandwidth = FLCIOptimal(40))
confint(m; method = Honest(40))          # covers whenever |m''| ≤ 40
confint(m; method = Honest(:rot))        # M from a global quartic, a starting point for sensitivity analysis
worstcasebias(m, 40)
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["src/honest.jl"]   # (not rdhonest.jl)
```

## Uniform bands

[`vcov`](@ref) gives the covariances of the estimates at the points. Each estimate
is linear in the data, and its standard error is the norm of the vector ``a_j``
with entries ``a_{ij} = \ell_{ij} \hat\varepsilon_{ij}``, the weight of observation
``i`` at point ``j`` times its residual from the variance estimator. The covariances
are the inner products ``a_j' a_l``, so that the matrix is positive semidefinite,
its diagonal holds the squared standard errors of the fit, and estimates whose
windows do not overlap are uncorrelated.

[`confband`](@ref) returns the sup-t band ``\hat\theta_j \pm c\, \widehat{\mathrm{se}}_j``
of Chernozhukov, Chetverikov and Kato (2014), where ``c`` is the quantile of
``\max_j |Z_j|`` for a Gaussian vector ``Z`` with the estimated correlations, found
by simulation. It covers the curve at all points at once, for regression and density
fits alike. Pointwise intervals used as a band cover far less often than their
level.

```julia
m = fit(LocalPolynomialRegression, x, y; points = range(0, 1; length = 50))
b = confband(m)              # robust bias-corrected 95% band
b.lower, b.upper, b.critical
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["bands.jl"]
```
