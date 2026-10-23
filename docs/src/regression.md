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
fit(LocalPolynomialRegression, x, y; degree = 2, derivative = 1)   # the derivative
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
the variance more than the nearest-neighbour estimator and HC3. At each point,
the bandwidths are raised where needed to the distance of the `minobs`-th nearest
observation, 21 by default as nprobust's `bwcheck`, and the fit records the points
where this floor binds. With ties, such a window can still hold too few distinct
values: the fit of order ``p`` needs ``p + 1`` distinct values of positive weight
and the bias correction ``p + 2``. Where they are missing, the estimates that need
them are `NaN`, so that the conventional estimate survives a missing bias
correction, and `coeftable` reports the number of distinct values at each point.

The default bandwidths follow nprobust: the plug-in bandwidth
[`MSEOptimal`](@ref) at a single point and [`IMSEOptimal`](@ref) at several, with
pilot bandwidths anchored on cross-validation; see
[Bandwidth selection](@ref bandwidths-page).

```@autodocs
Modules = [KernelSmoothing]
Pages = ["locpoly.jl"]
```

## Honest intervals

Robust bias correction removes an estimate of the bias, while [`Honest`](@ref)
intervals (Armstrong and Kolesár 2018, 2020) bound it. For the class of regression
functions whose derivative of order ``p + 1`` is bounded by ``M`` near ``x_0``, the
worst-case bias of the linear estimate ``\sum_i \ell_i y_i`` is

```math
B = \frac{M}{(p+1)!} \sum_{i=1}^n |\ell_i|\, |x_i - x_0|^{p+1},
```

since the estimate reproduces polynomials of order ``p``. The interval is
``\hat\theta \pm \mathrm{cv}(B/\widehat{\mathrm{se}})\, \widehat{\mathrm{se}}``,
where ``\mathrm{cv}(t)`` is the quantile of ``|N(t, 1)|`` at the level of the
interval, so that it covers for every function in the class. The intervals need
``M``, since adapting to unknown smoothness is impossible over such classes.
`Honest()` takes ``M`` from a global polynomial of order ``p + 3``, the rule of
thumb of Armstrong and Kolesár, which they offer as a starting point for a
sensitivity analysis. [`FLCIOptimal`](@ref) chooses the bandwidth that makes the
interval shortest, which honest intervals need: at the MSE-optimal bandwidths they
are much longer.

```julia
m = fit(LocalPolynomialRegression, x, y; points = [0.5], bandwidth = FLCIOptimal(40))
confint(m; method = Honest(40))          # covers whenever |m''| ≤ 40
confint(m; method = Honest())            # M from a global quartic, a starting point for sensitivity analysis
worstcasebias(m, 40)
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["honest.jl"]
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
