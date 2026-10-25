# [Regression discontinuity](@id rd-page)

`fit(RegressionDiscontinuity, x, y; cutoff)` estimates the effect of a treatment
assigned by whether a running variable reaches a cutoff, from local polynomials on
the two sides of the cutoff, with conventional and robust bias-corrected inference
(Calonico, Cattaneo and Titiunik 2014). It covers sharp, fuzzy and kink designs and
adjustment for covariates, and selects its bandwidths by the plug-in rules of
rdrobust. The estimates, standard errors and bandwidths agree with rdrobust to
rounding. Honest intervals and their bandwidths are those of RDHonest, and
`fit(DensityDiscontinuity, x; cutoff)` gives the manipulation test of rddensity.
Designs with several cutoffs, power calculations and inference under local
randomization follow rdmulti, rdpower and rdlocrand.

```julia
using KernelSmoothing

x = 2 .* rand(2000) .- 1
y = 0.5 .+ x .+ 0.4 .* (x .>= 0) .+ 0.3 .* randn(2000)
m = fit(RegressionDiscontinuity, x, y; cutoff = 0)
coef(m)                                  # the estimated jump
confint(m)                               # the robust bias-corrected 95% interval
coeftable(m)                             # conventional, bias-corrected and robust results
bandwidth(m)                             # the MSE-optimal bandwidths on the two sides
fit(RegressionDiscontinuity, x, y; derivative = 1)            # the jump in the slope
fit(RegressionDiscontinuity, x, y; treatment = t)             # a fuzzy design
fit(RegressionDiscontinuity, x, y; covariates = Z)            # adjusted for covariates
fit(RegressionDiscontinuity, x, y; bandwidth = RDOptimal(:cerrd))   # coverage-error rate
fit(RegressionDiscontinuity, x, y; bandwidth = 0.3)           # a given bandwidth
```

## The estimator

The observations with ``x_i < c`` form the left side and the others the right
side. On each side, the local polynomial fit of order ``p`` at ``c`` with bandwidth
``h`` and the fit of order ``p + 1`` with the pilot bandwidth ``b`` are those of
[Local polynomial regression](@ref regression-page), with the data of that side only,
so that they extrapolate to the cutoff from one side. Let ``\hat\beta_{+,\nu}`` and
``\hat\beta_{-,\nu}`` be the coefficients of ``(x - c)^\nu`` on the right and on the
left. In a sharp design, the estimate is

```math
\hat\tau = \nu! \, (\hat\beta_{+,\nu} - \hat\beta_{-,\nu}) ,
```

the jump in the regression function for `derivative = 0` and the jump in its slope
for `derivative = 1`, a kink design. The bias-corrected estimate subtracts the
estimated bias on each side, and its robust standard error includes the variability
of that estimate. The two sides are independent, so that the variances add.

In a fuzzy design, the treatment received is passed as `treatment`, and the
estimate is the ratio ``\hat\tau_Y / \hat\tau_T`` of the jumps in the outcome and in
the treatment. Its standard error comes from the delta method: it is the standard
error of the estimate for the combination ``(Y - \hat\tau T) / \hat\tau_T`` of the
outcome and the treatment, and `m.firststage` holds the jump in the treatment with
its standard errors.

With `covariates`, the outcome is adjusted by ``Z'\hat\gamma``, where the
coefficients ``\hat\gamma`` are common to both sides and come from the weighted
least-squares fit of the outcome on the covariates and the local polynomials of the
two sides (Calonico, Cattaneo, Farrell and Titiunik 2019). Collinear covariates are
dropped. In a fuzzy design, the treatment is adjusted in the same way.

The variances come from the same estimators as for local polynomial regression:
[`NNVariance`](@ref) (the default), [`HCVariance`](@ref), and for clustered data,
given as `cluster`, [`ClusterVariance`](@ref) (CR1 by default). As in rdrobust, the
residuals of the bias correction are adjusted with the leverages of the fits of
order ``p + 1`` that produce them, for HC2, HC3, CR2 and CR3; for local polynomial
regression, nprobust adjusts them with the leverages of the fits of order ``p``.

Two details depart from rdrobust, which `compat = true` reproduces. First, where a
leverage is one, the standard error is `NaN`, while rdrobust raises ``1 - h_{ii}``
to ``10^{-8}``, and the same holds for singular blocks of clusters. Second,
[`ClusterVariance`](@ref)`(3)` has the factor ``(G - 1)/G``, as in nprobust, which
rdrobust leaves out.

## Kink designs

In a regression kink design (Card, Lee, Pei and Weber 2015), the treatment is a
continuous function of the running variable whose slope changes at the cutoff, as a
benefit that is a fixed fraction of earnings up to a maximum. The regression
function is then continuous at the cutoff, and the effect of the treatment shows up
as a change in its slope: with `derivative = 1`, the estimate is the jump in the
first derivative, ``\hat\beta_{+,1} - \hat\beta_{-,1}``, from local polynomials of
order `degree = 2` by default (order ``p = \nu + 1``, as for derivatives in local
polynomial regression), and the bias correction uses fits of order 3. In a fuzzy
kink design, the treatment received is not a deterministic function of the running
variable, and the estimate is the ratio of the jumps in the slopes of the outcome
and of the treatment,

```math
\hat\tau = \frac{\hat\beta^Y_{+,1} - \hat\beta^Y_{-,1}}{\hat\beta^T_{+,1} - \hat\beta^T_{-,1}},
```

the effect of the treatment for the units at the cutoff, under the conditions of
Card, Lee, Pei and Weber (2015). The standard errors and the bandwidth selectors are
those of the jump, applied to the derivative: [`RDOptimal`](@ref) chooses bandwidths
that are optimal for the estimate of the jump in the slope, with the constants of
the derivative of order one. Kink estimates need many more observations than jumps,
since the derivative converges at a slower rate.

```julia
x = 2 .* rand(4000) .- 1
t = 0.5 .* x .- 0.4 .* max.(x, 0)                 # the slope of the treatment changes at 0
y = 1 .+ 1.5 .* t .+ 0.2 .* x .^ 2 .+ 0.3 .* randn(4000)
fit(RegressionDiscontinuity, x, y; derivative = 1)               # sharp kink: 1.5 × (-0.4)
fit(RegressionDiscontinuity, x, y; derivative = 1, treatment = t) # fuzzy kink: 1.5
```

## Bandwidths

By default the bandwidths come from [`RDOptimal`](@ref)`(:mserd)`, the bandwidth
common to both sides that minimizes the asymptotic mean squared error of the
estimate of the jump, with the pilot bandwidth ``b`` selected in the same way for
the bias correction. The other rules of rdrobust select a bandwidth for each side
or for the sum of the estimates, combine these, or shrink them to the rate that
minimizes the coverage error of the robust intervals.
`select_bandwidth(RDOptimal(), x, y; cutoff)` returns the bandwidths without the
fit, as rdrobust's `rdbwselect`.

The rules estimate the variance and the bias of the local fits with pilot fits, as
the plug-in rules of local polynomial regression do, after standardizing ``x`` and
``y``. Where 20 percent or more of the observations on a side repeat values of
``x``, `masspoints = :adjust` (the default) counts distinct values rather than
observations and keeps at least 10 distinct values in every window. A bandwidth can
also be given, as one number or a pair for the left and the right side, with
`pilot` (by default ``b = h``) or `rho` (``b = h / \rho``).

## Several cutoffs

In a design with several cutoffs (Cattaneo, Keele, Titiunik and Vazquez-Bare 2016),
different units face different cutoffs, as when a program's eligibility threshold
varies by region or year. `fit(MultipleCutoffs, x, y, c)`, with `c` the cutoff that
each unit faces, estimates the effect at each cutoff from its units, their weighted
average with weights proportional to the observations in the windows, and the
pooled effect, from the running variable normalized to ``x - c``; the pooled
estimand is an average of the effects at the cutoffs weighted by the densities of
the running variable there. With cumulative cutoffs, one running variable crosses
several thresholds at which the treatment changes, and `fit(CumulativeCutoffs, x, y;
cutoffs)` estimates the effect at each, optionally from the observations in a range
around it. Both reproduce rdmulti's `rdmc` and `rdms`.

```julia
fit(MultipleCutoffs, x, y, c)                          # cutoffs per unit
fit(CumulativeCutoffs, x, y; cutoffs = [33, 66])        # thresholds of one score
```

## Power and sample size

[`rd_power`](@ref) gives the power of the robust bias-corrected test of no effect
against an effect ``\tau`` (Cattaneo, Titiunik and Vazquez-Bare 2019),

```math
1 - \Phi\Big(z_{1-\alpha/2} - \frac{\tau}{\widehat{\mathrm{se}}}\Big)
+ \Phi\Big(-z_{1-\alpha/2} - \frac{\tau}{\widehat{\mathrm{se}}}\Big),
```

with the robust standard error of a fit, for the sample at hand or for a sample of
``n`` observations spread in the same way, whose standard error is
``\widehat{\mathrm{se}} \sqrt{N/n}`` at the same bandwidths. [`rd_samplesize`](@ref)
is the smallest ``n`` that reaches a given power. Both agree with rdpower's
`rdpower` and `rdsampsi`, which plan a study from a pilot sample or a previous one.

```julia
m = fit(RegressionDiscontinuity, x, y)
rd_power(m; tau = 0.3)                  # the power against an effect of 0.3
rd_samplesize(m; tau = 0.3, power = 0.8) # the observations needed for 80%
```

## Local randomization

Under local randomization (Cattaneo, Frandsen and Titiunik 2015), the treatment is
taken to be assigned as if at random in a small window ``[w_l, w_r]`` around the
cutoff, so that the units above and below the cutoff are comparable without a model
for the regression function. `fit(LocalRandomization, x, y; window)` tests the sharp
null hypothesis that the treatment changes every outcome in the window by ``\tau_0``
(`nulltau`, by default zero) by randomization inference: the treatment labels are
permuted within the window, with the number treated fixed, and the p-value is the
fraction of permutations whose statistic is at least as large in absolute value as
the observed one. The statistics are the difference in means, the Kolmogorov–Smirnov
statistic and the standardized rank sum, and each also has an asymptotic p-value:
from the normal distribution for the first and the last, and for the
Kolmogorov–Smirnov statistic the exact one given the ties when ``n_0 n_1 < 10000``.
The statistics and the asymptotic p-values agree with rdlocrand's `rdrandinf` to
rounding, and the randomization p-values within their Monte Carlo error. The window
is given; its choice by balance tests on covariates, as `rdwinselect`, is not
implemented.

```julia
m = fit(LocalRandomization, x, y; window = 0.2, statistic = :all, reps = 10_000)
coeftable(m)                             # the statistics and their p-values
fit(LocalRandomization, x, y; window = (-0.1, 0.3), nulltau = 0.5)
```

## Honest intervals

Robust bias correction estimates the bias and removes it, while honest intervals
(Armstrong and Kolesár 2018, 2020) bound it over a class of regression functions,
as RDHonest. Over the Hölder class of functions whose second derivative is bounded
by ``M`` on each side of the cutoff, the worst-case bias of a local linear estimate
with weights ``\ell_i`` is

```math
\bar b = \frac{M}{2} \Big| \sum_i \ell_i (x_i - c)^2 \Big| ,
```

since the weights of a local linear estimate at the boundary have the same sign on
each side, and the interval is ``\hat\tau \pm \mathrm{cv}(\bar b / \mathrm{se}) \,
\mathrm{se}``, where ``\mathrm{cv}(t)`` is the quantile of ``|N(t, 1)|``.
`class = :taylor` gives the Taylor class instead. In a fuzzy design, ``M = (M_Y,
M_T)`` bounds the outcome and the treatment, and the bias is that of
``(M_Y + M_T |\hat\tau|) / |\hat f|`` for the jump ``\hat f`` in the treatment.
`M = :rot` takes the rule of thumb of RDHonest, the largest second derivative of
global quartic fits on the two sides, which Armstrong and Kolesár offer as a
starting point for a sensitivity analysis rather than as a bound.

[`RDHonestOptimal`](@ref)`(M)` chooses the bandwidth that minimizes the worst-case
mean squared error, the length of the honest interval or the excess length of a
one-sided interval, from preliminary variances that are constant on each side, as
RDHonest. With these bandwidths and its standard errors, nearest-neighbour or
``\mathrm{HC0}`` (RDHonest's `"EHW"`), the intervals agree with RDHonest to
rounding.

```julia
m = fit(RegressionDiscontinuity, x, y; bandwidth = RDHonestOptimal(1.5))
confint(m; method = Honest(1.5))                       # the honest 95% interval
worst_case_bias(m, 1.5)                                  # its bound on the bias
m = fit(RegressionDiscontinuity, x, y; treatment = t, bandwidth = RDHonestOptimal((1.5, 0.5); criterion = :flci))
confint(m; method = Honest((1.5, 0.5)))
```

## The manipulation test

A regression discontinuity design assumes that units cannot sort themselves around
the cutoff, which would make the density of the running variable jump there.
`fit(DensityDiscontinuity, x; cutoff)` gives the test of Cattaneo, Jansson and Ma
(2020), as rddensity: local polynomial estimates of the density on the two sides
of the cutoff and the test that they are equal.

```julia
m = fit(DensityDiscontinuity, x; cutoff = 0)
m.pvalue                                 # the test of continuity of the density
coeftable(m)                             # the densities on the two sides and the jump
m.binomial                               # binomial tests in windows around the cutoff
```

On each side, the density is the slope at the cutoff of the local polynomial fit to
the empirical distribution function, as for
[`LocalPolynomialDensity`](@ref), with separate polynomials on the two sides, or
with `restricted = true` a distribution function that is continuous at the cutoff
and has common coefficients of order two and higher. The test uses the fits of
order `degree + 1` at bandwidths that minimize the mean squared error of the fits of
order `degree`, so that their bias does not distort its size, and the jackknife or
plug-in standard errors of rddensity. [`RDDensityOptimal`](@ref) chooses the
bandwidths as rddensity's `rdbwdensity`. With ties in ``x``, tied observations
share the value of the empirical distribution function. The binomial tests compare
the numbers of observations on the two sides in windows that grow from the
smallest one with 20 observations on each side up to the bandwidths. The results
agree with rddensity to rounding.

```@autodocs
Modules = [KernelSmoothing]
Pages = ["rd.jl", "rdbandwidth.jl", "rdhonest.jl", "rddensity.jl", "rdmulti.jl", "rdpower.jl", "rdlocrand.jl"]
```
