# The estimators added in #60–#65

In this note, we study the estimators that were added to the package in #60 to #65.
Their agreement with R is established by the test suite, but agreement with R does not
tell whether the intervals cover or which option to choose. For each estimator, we
ask one question, describe a design in the spirit of the paper that proposed the
method, report the results, and give a recommendation. The scripts are in
[`../simulations`](../simulations), and the results come from the run
`20261026T0100Z-ks-sim59`.

## Local likelihood (#65)

**Question.** Do the conventional and the robust bias-corrected intervals of
`LocalLikelihood` cover for the logit, Poisson, gamma and negative binomial families,
and how does the rule-of-thumb bandwidth of Fan, Heckman and Wand (1995) compare with
cross-validation?

**Design.** x is uniform on [0, 1], and the linear predictor η(x) has period one:
η = 2 sin(2πx) for binary responses (logit link), η = 1 + sin(2πx) for Poisson counts,
η = 0.5 + sin(2πx) for gamma responses with shape 2 (log link), and η = 1 + sin(2πx)
for negative binomial counts with θ = 2 (log link, θ known). We estimate η at 0.1,
0.25, 0.5, 0.75 and 0.9 with `CrossValidation()` and with `RuleOfThumb()`, for n = 500
and 2000, with 2000 replications (`simulations/locallik.jl`).

**Results.** Mean coverage of the 95% intervals for η over the five points, with the
smallest in parentheses, in percent; the ratio of the mean lengths of the robust and
the conventional intervals; the median bandwidth; and the number of points, out of
10000, at which a fit had no maximum:

| family | n | bandwidth | conventional | robust | robust/conventional length | median h | NaN |
|---|---|---|---|---|---|---|---|
| logit | 500 | cross-validation | 89.7 (83.2) | 94.4 (93.2) | 1.43 | 0.155 | 3 |
| logit | 500 | rule of thumb | 93.7 (92.2) | 94.6 (93.7) | 1.41 | 0.124 | 0 |
| logit | 2000 | cross-validation | 90.5 (86.5) | 94.3 (93.6) | 1.42 | 0.114 | 0 |
| logit | 2000 | rule of thumb | 93.3 (91.9) | 94.8 (93.8) | 1.41 | 0.095 | 0 |
| Poisson | 500 | cross-validation | 89.8 (78.4) | 93.4 (93.2) | 1.41 | 0.103 | 0 |
| Poisson | 500 | rule of thumb | 91.7 (84.2) | 93.9 (93.5) | 1.41 | 0.096 | 0 |
| Poisson | 2000 | cross-validation | 90.6 (81.2) | 94.0 (93.7) | 1.43 | 0.076 | 0 |
| Poisson | 2000 | rule of thumb | 92.0 (85.4) | 94.6 (93.8) | 1.43 | 0.073 | 0 |
| gamma | 500 | cross-validation | 88.8 (83.4) | 92.4 (91.8) | 1.43 | 0.116 | 0 |
| gamma | 500 | rule of thumb | 91.7 (88.8) | 93.0 (92.2) | 1.43 | 0.098 | 0 |
| gamma | 2000 | cross-validation | 90.8 (87.3) | 94.0 (93.2) | 1.44 | 0.086 | 0 |
| gamma | 2000 | rule of thumb | 93.0 (91.4) | 94.3 (93.4) | 1.44 | 0.075 | 0 |
| negative binomial | 500 | cross-validation | 89.8 (82.5) | 93.2 (92.2) | 1.42 | 0.13 | 0 |
| negative binomial | 500 | rule of thumb | 93.0 (89.8) | 93.5 (92.8) | 1.43 | 0.111 | 0 |
| negative binomial | 2000 | cross-validation | 90.4 (83.2) | 93.9 (92.9) | 1.44 | 0.097 | 0 |
| negative binomial | 2000 | rule of thumb | 92.4 (88.2) | 94.2 (93.4) | 1.44 | 0.085 | 0 |

The robust bias-corrected intervals cover at 92% to 95% on average in every family
and with both bandwidths, and at no point below 91.8%; they are 1.41 to 1.44 times as
long as the conventional ones. The conventional intervals undercover, most at the
peak and the trough of the linear predictor (0.25 and 0.75), where their coverage at
the worst point falls to between 78% and 92%. The
rule-of-thumb bandwidth is smaller than the cross-validated one, so that the
conventional intervals undercover less with it, and the robust intervals cover as
well or slightly better; it is also over a thousand times faster (see the
[timings](timings.md)). A fit has no maximum at only 3 of 10000 points, all for the
logit family at n = 500 with cross-validation.

**Recommendations.** Use the robust intervals, the default of `confint`. Prefer the
rule-of-thumb bandwidth, which covered at least as well as cross-validation in these
designs at a fraction of its cost.

## Shape constraints (#63)

**Question.** How much do `rearrange` and `constrain` improve a local linear fit of a
monotone function, what happens to the intervals and bands, and what does a wrong
shape cost?

**Design.** x is uniform on [0, 1] and y = m(x) + 0.3 ε for an increasing logistic
curve that is steep in the middle, m(x) = 1/(1 + exp(-(x - 0.5)/0.08)); a nearly flat
line, m(x) = 0.2x; and a sine, m(x) = 0.5 sin(2πx), which is not monotone. The fit is
`fit(LocalPolynomialRegression, x, y)` with its defaults; we rearrange it and project
it onto increasing sequences, and compare the mean squared error over its 30 points
and the coverage of the robust pointwise intervals and of the robust band (2000
draws), before and after the rearrangement, for n = 200 and 1000 with 2000
replications (`simulations/shape.jl`).

**Results.**

| design | n | MSE × 1000: fit | rearranged | constrained | pointwise: fit | rearranged | band: fit | rearranged | fit monotone |
|---|---|---|---|---|---|---|---|---|---|
| flat | 200 | 1.96 | 1.295 | 1.345 | 93.8 | 98.2 | 86.5 | 96.3 | 26.7 |
| flat | 1000 | 0.422 | 0.302 | 0.318 | 94.8 | 97.6 | 93.2 | 98.2 | 55.8 |
| logistic | 200 | 3.085 | 2.555 | 2.565 | 92.8 | 96.6 | 82.2 | 92.8 | 7.7 |
| logistic | 1000 | 0.736 | 0.647 | 0.653 | 94.4 | 97.0 | 90.8 | 96.2 | 8.0 |
| sine | 200 | 3.414 | 455.543 | 128.996 | 92.6 | 5.8 | 80.3 | 0.0 | 0.0 |
| sine | 1000 | 0.832 | 461.834 | 127.568 | 94.4 | 1.7 | 92.8 | 0.0 | 0.0 |

The last column is the fraction of fits that are already monotone. For the increasing
functions, both operations reduce the error, by 11% to 17% for the logistic curve and
by 25% to 34% for the flat line, whose fits are monotone only a quarter to a little
over half of the time; the projection is about as good as the rearrangement. The rearranged band
covers more often than the original one, 93% to 98% against 82% to 93%, as
Proposition 2 of Chernozhukov, Fernández-Val and Galichon (2009) allows; its mean
width is the same by construction. The rearranged pointwise intervals overcover
(97% to 98%). For the sine, imposing a monotone shape is a disaster: the error grows
by a factor of 40 to 550, and the intervals and bands no longer cover.

**Recommendations.** When the function is known to be monotone, rearrange the fit and
its band; the band can only gain. Impose a shape only when it is known, since a wrong
shape costs far more than it can gain.

## Densities on a bounded support (#61)

**Question.** Which boundary correction should one use on [0, 1], and how does each
compare with `LocalPolynomialDensity`, which needs no knowledge of the support?

**Design.** Samples of Beta(1, 3), whose density is 3 at 0 with a steep slope, of
Beta(2, 2), whose density vanishes at both ends, and of the uniform distribution, with
n = 200 and 1000 and 1000 replications. We compare the kernel density estimate that
ignores the boundary (`KDE`), reflection, the beta kernels of Chen (1999) and their
modified version, the logit transformation, and `LocalPolynomialDensity`, each with
its default bandwidth. The integrated squared error is over [0, 1]
(`simulations/bounded.jl`).

**Results.**

| design | n | method | ISE × 1000 | RMS error at 0.001 | at 0.05 | at 0.5 | seconds |
|---|---|---|---|---|---|---|---|
| beta13 | 200 | kde | 78.113 | 1.578 | 0.482 | 0.152 | 0.0013 |
| beta13 | 200 | reflection | 16.24 | 0.455 | 0.269 | 0.102 | 0.0062 |
| beta13 | 200 | beta | 31.983 | 0.801 | 0.359 | 0.116 | 0.0784 |
| beta13 | 200 | beta-modified | 31.477 | 0.848 | 0.386 | 0.105 | 0.0655 |
| beta13 | 200 | logit | 54.835 | 3.27 | 0.416 | 0.084 | 0.0039 |
| beta13 | 200 | lpdensity | 42.194 | 0.933 | 0.329 | 0.111 | 0.0052 (NaN in 52) |
| beta13 | 1000 | kde | 39.026 | 1.524 | 0.171 | 0.09 | 0.002 |
| beta13 | 1000 | reflection | 5.478 | 0.293 | 0.131 | 0.058 | 0.0071 |
| beta13 | 1000 | beta | 7.966 | 0.374 | 0.187 | 0.059 | 0.6182 |
| beta13 | 1000 | beta-modified | 7.636 | 0.4 | 0.192 | 0.051 | 0.5866 |
| beta13 | 1000 | logit | 14.119 | 1.636 | 0.219 | 0.043 | 0.005 |
| beta13 | 1000 | lpdensity | 7.667 | 0.427 | 0.147 | 0.055 | 0.0078 (NaN in 51) |
| beta22 | 200 | kde | 15.295 | 0.182 | 0.081 | 0.133 | 0.0011 |
| beta22 | 200 | reflection | 22.507 | 0.429 | 0.199 | 0.118 | 0.0062 |
| beta22 | 200 | beta | 25.744 | 0.198 | 0.165 | 0.171 | 0.0805 |
| beta22 | 200 | beta-modified | 20.931 | 0.237 | 0.126 | 0.145 | 0.0679 |
| beta22 | 200 | logit | 23.225 | 0.131 | 0.169 | 0.155 | 0.0039 |
| beta22 | 200 | lpdensity | 32.726 | 0.434 | 0.266 | 0.134 | 0.0053 (NaN in 2) |
| beta22 | 1000 | kde | 4.512 | 0.134 | 0.041 | 0.075 | 0.0019 |
| beta22 | 1000 | reflection | 7.051 | 0.292 | 0.085 | 0.07 | 0.0072 |
| beta22 | 1000 | beta | 6.882 | 0.097 | 0.088 | 0.091 | 0.6281 |
| beta22 | 1000 | beta-modified | 4.832 | 0.131 | 0.051 | 0.072 | 0.603 |
| beta22 | 1000 | logit | 6.323 | 0.094 | 0.086 | 0.086 | 0.0049 |
| beta22 | 1000 | lpdensity | 5.95 | 0.171 | 0.079 | 0.078 | 0.008 (NaN in 1) |
| uniform | 200 | kde | 33.657 | 0.502 | 0.296 | 0.118 | 0.0011 |
| uniform | 200 | reflection | 7.2 | 0.12 | 0.116 | 0.068 | 0.0063 |
| uniform | 200 | beta | 10.464 | 0.271 | 0.135 | 0.073 | 0.0577 |
| uniform | 200 | beta-modified | 10.878 | 0.22 | 0.135 | 0.073 | 0.0458 |
| uniform | 200 | logit | 37.375 | 1.657 | 0.259 | 0.111 | 0.0039 |
| uniform | 200 | lpdensity | 24.877 | 0.484 | 0.225 | 0.08 | 0.0054 |
| uniform | 1000 | kde | 16.723 | 0.498 | 0.169 | 0.075 | 0.0012 |
| uniform | 1000 | reflection | 2.582 | 0.073 | 0.068 | 0.047 | 0.0073 |
| uniform | 1000 | beta | 2.083 | 0.106 | 0.062 | 0.037 | 0.4128 |
| uniform | 1000 | beta-modified | 2.148 | 0.098 | 0.066 | 0.037 | 0.4036 |
| uniform | 1000 | logit | 9.519 | 0.845 | 0.129 | 0.06 | 0.0052 |
| uniform | 1000 | lpdensity | 4.825 | 0.208 | 0.101 | 0.041 | 0.0095 |

Where the density is positive at the boundary (Beta(1, 3) and the uniform), every
correction except the logit transformation beats the plain kernel estimate, whose
error at the boundary is half the density; the logit transformation loses for the
uniform at n = 200, and at the boundary point 0.001 in all four cases. Reflection has the smallest integrated error in three of these four cases,
and the beta kernels in the fourth; `LocalPolynomialDensity` is close to the beta
kernels for Beta(1, 3) at n = 1000.
The logit transformation is poor at the boundary, since its estimate tends to zero or
infinity there. Where the density vanishes at the boundary (Beta(2, 2)), no correction
is needed, and the plain estimate is the most accurate; reflection, which forces a
zero slope at the boundary, and `LocalPolynomialDensity` are then the least accurate
at the boundary point 0.001. Note that
`LocalPolynomialDensity` returns `NaN` at points more than half a bandwidth beyond the
largest observation, which for Beta(1, 3) happens in about 5% of the samples (52 and
51 of 1000), so that its integrated error there is computed over the other samples. The beta kernels choose their parameter by
cross-validation and are slower, 0.4 to 0.6 seconds at n = 1000 against under 0.01
seconds for the other estimates.

**Recommendations.** Use reflection when the density is believed to be positive at the
boundary, and the plain estimate when it vanishes there. Use `LocalPolynomialDensity`
when inference at the boundary is needed or the support is not known exactly.

## Local randomization (#64)

**Question.** Do the randomization tests of `LocalRandomization` have the right size,
how much power do they have, and how sensitive are they to a regression function that
is not flat within the window?

**Design.** x is uniform on [-1, 1] and y = 0.5 + τ 1{x ≥ 0} + β x + ε with standard
normal errors, in the window (-0.1, 0.1), which holds about 25 observations on each
side at n = 500 and 100 at n = 2000. The null hypothesis is τ = 0 and the alternative
τ = 0.5, with a flat regression function (β = 0), for which local randomization
holds, and with a slope (β = 2), for which it holds only approximately. Each test uses
1000 permutations, and there are 2000 replications (`simulations/rdlocrand.jl`).

**Results.** Rejection rates at the 5% level:

| design | n | statistic | randomization | asymptotic |
|---|---|---|---|---|
| alt-flat | 500 | diffmeans | 40.4 (1.1) | 42.1 (1.1) |
| alt-flat | 500 | ksmirnov | 31.2 (1.0) | 30.8 (1.0) |
| alt-flat | 500 | ranksum | 39.0 (1.1) | 39.0 (1.1) |
| alt-flat | 2000 | diffmeans | 93.9 (0.5) | 94.2 (0.5) |
| alt-flat | 2000 | ksmirnov | 86.8 (0.8) | 86.4 (0.8) |
| alt-flat | 2000 | ranksum | 93.3 (0.6) | 93.4 (0.6) |
| alt-slope | 500 | diffmeans | 67.0 (1.1) | 69.0 (1.0) |
| alt-slope | 500 | ksmirnov | 55.2 (1.1) | 54.8 (1.1) |
| alt-slope | 500 | ranksum | 65.0 (1.1) | 65.2 (1.1) |
| alt-slope | 2000 | diffmeans | 99.9 (0.1) | 99.9 (0.1) |
| alt-slope | 2000 | ksmirnov | 98.6 (0.3) | 98.5 (0.3) |
| alt-slope | 2000 | ranksum | 99.8 (0.1) | 99.8 (0.1) |
| null-flat | 500 | diffmeans | 5.4 (0.5) | 5.9 (0.5) |
| null-flat | 500 | ksmirnov | 5.1 (0.5) | 4.8 (0.5) |
| null-flat | 500 | ranksum | 5.1 (0.5) | 5.1 (0.5) |
| null-flat | 2000 | diffmeans | 5.1 (0.5) | 5.4 (0.5) |
| null-flat | 2000 | ksmirnov | 5.1 (0.5) | 4.5 (0.5) |
| null-flat | 2000 | ranksum | 5.2 (0.5) | 5.1 (0.5) |
| null-slope | 500 | diffmeans | 11.2 (0.7) | 12.0 (0.7) |
| null-slope | 500 | ksmirnov | 9.5 (0.7) | 9.4 (0.7) |
| null-slope | 500 | ranksum | 10.7 (0.7) | 11.0 (0.7) |
| null-slope | 2000 | diffmeans | 31.0 (1.0) | 30.8 (1.0) |
| null-slope | 2000 | ksmirnov | 24.2 (1.0) | 22.8 (0.9) |
| null-slope | 2000 | ranksum | 29.0 (1.0) | 29.0 (1.0) |

With a flat regression function, the three tests have the right size, and the
randomization and the asymptotic p-values agree; the difference in means and the rank
sum are the most powerful against a shift, and the Kolmogorov–Smirnov test the least.
However, a slope within the window breaks the tests: the treated and the control
outcomes differ by about 2 × 0.1 even without an effect, and the size grows with the
sample, from 9% to 12% at n = 500 to 23% to 31% at n = 2000. The power under a slope is inflated for the same reason.

**Recommendations.** Choose the window so that the regression function is close to
flat within it, and check the balance of covariates there; the window selection of
rdlocrand's `rdwinselect` and its polynomial adjustment are not implemented yet (#95). With
a window that is not small, prefer `RegressionDiscontinuity`.

## Local quantile regression (#62, #94)

**Question.** Do the conventional and robust bias-corrected (RBC) pointwise intervals
of `LocalQuantileRegression` cover with the default bandwidth of Yu and Jones (1998),
and what does the bias correction cost in interval length?

**Design.** The sine designs have x uniform on [0, 1] and
y = sin(2πx) + σ(x) ε, with heteroskedastic normal errors, σ(x) = 0.5 + x
(`sine-het`), or with σ = 0.5 and t₃ errors (`sine-t3`). We estimate the
quartiles and the median at 0.1, 0.25, 0.5, 0.75 and 0.9.
The `qy2` design has x uniform on [-1, 1], normal errors, and
Q(τ | x) = 0.5 + x + x² + sin(πx − 1) + (x + 1.25) Φ⁻¹(τ).
It adapts Qu and Yoon (2019, Section 8, equation (21), Model 2) and is used here
for pointwise checks only. We estimate τ ∈ {0.2, 0.5, 0.8} at −0.8:0.4:0.8.
Each design uses the default local linear fit and bandwidth, for n = 500 and 2000,
with 2000 replications per design and sample size (`simulations/quantreg.jl`).

**Results.** Coverage of the 95% pointwise intervals, in percent. Each point cell
gives conventional / RBC coverage. Monte Carlo standard errors range from 0.4 to
1.0 percentage points. Coverage, calibration and length comparisons use the common
valid replications at each point; median h uses all attempted replications.

| design | τ | n | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | median h |
|---|---|---|---|---|---|---|---|---|
| heteroskedastic normal | 0.25 | 500 | 92.4 / 95.4 | 82.8 / 96.4 | 95.1 / 95.2 | 86.6 / 94.0 | 93.6 / 95.2 | 0.153 |
| heteroskedastic normal | 0.25 | 2000 | 89.8 / 94.8 | 82.7 / 95.4 | 94.8 / 95.6 | 89.6 / 94.6 | 94.0 / 95.2 | 0.114 |
| heteroskedastic normal | 0.5 | 500 | 92.4 / 95.3 | 82.2 / 95.9 | 95.4 / 96.4 | 89.0 / 95.0 | 94.4 / 95.4 | 0.148 |
| heteroskedastic normal | 0.5 | 2000 | 88.8 / 94.8 | 81.8 / 95.6 | 95.6 / 95.6 | 88.9 / 95.0 | 93.2 / 94.8 | 0.111 |
| heteroskedastic normal | 0.75 | 500 | 91.4 / 94.0 | 79.6 / 95.1 | 94.8 / 95.0 | 89.4 / 94.6 | 93.8 / 95.0 | 0.155 |
| heteroskedastic normal | 0.75 | 2000 | 88.9 / 95.1 | 80.2 / 94.8 | 94.9 / 95.0 | 89.8 / 94.8 | 95.0 / 94.8 | 0.116 |
| t₃ | 0.25 | 500 | 95.0 / 95.7 | 84.4 / 96.9 | 95.5 / 95.9 | 82.5 / 95.2 | 93.0 / 95.5 | 0.139 |
| t₃ | 0.25 | 2000 | 93.4 / 96.5 | 82.8 / 95.5 | 96.8 / 95.8 | 84.6 / 96.1 | 90.8 / 95.6 | 0.105 |
| t₃ | 0.5 | 500 | 92.2 / 96.0 | 80.0 / 96.0 | 96.2 / 96.1 | 78.2 / 96.0 | 94.2 / 96.8 | 0.135 |
| t₃ | 0.5 | 2000 | 91.0 / 95.6 | 80.9 / 95.4 | 96.2 / 96.4 | 78.8 / 96.1 | 89.4 / 95.6 | 0.102 |
| t₃ | 0.75 | 500 | 93.4 / 96.3 | 83.7 / 96.0 | 95.9 / 96.2 | 82.8 / 96.8 | 94.4 / 95.8 | 0.141 |
| t₃ | 0.75 | 2000 | 92.1 / 94.8 | 84.2 / 95.6 | 96.0 / 96.2 | 83.0 / 96.3 | 93.2 / 95.4 | 0.105 |

For the Qu–Yoon adaptation:

| design | τ | n | -0.8 | -0.4 | 0 | 0.4 | 0.8 | median h |
|---|---|---|---|---|---|---|---|---|
| qy2 | 0.2 | 500 | 95.8 / 94.5 | 79.6 / 94.6 | 83.7 / 94.9 | 94.0 / 94.7 | 93.6 / 94.8 | 0.347 |
| qy2 | 0.2 | 2000 | 94.8 / 95.4 | 76.9 / 94.5 | 85.4 / 94.6 | 95.1 / 95.1 | 94.4 / 95.2 | 0.260 |
| qy2 | 0.5 | 500 | 96.0 / 95.2 | 76.8 / 96.7 | 85.4 / 96.2 | 95.1 / 94.5 | 95.3 / 95.6 | 0.334 |
| qy2 | 0.5 | 2000 | 94.0 / 95.3 | 77.2 / 95.3 | 86.2 / 95.6 | 94.2 / 95.2 | 93.1 / 95.0 | 0.249 |
| qy2 | 0.8 | 500 | 95.9 / 95.4 | 75.3 / 96.0 | 84.9 / 95.7 | 94.8 / 94.8 | 92.5 / 93.6 | 0.346 |
| qy2 | 0.8 | 2000 | 94.2 / 94.6 | 77.0 / 96.6 | 85.4 / 95.6 | 94.6 / 95.2 | 93.8 / 95.2 | 0.261 |

The pre-registered criterion required RBC coverage within [92.5%, 97.5%] at every
point of every design at n = 2000. It held: RBC coverage ranged from 94.5% to 96.6%.
At n = 500, which was not gated, RBC coverage ranged from 93.6% to 96.9%.

Bias correction reduces the bias at the sine peaks and improves coverage there.
For example, with t₃ errors, τ = 0.5 and n = 2000, bias/SD changes from −0.902 to
0.018 at x = 0.25, and from 0.934 to 0.024 at x = 0.75. The robust standard errors
are well calibrated in these designs: RMS `se_rbc` / SD of the corrected estimate
ranges from 0.957 to 1.133. The intervals are longer: the ratio of mean lengths,
RBC / conventional, ranges from 1.34 to 1.45, with a median of 1.44.
Both methods have non-finite results at the same 4 replication-points out of 180000.

These results come from commit `f31b591`. To reproduce them, run
`julia --project=simulations simulations/quantreg.jl DESIGN N FIRSTREP NREPS OUT`
for the designs and sample sizes above. `DESIGN` is the design code followed by
`-τ`; follow the argument convention in the
[simulation instructions](../simulations/README.md). Combine the outputs into
`results-quantreg.csv` and run
`julia --project=simulations simulations/summarize.jl quantreg`.

**Recommendations.** Use the default `confint` (RBC) where the quantile function may
curve. The conventional intervals, available with `method = Conventional()`, are
intervals for the smoothed quantile; inference for the quantile itself requires
negligible smoothing bias. Validity of the correction is claimed only for odd
`degree - derivative`, including the default `degree = derivative + 1`, under the
pointwise inference assumptions in the [documentation](../docs/src/conditional.md).
