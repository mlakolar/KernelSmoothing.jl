# [Bandwidth selection](@id bandwidths-page)

Estimators accept a number or a [`BandwidthSelector`](@ref) as bandwidth.
[`selectbandwidth`](@ref) applies a rule and returns a [`BandwidthSelection`](@ref)
that records what the rule computed, such as pilot bandwidths or a
cross-validation curve, and every fit stores the selection that chose its
bandwidth.

For densities, [`SheatherJones`](@ref) is the default; [`LSCV`](@ref),
[`NormalReference`](@ref) and [`SilvermanRule`](@ref) are alternatives. For
regression, the default follows nprobust: [`MSEOptimal`](@ref) at a single point
and [`IMSEOptimal`](@ref) at several, the plug-in bandwidths of Calonico, Cattaneo
and Farrell (2019), with pilot bandwidths anchored on cross-validation.
[`CoverageOptimal`](@ref) rescales them for robust bias-corrected inference,
[`FLCIOptimal`](@ref) chooses bandwidths for honest intervals, and
[`RuleOfThumb`](@ref) and [`CrossValidation`](@ref) are alternatives.

```julia
d = SmoothingData(randn(1000))
s = selectbandwidth(LSCV(), d, EpanechnikovKernel())
s.h, s.details.minimizers        # the bandwidth, and all local minimizers of the criterion
```

## Rules of thumb

[`NormalReference`](@ref) gives the bandwidth that minimizes the asymptotic mean
integrated squared error of a kernel density estimate when the density is normal,

```math
h = \hat\sigma \left( \frac{8 \sqrt{\pi}\, R(K)}{3 \mu_2(K)^2\, n} \right)^{1/5},
```

with ``R(K) = \int K^2``, ``\mu_2(K) = \int u^2 K(u)\,du`` and ``\hat\sigma`` the
smaller of the standard deviation and the interquartile range divided by 1.349.
This is ``1.06\, \hat\sigma n^{-1/5}`` for the Gaussian kernel and
``2.34\, \hat\sigma n^{-1/5}`` for the Epanechnikov kernel. [`SilvermanRule`](@ref)
is R's `bw.nrd0`. Weighted samples use Kish's effective sample size for ``n``.

For regression, [`RuleOfThumb`](@ref) is the rule of Fan and Gijbels (1996,
Section 4.2),

```math
h = C_{\nu,p}(K) \left( \frac{\tilde\sigma^2 (\max_i x_i - \min_i x_i)}{n D} \right)^{1/(2p+3)},
```

where the residual variance ``\tilde\sigma^2`` and the mean squared derivative
``D`` of order ``p + 1`` come from a global polynomial fit of order ``p + 3`` in the
standardized covariate, and ``C_{\nu,p}(K)`` depends on the equivalent kernel.

## Sheather and Jones

The rule of Sheather and Jones (1991) estimates ``\psi_4 = \int f''(x)^2\,dx``
with a Gaussian pilot kernel and solves

```math
h = \left( \frac{R(K)}{\mu_2(K)^2\, \hat\psi_4(\alpha_2 h^{5/7})\, n} \right)^{1/5},
```

where ``\alpha_2`` comes from estimates of ``\psi_4`` and ``\psi_6`` at
normal-reference pilot bandwidths. The estimates are the double sums

```math
\hat\psi_r(g) = \frac{1}{W^2 - \sum_i w_i^2} \sum_{i=1}^n \sum_{j=1}^n w_i w_j\, \phi_g^{(r)}(x_i - x_j),
```

with the diagonal included, which with equal weights is the definition of R's
`bw.SJ`. They are computed exactly for up to 100 observations. Larger samples are
binned linearly once, on a grid with about 100 points per pilot bandwidth, and the
lagged products of the bin weights come from an FFT, which changes the bandwidth by
about ``10^{-6}`` relative. Far outliers and heavy tails would need too fine a grid
over the whole range: the data are then split at gaps across which the kernels are
negligible, only the dense part of each run is binned, and the pairs with the
remaining observations, which are sparse, are summed exactly. The equation is solved
to a relative accuracy of ``10^{-10}``. For other kernels than the Gaussian, the bandwidth is the Gaussian
one times the ratio ``(R(K)/\mu_2(K)^2)^{1/5}`` of canonical bandwidths. With
`method = :dpi`, the pilot bandwidth is the direct plug-in instead of
``\alpha_2 h^{5/7}``. The rule is the recommended default for densities without
boundaries (Jones, Marron and Sheather 1996).

## Cross-validation

Least-squares cross-validation, [`LSCV`](@ref), minimizes an unbiased estimate of
the integrated squared error minus ``\int f^2``,

```math
\mathrm{CV}(h) = \int \hat f(x)^2\,dx - \frac{2}{W^2 - \sum_i w_i^2} \sum_{i \neq j} w_i w_j\, \frac{1}{h} K\Big(\frac{x_i - x_j}{h}\Big),
```

on a grid of bandwidths spaced evenly on the log scale, whose lower end comes from a
robust scale estimate, so that outliers do not lift it. The criterion often has
several local minimizers. Following Sheather (2004), the rule prefers the largest,
since spurious ones appear at small bandwidths, as for rounded data, where the
criterion falls without bound as ``h`` shrinks. With compact kernels, however,
clustered data give local minimizers at the scale of the gaps between the clusters,
where the criterion is far above its lowest local minimum. The rule therefore takes
the largest local minimizer whose criterion is within half of the lowest one, and
refines it. Where the criterion is lowest at the lower end of the grid and the data
have no ties, the grid extends downward first. For regression,
[`CrossValidation`](@ref) minimizes

```math
\mathrm{CV}(h) = \frac{\sum_i w_i \big(y_i - \hat m_{-i}(x_i)\big)^2}{\sum_i w_i},
\qquad \hat m_{-i}(x_i) = \frac{\hat m(x_i) - \ell_{ii} y_i}{1 - \ell_{ii}},
```

where ``\ell_{ii}`` is the weight of ``y_i`` in the fit at ``x_i``. The fits at all
observations come from [`kernelsums`](@ref), so that a bandwidth costs ``O(n)``
operations for polynomial kernels; for Gaussian kernels and more than 1000
observations, the sums come from binned data where that is faster. The grid starts
where the leave-one-out fits exist, at the smallest bandwidth that leaves ``p + 1``
other distinct values in the window of each observation. Isolated observations,
whose distance to their ``(p+1)``-th nearest other distinct value is more than three
interquartile ranges above the upper quartile of these distances, would set this
floor for all, so the criterion leaves them out and the sums run over the others.
The rule takes the global minimizer over the grid, refines it between its
neighbours, and warns when it lies at the lower end of the grid. Cross-validation assumes independent
errors; with positively correlated errors it undersmooths.

## Plug-in bandwidths for regression

At a point, the mean squared error of the estimate of ``m^{(\nu)}`` by a fit of
order ``p`` is approximately ``h^{2(p+1-\nu)} B^2 + V/(n h^{1+2\nu})``. Here
``B = C_B \hat\beta_{p+1}`` is the bias, with ``\hat\beta_{p+1}`` an estimate of the
next Taylor coefficient, and ``V`` the variance, both with the finite-sample
constants of the local fits. [`MSEOptimal`](@ref) minimizes it at each point,

```math
h = \left( \frac{(1 + 2\nu)\, V}{2 (p + 1 - \nu)\, n\, (B^2 + \lambda R)} \right)^{1/(2p+3)},
```

where ``R = 3 C_B^2 \hat V[\hat\beta_{p+1}]`` regularizes the bandwidth where the
derivative nearly vanishes, scaled by `regularization` ``= \lambda``. When ``p - \nu``
is even, the bias includes its next term and the error is minimized numerically,
with the squared bias ``(|B_1| + h |B_2|)^2 + \lambda R``; nprobust's
``(B_1 + h B_2 + \lambda R)^2``, which `compat = true` keeps, vanishes where the
estimated terms cancel, and its bandwidth depends on the units of ``y``.
The derivative in ``B`` is estimated by a fit of order ``p + 1`` with a pilot
bandwidth ``b``. By default, ``b`` is the cross-validation bandwidth of the fit of
order ``p`` times the ratio of the rule-of-thumb bandwidths for the derivative of
order ``p + 1`` and for the regression function. Since cross-validation adapts to
the roughness of the regression function, so does the pilot. With `compat = true`, the pilots come from the stages
of nprobust's `lpbwselect`, whose bias constants are estimated by fits over the
whole range of the data. At interior points these stages end at their cap, the
distance to the farther end of the data, so that the derivative is estimated over
nearly the whole range and the bandwidths are too large where the curvature is
local.

[`IMSEOptimal`](@ref) averages the variance and squared-bias constants over a grid,
30 quantiles of the covariate of order 0.05 to 0.95, or the evaluation points with
`grid = :points`, and gives one bandwidth, at most the range of the data. nprobust,
whose rule `compat = true` reproduces, averages over 30 equally spaced points
between the smallest and the largest observation, which for a covariate with heavy
tails lie mostly where few observations are left, and its squared bias leaves out
the regularization.
[`CoverageOptimal`](@ref) rescales the plug-in bandwidths by a power of the sample
size, which makes the coverage error of robust bias-corrected intervals shrink at
the fastest rate (Calonico, Cattaneo and Farrell 2018, 2022). For odd ``p - \nu``
this rate differs between interior and boundary points. By default the rule takes
the interior rate, which leaves local linear bandwidths unchanged, and
`compat = true` the boundary rate at every point, as nprobust. The same rules give
the bandwidths of the density estimators; see
[Density estimation](@ref density-page).

```@autodocs
Modules = [KernelSmoothing]
Pages = ["bandwidths.jl", "selectors.jl", "plugin.jl"]
```
