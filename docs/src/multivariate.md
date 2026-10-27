# [Multivariate density estimation](@id multivariate-page)

## Kernel density estimates with a bandwidth matrix

For observations ``x_1, \dots, x_n \in \mathbb{R}^d`` with weights ``w_i`` and a
positive definite bandwidth matrix ``H``, the kernel density estimate with the
normal kernel is

```math
\hat f(x) = \frac{1}{W} \sum_{i=1}^n w_i\, \phi_H(x - x_i), \qquad
\phi_H(u) = (2\pi)^{-d/2} |H|^{-1/2} \exp\big(-\tfrac12 u^\top H^{-1} u\big),
```

where ``W = \sum_i w_i``. The matrix ``H`` is the covariance matrix of the kernel:
it sets both the amount of smoothing and its orientation, so that the estimate can
smooth along the directions in which the data are spread. `fit(MultivariateKDE, x)`
returns the estimate as a continuous multivariate distribution of Distributions.jl,
with `pdf`, `logpdf`, `rand`, `mean`, `cov`, `var`, `length` and `insupport`. The
sample `x` is a ``d \times n`` matrix whose columns are the observations, as for
`fit(MvNormal, x)`, or a tuple of vectors, one for each coordinate; note that this
is the transpose of the matrices of points elsewhere in this package. The density is
computed exactly from the observations; `logpdf` sums about its largest term, so
that it stays finite far from the data, where the density underflows. A matrix of
points is evaluated column by column in parallel, and `pdf(f, (g1, g2))` evaluates
the estimate on the grid of two vectors. A draw is an observation chosen with
probability proportional to its weight plus a draw from ``N(0, H)``, and
`marginal(f, j)` is the estimate of the density of coordinate ``j``, a
one-dimensional [`KDE`](@ref) with bandwidth ``\sqrt{H_{jj}}``.

The mean integrated squared error of the estimate is asymptotically

```math
\mathrm{AMISE}(H) = \frac{(4\pi)^{-d/2}}{n |H|^{1/2}}
+ \frac14 \sum_{a,b,c,e=1}^d H_{ab} H_{ce}\, \psi_{abce},
\qquad \psi_{abce} = \int \frac{\partial^4 f(x)}{\partial x_a \partial x_b \partial x_c \partial x_e}\, f(x)\, dx,
```

with ``n`` the effective number of observations ``W^2 / \sum_i w_i^2``. The
bandwidth matrix selectors estimate this error, or the exact integrated squared
error, and minimize the estimate over positive definite matrices, parametrized by
their Cholesky factors, with a quasi-Newton method and analytic gradients. The
plug-in and cross-validation selectors work on the data transformed to unit
sample covariance (``y_i = S^{-1/2} x_i``, the symmetric square root) or to unit
variances, and transform the selected matrix back.

## Selectors

**Normal scale.** If ``f`` is normal with covariance ``\Sigma``, the AMISE is
minimized by ``H = (4/((d+2)n))^{2/(d+4)} \Sigma``; [`NormalScaleMatrix`](@ref)
substitutes the sample covariance matrix. The diagonal version minimizes the same
normal-reference AMISE over diagonal matrices.

**Plug-in.** [`PlugInMatrix`](@ref) (Wand and Jones 1994; Duong and Hazelton 2003)
minimizes the AMISE with the functionals replaced by the kernel estimates

```math
\hat\psi_r(g) = \frac{1}{W^2} \sum_{i=1}^n \sum_{j=1}^n w_i w_j\, \partial^r \phi_{g^2 I}(y_i - y_j),
```

which include the pairs ``i = j``. The estimate of order ``|r| = j`` has the
asymptotic bias ``A_r g^{-d-j}/n + B_r g^2/2``, with ``A_r = \partial^r\phi_I(0)`` and
``B_r = \sum_a \psi_{r + 2e_a}``, and the pilot ``g`` minimizes the sum of the squared
biases over the multi-indices ``r`` whose entries are all even:

```math
g = \Big(\frac{t}{n}\Big)^{1/(d+j+2)}, \qquad
t = \frac{(d + j - 2)B + \sqrt{(d + j - 2)^2 B^2 + 8 (d + j) A C}}{2C},
```

with ``A = \sum_r A_r^2``, ``B = \sum_r A_r B_r`` and ``C = \sum_r B_r^2``. The two-stage
rule estimates the functionals of order 6 with a pilot computed from the normal
reference values of order 8, and uses them for the pilot of order 4.

**Least-squares cross-validation.** [`LSCVMatrix`](@ref) minimizes

```math
\mathrm{LSCV}(H) = \frac{1}{W^2} \sum_{i,j} w_i w_j\, \phi_{2H}(x_i - x_j)
- \frac{2}{W} \sum_i \frac{w_i}{W - w_i} \sum_{j \neq i} w_j\, \phi_H(x_i - x_j),
```

an unbiased estimate of the integrated squared error minus ``\int f^2``. Tied or
rounded data can give spuriously small bandwidths. The rule warns when any
observation duplicates an earlier one or a coordinate's tie fraction,
``1 - \text{distinct}/n``, exceeds 10%. The details record `duplicates` and
the per-coordinate `ties` in the original coordinates after zero-weight
observations are dropped. These counts are unweighted and treat `-0.0` as `0.0`.
Consider [`SCVMatrix`](@ref) or [`PlugInMatrix`](@ref) for such data.

The criterion can have no proper minimum when observations coincide or share
binning cells. A degenerate selection emits a separate warning and reports
`details.collapsed = true` and `details.converged = false`. The check uses both
extreme eigenvalues relative to the starting normal-scale matrix, recorded in
`details.eigenratio`, with limits ``10^{-8}`` and ``10^8``. It also checks that
the selected and returned matrices are finite and the returned matrix is positive
definite. The matrix and criterion are returned unchanged. Use [`SCVMatrix`](@ref),
[`PlugInMatrix`](@ref) or `binned = false` if a collapse is reported.

**Smoothed cross-validation.** [`SCVMatrix`](@ref) (Hall, Marron and Park 1992;
Duong and Hazelton 2005) minimizes

```math
\mathrm{SCV}(H) = \frac{(4\pi)^{-d/2}}{n |H|^{1/2}}
+ \frac{1}{W^2} \sum_{i,j} w_i w_j \big(\phi_{2H+2G} - 2\phi_{H+2G} + \phi_{2G}\big)(y_i - y_j),
```

with the pilot ``G = g^2 I`` of Duong (2004, Theorem 4), which minimizes the
squared asymptotic bias of the selected matrix and depends on the plug-in matrix
and on an estimate of ``\Theta_6 = \int (D^2)^3 f(x)\, f(x)\, dx``.

## Binned bandwidth selection

Only the plug-in and smoothed cross-validation rules approximate their pair sums
by linear binning by default, in two dimensions above 500 observations with
positive weight. This threshold counts observations, rather than the effective
sample size. All rules use exact sums by default in three dimensions because
binning at practical grid sizes misses the 2% selection-error target.
Least-squares cross-validation uses exact sums by default in every dimension.
`binned = true` forces binning in two or three dimensions, and `gridsize` chooses
the number of nodes per axis, at least 3. It defaults to 151 in two dimensions
and 51 in three:

```julia
select_bandwidth(PlugInMatrix(binned = true, gridsize = 201), x)
select_bandwidth(SCVMatrix(binned = false), x)   # exact pair sums
```

A binned LSCV criterion can develop a spurious minimum as ``H \to 0`` when
observations share grid cells; a hybrid fix is tracked in issue #101.
Use `binned = false` in this case.
`LSCVMatrix(binned = true)` remains available explicitly.

Each selection builds one grid spanning the transformed observations and reuses
it through the pilot chain and criterion minimization. Multilinear weights put
each observation on its surrounding grid nodes. A zero-padded FFT computes the
lag correlations; all lags enter the criterion, and the analytic gradients are
the exact gradients of this binned criterion. Binning and the FFT are serial;
the lag sums run in parallel, with a fixed summation order independent of the
number of threads.

Binning spreads an observation's self-pairs over the neighbouring lags. We remove
this contribution separately for each observation, so that the binned sums
approximate only pairs ``i \ne j``. The diagonal terms stay exact. Distinct
observations at the same location still contribute to the pair sums. The total
weight, squared weights, effective sample size and covariance come from the
original data. Unequal weights in LSCV use cross-correlations with the
leave-one-out weights.

The selection's `details` record `binned`, `gridsize`, `spacing` and `resolution`.
The spacing is in the transformed coordinates. `resolution.selected` is the
smallest kernel standard deviation of the selected matrix in these coordinates,
divided by the largest grid spacing. `resolution.pilot` is the same ratio for the
pilot standard deviation, the smallest pilot for PI, or `nothing` for LSCV.
A ratio below 1 warns that the grid is coarse relative to the bandwidth or pilot.
Increase `gridsize` or set `binned = false`; outliers that stretch the grid can
make a finer grid necessary.

`binned = false` restores exact sums, which cost a time of order ``n^2`` and also
run in parallel. In exact mode the grid is ignored and its diagnostics are
`nothing`. Binning is available only in two and three dimensions, for both full
and diagonal matrices. `compat` controls the plug-in pilot ordering independently
of binning. The density evaluation by `pdf` always uses the original observations
exactly.

## Example

```julia
using KernelSmoothing, Plots

x = [randn(500)'; randn(500)']
x[2, :] .+= 0.8 .* x[1, :] .^ 2          # a curved density
f = fit(MultivariateKDE, x)               # the plug-in bandwidth matrix
bandwidth(f)
pdf(f, [0.0, 0.5]), logpdf(f, x[:, 1:3])
pdf(f, (range(-3, 3; length = 61), range(-1, 6; length = 71)))   # on a grid
fit(MultivariateKDE, x; bandwidth = LSCVMatrix())
plot(f)                                   # contours, with Plots loaded
```

## Reference

```@docs
MultivariateKDE
fit(::Type{MultivariateKDE}, ::Union{AbstractMatrix,Tuple{Vararg{AbstractVector}}})
pdf(::MultivariateKDE, ::Tuple{Vararg{AbstractVector{<:Real}}})
marginal
MatrixSelector
NormalScaleMatrix
PlugInMatrix
SCVMatrix
LSCVMatrix
```
