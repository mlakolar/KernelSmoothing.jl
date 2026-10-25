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

an unbiased estimate of the integrated squared error minus ``\int f^2``.

**Smoothed cross-validation.** [`SCVMatrix`](@ref) (Hall, Marron and Park 1992;
Duong and Hazelton 2005) minimizes

```math
\mathrm{SCV}(H) = \frac{(4\pi)^{-d/2}}{n |H|^{1/2}}
+ \frac{1}{W^2} \sum_{i,j} w_i w_j \big(\phi_{2H+2G} - 2\phi_{H+2G} + \phi_{2G}\big)(y_i - y_j),
```

with the pilot ``G = g^2 I`` of Duong (2004, Theorem 4), which minimizes the
squared asymptotic bias of the selected matrix and depends on the plug-in matrix
and on an estimate of ``\Theta_6 = \int (D^2)^3 f(x)\, f(x)\, dx``.

The criteria are double sums over the pairs of observations, so that a criterion
costs a time of order ``n^2``; the sums run in parallel when Julia runs with
several threads.

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
