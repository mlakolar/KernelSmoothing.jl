# [Densities on a bounded support](@id bounded-page)

A kernel density estimate [`KDE`](@ref) puts mass outside the support of the
data: near a boundary at ``a``, the kernels of the observations within a bandwidth
of ``a`` spill over it, so that the estimate is too low at the boundary and does
not integrate to one on the support. `fit(BoundedKDE, x; support = (a, b), method)`
returns an estimate that is a distribution on the support, with `pdf`, `logpdf`,
`cdf`, `ccdf`, `quantile`, `rand`, `mean` and `var`. Either boundary can be
infinite, as in `support = (0, Inf)`. For estimates of the density at a boundary
with confidence intervals, see [`LocalPolynomialDensity`](@ref).

## Methods

**Reflection** (Schuster 1985). The estimate of the observations and their mirror
images at the boundaries (for an upper boundary ``b`` alone, the images ``2b - x_i``),

```math
\hat f(x) = \frac{1}{hW} \sum_{i=1}^n w_i \sum_{m \in \mathbb{Z}}
\Big[ K\Big(\frac{x - x_i - 2mL}{h}\Big) + K\Big(\frac{x + x_i - 2a - 2mL}{h}\Big) \Big],
\qquad a \leq x \leq b,
```

with ``L = b - a``, and only the terms ``m = 0`` of the first sum and of the images
at ``a`` when ``b = \infty``. It is the density of ``x_i + hε`` folded into
``[a, b]``, so that it integrates to one, and it is computed exactly from the
observations, with its distribution function. Its derivative at a boundary is
zero, so that it has a bias of order ``h`` there unless ``f'(a) = 0``. The bandwidth
rule is applied to the observations and their first images: their estimate is the
reflected density, whose optimal bandwidth is that of ``f`` on the support.

**Beta kernels** (Chen 1999), on ``[a, b]``, in the variable ``u = (x - a)/(b - a)``:

```math
\tilde f(u) = \sum_{i=1}^n \frac{w_i}{W}\, \beta\Big(u_i; \frac{u}{s} + 1, \frac{1 - u}{s} + 1\Big),
```

where ``\beta(\cdot; p, q)`` is the Beta(p, q) density and ``s`` the smoothing
parameter. The kernel changes its shape with ``u`` so that it never leaves the
support, and the bias is of order ``s`` everywhere, ``s\{(1 - 2u) f'(u) + \tfrac12
u(1 - u) f''(u)\}``. The modified estimator of Chen (1999) uses ``\beta(\cdot;
u/s, (1-u)/s)`` in the interior, whose bias does not involve ``f'``, and modified
first and second parameters within ``2s`` of the boundaries.

**Gamma kernels** (Chen 2000), on ``[a, \infty)``, with ``u = x - a``:
``\tilde f(u) = \sum_i (w_i/W)\, \gamma(u_i; u/s + 1, s)``, the gamma density with
shape ``u/s + 1`` and scale ``s``, and the modified version with shape ``u/s``
above ``2s`` and ``(u/(2s))^2 + 1`` below.

Neither the beta nor the gamma kernel estimate integrates to one. The estimate is
``\tilde f`` divided by its integral, computed by Gauss–Legendre quadrature on panels
a quarter of the spread of the kernel wide, over the range where the estimate is
not negligible; the same panels give the distribution function and its upper tail,
and the quantiles invert it within a panel. The computations for gamma kernels are
done on the data divided by their mean distance from the boundary, so that they do
not depend on the scale of the data. An observation exactly at a boundary gets no
weight from these kernels anywhere inside the support, so such observations are
rejected. The
smoothing parameter is chosen by least-squares cross-validation of ``\tilde f``,
``\int \tilde f^2 - 2 \sum_i w_i \tilde f_{-i}(u_i)``, over a grid around the
parameter whose kernel has the variance of a normal-reference bandwidth.

**Transformation** (Wand, Marron and Ruppert 1991), `Transformed()`. For a map ``t`` of the support
onto the real line, the estimate ``\hat g`` of the density of ``t(x_i)``,
transformed back:

```math
\hat f(x) = \hat g\big(t(x)\big)\, |t'(x)|, \qquad \hat F(x) = \hat G\big(t(x)\big),
```

with ``t(x) = \log(x - a)`` for ``[a, \infty)``, ``-\log(b - x)`` for ``(-\infty, b]``, and the logit or probit of
``(x - a)/(b - a)`` for ``[a, b]``. The estimate is a distribution exactly, its
quantiles and draws come from those of ``\hat g``, and its effective bandwidth
grows with the distance from the boundary. It suits densities that are steep or
infinite at a boundary, which the other methods smooth away.

## Example

```julia
using KernelSmoothing, Distributions

x = rand(Exponential(), 500)
f = fit(BoundedKDE, x; support = (0, Inf))                          # reflection
g = fit(BoundedKDE, x; support = (0, Inf), method = GammaKernels())
pdf(f, 0.0), pdf(g, 0.0), cdf(g, 1.0), quantile(g, 0.9)

y = rand(Beta(0.5, 0.5), 500)                                       # infinite at both ends
fit(BoundedKDE, y; support = (0, 1), method = Transformed())
```

## Reference

```@docs
BoundedKDE
fit(::Type{BoundedKDE}, ::AbstractVector)
BoundaryMethod
Reflection
BetaKernels
GammaKernels
Transformed
```
