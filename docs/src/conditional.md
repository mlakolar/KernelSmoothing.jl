# [Conditional densities and quantiles](@id conditional-page)

## Conditional density estimation

For observations ``(x_i, y_i)`` with weights ``w_i``, the kernel estimate of the
conditional density of ``y`` given ``x`` smooths the kernels of the responses as a
local polynomial regression smooths the responses:

```math
\hat f(y \mid x) = \sum_{i=1}^n \ell_i(x)\, \frac{1}{b} K\Big(\frac{y - y_i}{b}\Big),
```

where ``K`` is the kernel in ``y`` with bandwidth ``b`` and ``\ell_i(x)`` are the
weights of the local polynomial fit of order ``p`` at ``x`` with kernel ``k`` and
bandwidth ``h``. For ``p = 0`` (Rosenblatt 1969; Hyndman, Bashtannyk and Grunwald
1996),

```math
\ell_i(x) = \frac{w_i\, k((x_i - x)/h)}{\sum_j w_j\, k((x_j - x)/h)},
```

which are nonnegative and sum to one, so that ``\hat f(\cdot \mid x)`` is a density:
`conditional(m, x)` returns it as a [`KDE`](@ref) of the responses with these
weights, with its distribution function and quantiles. For ``p = 1`` (Fan, Yao and
Tong 1996), the estimate has a smaller bias near the ends of the range of ``x``, but
the weights can be negative.

[`ConditionalCV`](@ref) chooses ``(h, b)`` for ``p = 0`` by maximizing the
leave-one-out log-likelihood ``\sum_i w_i \log \hat f_{-i}(y_i \mid x_i)``, the default
of the R package np, or by minimizing the least-squares criterion of Hall, Racine
and Li (2004),

```math
\mathrm{CV}(h, b) = \frac{1}{W} \sum_{i=1}^n w_i \int \hat f_{-i}(y \mid x_i)^2\, dy
- \frac{2}{W} \sum_{i=1}^n w_i\, \hat f_{-i}(y_i \mid x_i),
```

where ``\hat f_{-i}`` leaves observation ``i`` out of the numerator and the
denominator of the estimate.

## Local polynomial quantile regression

The local polynomial estimate of the conditional quantile ``q_\tau(x)`` of order
``\tau`` (Yu and Jones 1998) minimizes at each point ``x_0`` the kernel-weighted check
loss,

```math
\hat\beta = \arg\min_\beta \sum_{i=1}^n w_i\, k\Big(\frac{x_i - x_0}{h}\Big)\,
\rho_\tau\Big(y_i - \sum_{j=0}^p \beta_j (x_i - x_0)^j\Big),
\qquad \rho_\tau(u) = u\,(\tau - 1\{u < 0\}),
```

and ``\nu!\, \hat\beta_\nu`` estimates the derivative of order ``\nu`` of ``q_\tau``. Each
local fit is a linear program, which the simplex method solves exactly on its dual,

```math
\max_a \sum_i v_i y_i a_i \quad \text{subject to} \quad \sum_i v_i z_i a_i = (1 - \tau) \sum_i v_i z_i,
\quad 0 \le a_i \le 1,
```

with ``v_i`` the kernel weights times the weights of the observations and ``z_i`` the
local design; the multipliers of the constraints are ``\hat\beta``, and the fit
interpolates ``p + 1`` observations of its window. The default bandwidth,
[`YuJones`](@ref), rescales the cross-validation bandwidth ``h`` of local polynomial
mean regression,

```math
h_\tau = h \Big(\frac{\tau (1 - \tau)}{\phi(\Phi^{-1}(\tau))^2}\Big)^{1/5}.
```

The standard errors come from the sandwich of a weighted quantile regression,
``\tau(1 - \tau)\, s^2 (Z^\top V Z)^{-1} (Z^\top V^2 Z) (Z^\top V Z)^{-1}``, with the
sparsity ``s = 1/f(q_\tau(x_0) \mid x_0)`` estimated by the difference quotient of local
fits at the orders ``\tau \pm \delta``, ``\delta`` the bandwidth of Hall and Sheather
(1988). They ignore the bias.

## Example

```julia
using KernelSmoothing

x = randn(500); y = sin.(x) .+ 0.4 .* (1 .+ x .^ 2 ./ 2) .* randn(500)
m = fit(ConditionalDensity, x, y)        # bandwidths by likelihood cross-validation
pdf(m, 0.5, 1.0), cdf(m, 0.5, 1.0), quantile(m, 0.9, 1.0)
conditional(m, 1.0)                      # the conditional distribution at x = 1, a KDE

q = fit(LocalQuantileRegression, x, y; tau = 0.9, points = -2:0.25:2)
coef(q), confint(q)
```

## Reference

```@docs
ConditionalDensity
fit(::Type{ConditionalDensity}, ::AbstractVector, ::AbstractVector)
pdf(::ConditionalDensity, ::Real, ::Real)
cdf(::ConditionalDensity, ::Real, ::Real)
quantile(::ConditionalDensity, ::Real, ::Real)
conditional
ConditionalCV
LocalQuantileRegression
fit(::Type{LocalQuantileRegression}, ::AbstractVector, ::AbstractVector)
stderror(::LocalQuantileRegression)
confint(::LocalQuantileRegression)
YuJones
```
