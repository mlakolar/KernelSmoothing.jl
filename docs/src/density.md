# [Density estimation](@id density-page)

## Kernel density estimates

For observations ``x_1, \dots, x_n`` with weights ``w_i``, a kernel ``K`` and a
bandwidth ``h``, the kernel density estimate is

```math
\hat f(x) = \frac{1}{hW} \sum_{i=1}^n w_i K\Big(\frac{x - x_i}{h}\Big), \qquad W = \sum_{i=1}^n w_i .
```

`fit(KDE, x)` returns it as a continuous univariate distribution of
Distributions.jl, so that `pdf`, `logpdf`, `cdf`, `ccdf`, `quantile`, `cquantile`,
`rand`, `mean`, `var`, `minimum`, `maximum` and `insupport` work as for any
distribution. The density is
computed exactly from the observations, without a grid. A point costs a binary
search in the sorted data and a sum over the observations within ``h`` of it, and
broadcasting `pdf.(f, xs)` over many points uses the engines of
[`kernelsums`](@ref), also inside fused broadcasts such as `pdf.(f, xs .+ 1)`.
Points that are not plain numbers, such as the dual numbers of ForwardDiff.jl, go
through the scalar methods, so that the estimate can be differentiated
automatically. Bandwidths are numbers or [`BandwidthSelector`](@ref)s, by default
[`SheatherJones`](@ref); see [Bandwidth selection](@ref bandwidths-page).

The distribution function is ``\hat F(x) = W^{-1} \sum_i w_i \mathcal{K}((x - x_i)/h)``,
where ``\mathcal{K}`` is the distribution function of the kernel, and
`quantile(f, p)` is ``\inf\{x : \hat F(x) \geq p\}``, found by safeguarded Newton
steps. `ccdf(f, x)` sums the upper tail ``1 - \hat F(x)`` directly, and
`cquantile(f, p)` inverts it, so that both keep their relative accuracy far above
the data, where ``1 - \hat F(x)`` computed from ``\hat F(x)`` would round to zero. A draw from the estimate is an observation chosen with probability
proportional to its weight plus ``h`` times a draw from the kernel, the smoothed
bootstrap. The estimate of the derivative of order ``r``,

```math
\hat f^{(r)}(x) = \frac{1}{h^{r+1} W} \sum_{i=1}^n w_i K^{(r)}\Big(\frac{x - x_i}{h}\Big),
```

is available as `derivative(f, r)` for ``r`` up to the smoothness of the kernel.

`stderror(f, x)` is the plug-in estimate of the exact standard deviation of
``\hat f(x)`` for independent observations,

```math
\widehat{\mathrm{se}}(x) = \frac{1}{W} \Big( \sum_{i=1}^n w_i^2 \big(g_i - \hat f(x)\big)^2 \Big)^{1/2},
\qquad g_i = \frac{1}{h} K\Big(\frac{x - x_i}{h}\Big),
```

which with equal weights is the sample variance of the ``g_i`` divided by ``n``.
It ignores the bias, so that ``\hat f(x) \pm 1.96\, \widehat{\mathrm{se}}(x)``
covers ``f(x)`` only for bandwidths well below the ones that minimize the mean
squared error. The robust bias-corrected intervals below account for the bias, and
`confint(f, points)` gives them at the bandwidth of the estimate.

A kernel density estimate with a kernel that takes negative values is not a
density: it is negative wherever the negative lobes dominate, which breaks
`logpdf`, the monotonicity of the distribution function, `quantile` and `rand`.
Therefore, `fit(KDE, …)` accepts only nonnegative kernels that integrate to one.
Higher-order and derivative kernels are available through `derivative(f, r)` and
the pointwise estimators.

```@autodocs
Modules = [KernelSmoothing]
Pages = ["kde.jl"]
```

## Inference

[`PointwiseDensity`](@ref) estimates the density at a set of points with
conventional and robust bias-corrected standard errors and intervals (Calonico,
Cattaneo and Farrell 2018). For a second-order kernel, the leading bias of
``\hat f(x)`` is ``h^2 \mu_2(K) f''(x)/2``, where ``\mu_j(K) = \int u^j K(u)\,du``.
We estimate ``f''`` with a bandwidth ``b`` and the kernel

```math
D(u) = K(u) (c_0 + c_2 u^2), \qquad c_2 = \frac{2}{\mu_4(K) - \mu_2(K)^2}, \qquad c_0 = -\mu_2(K) c_2 ,
```

which is the equivalent kernel of the second derivative in a local quadratic fit
with kernel ``K``. This gives the bias-corrected estimate

```math
\hat f^{\mathrm{bc}}(x) = \frac{1}{hW} \sum_{i=1}^n w_i M\Big(\frac{x - x_i}{h}\Big),
\qquad M(u) = K(u) - \rho^3 \frac{\mu_2(K)}{2} D(\rho u), \qquad \rho = \frac{h}{b} .
```

The kernel ``D`` satisfies ``\int u^j D(u)\,du = 2 \delta_{j2}`` for ``j \leq 3``,
so that ``M`` is a fourth-order kernel for ``\rho = 1``. The robust standard error
is the exact standard deviation of this linear estimator, estimated as for `KDE`,
so that it includes the variability of the bias estimate. By default ``b = h``.
nprobust 1.0.0 multiplies ``D`` by ``\int u^4 D(u)\,du / 24`` instead of
``\mu_2(K)/2``, which is ``1/18`` instead of ``1/10`` for the Epanechnikov kernel
and removes only 5/9 of the leading bias. `compat = true` reproduces it.

The default bandwidths are plug-in estimates of the bandwidths that minimize the
asymptotic mean squared error at each point, or its integral over the points,

```math
h(x) = \left( \frac{R(K)\, \hat f(x)}{4n \big(\hat B(x)^2 + \hat R(x)\big)} \right)^{1/5},
\qquad \hat B(x) = \frac{\mu_2(K)}{2} \hat f''(x),
\qquad \hat R(x) = 3 \Big(\frac{\mu_2(K)}{2}\Big)^2 \hat V\big[\hat f''(x)\big],
```

where ``R(K) = \int K(u)^2\,du``. The regularization ``\hat R``, the counterpart of
the one in nprobust's regression bandwidths, keeps the bandwidth finite where
``f''`` vanishes. We estimate ``f`` at the Sheather–Jones bandwidth ``h_{SJ}`` and
``f''`` at ``b = h_{SJ}\, b_{NR}/h_{NR}``. Here ``b_{NR}`` and ``h_{NR}`` are the
normal-reference bandwidths for ``f''``, with the kernel ``D`` and the rate
``n^{-1/9}``, and for ``f``. Since ``h_{SJ}`` adapts to the roughness of the density,
so does the pilot. With `compat = true`, the pilots are nprobust's normal-reference ones, which
can be far too wide for multimodal densities.

```julia
x = randn(1000)
m = fit(PointwiseDensity, x; points = [-1, 0, 1])   # plug-in bandwidths at each point
confint(m)                                          # robust bias-corrected 95% intervals
confint(fit(KDE, x), [-1, 0, 1])                    # at the Sheather–Jones bandwidth
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["kdeinference.jl"]
```

## At boundaries

A kernel density estimate is inconsistent at a boundary of the support, since the
kernel puts part of its mass outside. [`LocalPolynomialDensity`](@ref) is the
estimator of Cattaneo, Jansson and Ma (2020), as R's lpdensity computes it, which
needs no knowledge of where the boundary lies. At a point ``x_0``, it regresses the
empirical distribution function ``\hat F`` on a polynomial of order ``p`` with
kernel weights, over the distinct values ``u_g`` with total weights ``P_g``,

```math
\hat\beta = \arg\min_\beta \sum_g P_g K_h(u_g - x_0) \Big( \hat F(u_g) - \sum_{j=0}^p \beta_j \Big(\frac{u_g - x_0}{h}\Big)^j \Big)^2 .
```

The coefficient ``v!\, \hat\beta_v / h^v`` estimates the derivative of order ``v`` of
``F``, so that ``v = 1`` gives the density and `derivative = v - 1` its
derivatives. Since the polynomial adapts to the data on the side of ``x_0`` where
they lie, the estimate keeps its rate at a boundary. The bias-corrected estimate is
the fit of order ``p + 1`` with the same bandwidth. Since ``\hat F`` is estimated
from the same data, the standard errors come from the influence function of the
estimate, a U-statistic, which reverse cumulative sums over the distinct values
compute in one pass.

The default bandwidth, [`MSEOptimal`](@ref), is the plug-in estimate of lpdensity.
It minimizes over the bandwidth ``a``, on the data standardized by their mean and
standard deviation, the estimated mean squared error

```math
a^{2p + 2 - 2v} \big( |B_1| + a\, |B_2| \big)^2 + \frac{V}{a^{2v - 1}} ,
```

where ``B_1`` and ``B_2`` are the first two terms of the bias and ``V`` is the
variance constant. The bias terms come from local estimates of ``F^{(p+1)}`` and
``F^{(p+2)}`` at rule-of-thumb pilot bandwidths. lpdensity uses ``(B_1 + a B_2)^2``, which vanishes
at ``a = -B_1/B_2`` when the estimates have opposite signs, so that it can choose
bandwidths of the order of the range of the data. `compat = true` restores its
rule, and `NormalReference()` gives its rule of thumb for normal data.

```julia
using Random
x = randexp(5000)
m = fit(LocalPolynomialDensity, x; points = [0.0, 0.5, 1.0])   # consistent at the boundary 0
confint(m)
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["lpdensity.jl"]
```
