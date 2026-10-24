# [Kernels](@id kernels-page)

A kernel ``K`` enters an estimator through the weights ``K((x_i - x)/h)/h``, where
``h > 0`` is the bandwidth. The package provides six second-order kernels, that is,
symmetric probability densities with a finite and positive second moment.

| kernel | ``K(u)`` on its support | ``\mu_2(K)`` | ``R(K)`` | smoothness |
|---|---|---|---|---|
| [`UniformKernel`](@ref) | ``1/2`` | ``1/3`` | ``1/2`` | 0 |
| [`TriangularKernel`](@ref) | ``1 - \lvert u \rvert`` | ``1/6`` | ``2/3`` | 1 |
| [`EpanechnikovKernel`](@ref) | ``\tfrac{3}{4} (1 - u^2)`` | ``1/5`` | ``3/5`` | 1 |
| [`BiweightKernel`](@ref) | ``\tfrac{15}{16} (1 - u^2)^2`` | ``1/7`` | ``5/7`` | 2 |
| [`TriweightKernel`](@ref) | ``\tfrac{35}{32} (1 - u^2)^3`` | ``1/9`` | ``350/429`` | 3 |
| [`GaussianKernel`](@ref) | ``\phi(u) = e^{-u^2/2}/\sqrt{2\pi}`` | ``1`` | ``1/(2\sqrt{\pi})`` | ``\infty`` |

Here ``\mu_2(K) = \int u^2 K(u)\,du`` is the variance of the kernel and
``R(K) = \int K(u)^2\,du`` its roughness. The compactly supported kernels live on
``[-1, 1]``, so that ``h`` is the half-width of the smoothing window, as in
nprobust, rdrobust and lpdensity. The Gaussian kernel is the standard normal
density, so that its bandwidth is a standard deviation.

Every kernel can be evaluated as `k(u)` and supports the following functions.

| function | value |
|---|---|
| `support(k)` | the interval `(lo, hi)` outside which ``K`` vanishes, as a tuple |
| `insupport(k, u)` | whether `lo ≤ u ≤ hi` |
| `cdf(k, u)` | ``\int_{-\infty}^{u} K(t)\,dt`` |
| `ccdf(k, u)` | ``\int_{u}^{\infty} K(t)\,dt``, accurate where it is small |
| `moment(k, j)` | ``\int u^j K(u)\,du`` |
| `roughness(k)` | ``\int K(u)^2\,du`` |
| `smoothness(k)` | the number of derivatives of ``K`` that are functions |
| `derivative(k, r)` | ``K^{(r)}``, as a kernel |
| `convolve(k1, k2)` | ``K_1 * K_2``, as a kernel |
| `k1 * k2` | the product ``K_1 K_2``, as a kernel |

Several packages export a function named `derivative`, Polynomials.jl among them.
With one of them loaded as well, Julia asks for the name to be qualified, as in
`KernelSmoothing.derivative(k, 2)`. The same holds for `GaussianKernel` and
`PiecewisePolynomialKernel` with KernelFunctions.jl, whose kernels of the same names
are covariance functions of Gaussian processes. Since `support(k)` is a tuple,
`u in support(k)` compares `u` with the ends; `insupport(k, u)` tests membership.

Since derivatives, convolutions and products are kernels again, the estimators
treat them like any other kernel. A kernel has smoothness ``s`` if
``K \in C^{s-1}`` and ``K^{(s)}`` exists as a function, possibly with jumps; then
``K^{(r)}`` for ``r \leq s`` is the derivative in the sense of distributions, and
`derivative(k, r)` refuses larger orders, whose distributional derivatives have
point masses. Integration by parts gives the moments of a derivative,

```math
\int u^j K^{(r)}(u)\,du = (-1)^r \frac{j!}{(j - r)!}\, \mu_{j-r}(K), \qquad j \geq r .
```

The compactly supported kernels and their derivatives, convolutions and products
are polynomials between breakpoints, which
`PiecewisePolynomialKernel(breaks, coefs)` also lets the user define, for example
higher-order kernels. Each piece is stored as a polynomial about its midpoint,
which keeps evaluation accurate far from zero, and convolutions and products are
computed exactly in this representation. The named kernels are evaluated as powers
of ``1 - u^2`` and their distribution functions near ``-1`` as
``(1 + u)^m P(u)``, with the terms of ``P`` of one sign, since the expanded
polynomials cancel near ``\pm 1`` and can turn negative there. For the same reason,
a piecewise polynomial kernel keeps its distribution function on the first piece
and its upper tail on the last piece as polynomials about the ends of the support,
computed in extended precision from the exact pieces of a convolution or a product,
so that both keep their relative accuracy at the ends. Derivatives, convolutions and products of
Gaussian kernels stay in the family

```math
K_{\sigma, r, a}(u) = \frac{a}{\sigma^{r+1}}\, \phi^{(r)}\Big(\frac{u}{\sigma}\Big),
\qquad \phi^{(r)}(z) = (-1)^r \mathrm{He}_r(z)\, \phi(z),
```

where ``\mathrm{He}_r`` is the probabilists' Hermite polynomial of degree ``r``;
a convolution adds the variances and the orders.

```@autodocs
Modules = [KernelSmoothing]
Pages = ["kernels.jl"]
```
