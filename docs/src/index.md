# KernelSmoothing.jl

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference. The package is under
development.

The package is built in layers. [Kernels](@ref kernels-page) are the smoothing
kernels with their distribution functions, moments, derivatives, convolutions
and products. [Prepared data](@ref data-page) is a sample sorted once, with its
weights and ties recorded. [Kernel sums](@ref sums-page) are the sums
`∑ wᵢ zᵢ k(uᵢ) uᵢʳ` that all estimators are built from, computed by exact
direct, windowed and fast engines or approximately from binned data. On top of
them, [kernel density estimates](@ref density-page) are distributions of
Distributions.jl, and [local polynomial regression](@ref regression-page) comes
with conventional and robust bias-corrected inference.

```@docs
KernelSmoothing
```

## Example

```julia
using KernelSmoothing

x = randn(1000)
f = fit(KDE, x)                     # Epanechnikov kernel, normal-reference bandwidth
pdf(f, 0.0), cdf(f, 0.0), quantile(f, 0.975)
pdf.(f, range(-3, 3; length = 401))  # many points at once, exactly
derivative(f, 1)(0.5)                # the estimated derivative of the density
stderror(f, 0.0)                     # standard error of the estimate at 0
```

## Bandwidth convention

The compactly supported kernels live on `[-1, 1]`, so the bandwidth `h` is the
half-width of the smoothing window: an observation gets weight zero unless it is
within `h` of the point. The Gaussian kernel is the standard normal density, so
its bandwidth is a standard deviation. KernelDensity.jl, KDEpy and R's `density`
instead scale every kernel to unit variance; their bandwidth `h_sd` corresponds
to the half-width `h_sd / sqrt(moment(k, 2))` here, for example `h_sd √5` for the
Epanechnikov kernel.

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/mlakolar/KernelSmoothing.jl")
```
