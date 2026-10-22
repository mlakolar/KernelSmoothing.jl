"""
    KDE <: ContinuousUnivariateDistribution

A kernel density estimate

    f̂(x) = ∑ᵢ wᵢ k((x - xᵢ) / h) / (h ∑ᵢ wᵢ),

fitted with [`fit`](@ref)`(KDE, x)`. A `KDE` is a continuous univariate
distribution of Distributions.jl, so `pdf`, `logpdf`, `cdf`, `quantile`, `rand`,
`mean`, `var`, `minimum`, `maximum` and `insupport` work as for any other
distribution. The density and the distribution function are computed exactly from
the observations, without a grid. Broadcasting `pdf.(f, xs)`, `logpdf.(f, xs)`,
`cdf.(f, xs)` or `stderror.(f, xs)` over many points computes them with
[`kernelsums`](@ref), which is much faster than evaluating point by point.

[`derivative`](@ref)`(f, r)` estimates the `r`-th derivative of the density,
[`stderror`](@ref)`(f, x)` gives the standard error of `f̂(x)` and
[`bandwidth`](@ref)`(f)` the bandwidth.
"""
struct KDE{K<:AbstractKernel,RK<:AbstractKernel,D<:SmoothingData,S<:BandwidthSelection,
           E<:Union{Nothing,SumEngine}} <: Distributions.ContinuousUnivariateDistribution
  data::D
  kernel::K
  reflected::RK   # u ↦ k(-u), since the sums are over u = (xᵢ - x) / h
  h::Float64
  selection::S
  engine::E
end

"""
    fit(KDE, x; kernel = EpanechnikovKernel(), bandwidth = NormalReference(), weights = nothing, engine = nothing)
    fit(KDE, d::SmoothingData; kernel, bandwidth, engine)

The kernel density estimate of the sample `x`, with optional `weights`, or of a
prepared sample `d`. The `bandwidth` is a positive number or a
[`BandwidthSelector`](@ref); with the default, [`NormalReference`](@ref), it
minimizes the asymptotic mean integrated squared error for normal data. It is a
half-width for the compactly supported kernels and a standard deviation for the
Gaussian kernel.

The kernel must be a probability density, nonnegative and integrating to one, so
that the estimate is one too. Kernels that take negative values, such as
higher-order kernels and derivatives of kernels, are rejected; for derivatives of
the density, see [`derivative`](@ref)`(f, r)`.

`engine` fixes the [`SumEngine`](@ref) that evaluates the estimate at many points
at once; by default the faster exact engine is chosen for each evaluation.

# Examples
```julia
julia> f = fit(KDE, [0.0, 1.0, 1.5, 3.0]; bandwidth = 1.0);

julia> pdf(f, 1.0), cdf(f, 1.0)
(0.328125, 0.4140625)

julia> quantile(f, 0.5)
1.25
```
"""
function fit(::Type{KDE}, x::AbstractVector{<:Real}; weights = nothing, kwargs...)
  return fit(KDE, SmoothingData(x; weights); kwargs...)
end

function fit(::Type{KDE}, d::SmoothingData; kernel::AbstractKernel = EpanechnikovKernel(),
             bandwidth = NormalReference(), engine::Union{Nothing,SumEngine} = nothing)
  _isnonnegative(kernel) || throw(ArgumentError("$kernel takes negative values, so a " *
    "kernel density estimate with it is not a density; use a nonnegative kernel, and " *
    "derivative(f, r) for derivatives of the density"))
  mass = moment(kernel, 0)
  isapprox(mass, 1; rtol = 1e-10) ||
    throw(ArgumentError("$kernel integrates to $mass, but a density kernel must integrate to one"))
  sel = _selection(bandwidth, d, kernel)
  sel.h isa Float64 || throw(ArgumentError("a kernel density estimate needs one bandwidth, got one per point"))
  return KDE(d, kernel, _reflect(kernel), sel.h, sel, engine)
end

function Base.show(io::IO, f::KDE)
  print(io, "KDE(", f.kernel, ", bandwidth = ", round(f.h; sigdigits = 4), ", ",
        nobs(f.data), " observations", f.data.w === nothing ? "" : ", weighted", ")")
end

bandwidth(f::KDE) = f.h
nobs(f::KDE) = nobs(f.data)

# ∑ᵢ vᵢ k((xᵢ - x) / h) over the observations where k can be nonzero; vᵢ = wᵢ, or
# 1 without weights
function _pointsum(d::SmoothingData, k::AbstractKernel, h::Float64, x::Real, v)
  xs, n = d.x, nobs(d)
  lo, hi = support(k)
  a = isfinite(lo) ? _firsttrue(i -> (xs[i] - x) / h >= lo, n) : 1
  b = isfinite(hi) ? _firsttrue(i -> (xs[i] - x) / h > hi, n) : n + 1
  s = zero(float(x))
  @inbounds for i in a:b-1
    s += _weight(v, i) * k((xs[i] - x) / h)
  end
  return s
end

# vector(points) at the finite entries of x and scalar(t) at the others, in the
# shape of x
function _evaluate(vector, scalar, x::AbstractArray{<:Real})
  out = similar(x, Float64)
  idx = [i for i in eachindex(x) if isfinite(x[i])]
  if !isempty(idx)
    v = vector(Float64[x[i] for i in idx])
    for (j, i) in enumerate(idx)
      out[i] = v[j]
    end
  end
  for i in eachindex(x)
    isfinite(x[i]) || (out[i] = scalar(x[i]))
  end
  return out
end

_sums(f::KDE, k, points; z = nothing, engine = f.engine) =
  vec(kernelsums(f.data, k, f.h, points; z, engine))

function pdf(f::KDE, x::Real)
  isnan(x) && return float(x)
  return _pointsum(f.data, f.reflected, f.h, x, f.data.w) / (f.h * f.data.sumw)
end

logpdf(f::KDE, x::Real) = log(pdf(f, x))

# clamped at zero, since fast sums can round a zero density to a tiny negative number
_pdf(f::KDE, x) =
  _evaluate(p -> max.(_sums(f, f.reflected, p) ./ (f.h * f.data.sumw), 0.0), t -> pdf(f, t), x)

function cdf(f::KDE, x::Real)
  isnan(x) && return float(x)
  x <= minimum(f) && return 0.0
  x >= maximum(f) && return 1.0
  d, k, h = f.data, f.kernel, f.h
  xs, n = d.x, nobs(d)
  lo, hi = support(k)
  # observations with (x - xᵢ) / h > hi count fully, those below lo not at all
  a = isfinite(hi) ? _firsttrue(i -> (xs[i] - x) / h >= -hi, n) : 1
  b = isfinite(lo) ? _firsttrue(i -> (xs[i] - x) / h > -lo, n) : n + 1
  s = zero(float(x))
  @inbounds for i in a:b-1
    s += _weight(d.w, i) * cdf(k, (x - xs[i]) / h)
  end
  return clamp((d.cumw[a] + s) / d.sumw, 0.0, 1.0)
end

function _cdf(f::KDE, x)
  d, k, h = f.data, f.kernel, f.h
  g = _cdfkernel(k)
  lo = first(support(g))
  return _evaluate(t -> cdf(f, t), x) do p
    s = _sums(f, g, p)
    xmin, xmax = minimum(f), maximum(f)
    for j in eachindex(p)
      if isfinite(lo)   # add the weight of the observations below the window
        a = _firsttrue(i -> (d.x[i] - p[j]) / h >= lo, nobs(d))
        s[j] += d.cumw[a]
      end
      s[j] = p[j] <= xmin ? 0.0 : p[j] >= xmax ? 1.0 : clamp(s[j] / d.sumw, 0.0, 1.0)
    end
    s
  end
end

function quantile(f::KDE, p::Real)
  0 <= p <= 1 || throw(DomainError(p, "a probability must be between 0 and 1"))
  p == 0 && return minimum(f)
  p == 1 && return maximum(f)
  # F̂(x) lies between k's distribution function at (x - xₙ) / h and at (x - x₁) / h
  q = _kernelquantile(f.kernel, p)
  return _invertcdf(t -> cdf(f, t), t -> pdf(f, t), p, f.data.x[1] + f.h * q,
                    f.data.x[end] + f.h * q)
end

Base.minimum(f::KDE) = f.data.x[1] + f.h * first(support(f.kernel))
Base.maximum(f::KDE) = f.data.x[end] + f.h * last(support(f.kernel))
insupport(f::KDE, x::Real) = minimum(f) <= x <= maximum(f)

mean(f::KDE) = _mean(f.data) + f.h * moment(f.kernel, 1)
var(f::KDE) =
  _variance(f.data) + f.h^2 * (moment(f.kernel, 2) - moment(f.kernel, 1)^2)

# the smoothed bootstrap: an observation drawn by weight plus h times a draw from k
function Base.rand(rng::AbstractRNG, f::KDE)
  d = f.data
  i = d.w === nothing ? rand(rng, 1:nobs(d)) : searchsortedlast(d.cumw, rand(rng) * d.sumw)
  return d.x[i] + f.h * _randkernel(rng, f.kernel)
end

"""
    stderror(f::KDE, x)

The standard error of the density estimate `f̂(x)`,

    √(∑ᵢ wᵢ² (gᵢ - f̂(x))²) / ∑ᵢ wᵢ,   gᵢ = k((x - xᵢ) / h) / h,

the plug-in estimate of its exact variance for independent observations; with
equal weights, the sample variance of the `gᵢ` divided by `n`. It measures how
`f̂(x)` varies around its expectation, not its bias, so `f̂(x) ± 1.96 stderror(f, x)`
covers the density only for bandwidths small enough that the bias is negligible,
which the bandwidths that minimize the mean squared error are not.
"""
function stderror(f::KDE, x::Real)
  isnan(x) && return float(x)
  d, k, h = f.data, f.reflected, f.h
  xs, n = d.x, nobs(d)
  lo, hi = support(k)
  a = isfinite(lo) ? _firsttrue(i -> (xs[i] - x) / h >= lo, n) : 1
  b = isfinite(hi) ? _firsttrue(i -> (xs[i] - x) / h > hi, n) : n + 1
  s0 = s1 = s2 = 0.0   # ∑ wᵢ kᵢ, ∑ wᵢ² kᵢ and ∑ wᵢ² kᵢ² with kᵢ = k((xᵢ - x) / h)
  @inbounds for i in a:b-1
    w, ki = _weight(d.w, i), k((xs[i] - x) / h)
    s0 += w * ki
    s1 += w^2 * ki
    s2 += w^2 * ki^2
  end
  return _kdese(f, s0, s1, s2)
end

# ∑ wᵢ² (gᵢ - f̂)² = ∑ wᵢ² gᵢ² - 2 f̂ ∑ wᵢ² gᵢ + f̂² ∑ wᵢ², with ∑ wᵢ² = W² / neff
function _kdese(f::KDE, s0, s1, s2)
  d, h, W = f.data, f.h, f.data.sumw
  fx = s0 / (h * W)
  v = s2 / h^2 - 2 * fx * s1 / h + fx^2 * W^2 / d.neff
  return sqrt(max(v, 0.0)) / W
end

# The variance is a small difference of sums where the estimate is nearly zero, and
# the square root magnifies the rounding of fast sums there, so the sums are taken
# over the windows unless the estimate fixes an engine.
function _stderror(f::KDE, x)
  k = f.reflected
  engine = f.engine === nothing && all(isfinite, support(k)) ? WindowSums() : f.engine
  return _evaluate(t -> stderror(f, t), x) do p
    s0 = _sums(f, k, p; engine)
    s1 = f.data.w === nothing ? s0 : _sums(f, k, p; z = f.data.w, engine)
    s2 = _sums(f, _square(k), p; z = f.data.w, engine)
    _kdese.(Ref(f), s0, s1, s2)
  end
end

"""
    derivative(f::KDE, r = 1)

The estimate of the `r`-th derivative of the density,

    f̂⁽ʳ⁾(x) = ∑ᵢ wᵢ k⁽ʳ⁾((x - xᵢ) / h) / (h^(r + 1) ∑ᵢ wᵢ),

the `r`-th derivative of `f̂`, as a function: `derivative(f, r)(x)`, and
`derivative(f, r).(xs)` for many points at once. The kernel must have `r`
derivatives that are functions, see [`smoothness`](@ref): the Epanechnikov
kernel allows `r = 1`, the biweight kernel `r ≤ 2` and the Gaussian kernel any
order. Derivatives of a density are estimated best with a larger bandwidth than
the density itself.
"""
function derivative(f::KDE, r::Integer = 1)
  _check_order(r)
  return KDEDerivative(f, Int(r), _reflect(derivative(f.kernel, r)))
end

struct KDEDerivative{F<:KDE,K<:AbstractKernel}
  kde::F
  r::Int
  reflected::K   # u ↦ k⁽ʳ⁾(-u)
end

function (g::KDEDerivative)(x::Real)
  isnan(x) && return float(x)
  f = g.kde
  return _pointsum(f.data, g.reflected, f.h, x, f.data.w) / (f.h^(g.r + 1) * f.data.sumw)
end

function Base.show(io::IO, g::KDEDerivative)
  print(io, "derivative of order ", g.r, " of ", g.kde)
end

const _KDELike = Union{KDE,Base.RefValue{<:KDE}}
_unref(f::KDE) = f
_unref(f::Base.RefValue{<:KDE}) = f[]

# broadcasting over many points sums with an engine
Base.broadcasted(::typeof(pdf), f::_KDELike, x::AbstractArray{<:Real}) = _pdf(_unref(f), x)
Base.broadcasted(::typeof(logpdf), f::_KDELike, x::AbstractArray{<:Real}) = log.(_pdf(_unref(f), x))
Base.broadcasted(::typeof(cdf), f::_KDELike, x::AbstractArray{<:Real}) = _cdf(_unref(f), x)
Base.broadcasted(::typeof(stderror), f::_KDELike, x::AbstractArray{<:Real}) =
  _stderror(_unref(f), x)

function Base.broadcasted(g::KDEDerivative, x::AbstractArray{<:Real})
  f = g.kde
  return _evaluate(p -> _sums(f, g.reflected, p) ./ (f.h^(g.r + 1) * f.data.sumw), g, x)
end
