## Kernel density estimation on a bounded support

"""
    BoundaryMethod

Supertype of the boundary corrections of a [`BoundedKDE`](@ref):
[`Reflection`](@ref), [`BetaKernels`](@ref), [`GammaKernels`](@ref) and
[`Transformed`](@ref).
"""
abstract type BoundaryMethod end

"""
    Reflection()

Reflection at the boundaries (Schuster 1985). With a lower boundary `a`, the
estimate is

    f̂(x) = ∑ᵢ wᵢ (k((x - xᵢ)/h) + k((x + xᵢ - 2a)/h)) / (h W),   x ≥ a,

the kernel estimate of the observations and their mirror images, restricted to the
support, and symmetrically for an upper boundary. With two boundaries `a < b`, the
images are reflected at both, repeatedly, so that `f̂` is the density of `xᵢ + h ε`
folded into `[a, b]`, for an observation `xᵢ` drawn by weight and `ε` from the kernel.
The estimate integrates to one and is computed exactly, with its distribution
function. Its derivative vanishes at a boundary, so that it has a bias of order `h`
there unless the density's derivative vanishes too. The kernel must be symmetric;
a bandwidth rule is applied to the observations and their first images, which
targets the bandwidth of the density on the support.
"""
struct Reflection <: BoundaryMethod end

"""
    BetaKernels(; modified = false)

The beta kernel estimator of Chen (1999) on `[a, b]`: with `u = (x - a)/(b - a)` and
`uᵢ = (xᵢ - a)/(b - a)`,

    f̃(x) = ∑ᵢ wᵢ β(uᵢ; u/s + 1, (1 - u)/s + 1) / ((b - a) W),

where `β(⋅; p, q)` is the density of the Beta(p, q) distribution and `s` the
smoothing parameter, given in `bandwidth`. The kernel changes its shape with the
point, so that it never puts mass outside the support, and the bias is of order
`s` everywhere. With `modified = true`, the second estimator of Chen (1999), whose
interior bias does not involve `f'`: `β(⋅; u/s, (1 - u)/s)` for `u ∈ [2s, 1 - 2s]`,
with the first parameter replaced by `ρ(u) = 2s² + 2.5 - √(4s⁴ + 6s² + 2.25 - u² -
u/s)` for `u < 2s` and the second by `ρ(1 - u)` for `u > 1 - 2s`; it needs `s < 1/4`.
Neither integrates to one; the estimate is `f̃` divided by its integral, computed by
quadrature, which also gives the distribution function.

An observation exactly at a boundary would get no weight anywhere inside the
support, since the kernels vanish there, so such observations are rejected; shift
the support slightly or use [`Reflection`](@ref).
"""
struct BetaKernels <: BoundaryMethod
  modified::Bool
end

BetaKernels(; modified::Bool = false) = BetaKernels(modified)

"""
    GammaKernels(; modified = false)

The gamma kernel estimator of Chen (2000) on `[a, ∞)`: with `u = x - a` and
`uᵢ = xᵢ - a`,

    f̃(x) = ∑ᵢ wᵢ γ(uᵢ; u/s + 1, s) / W,

where `γ(⋅; κ, s)` is the density of the gamma distribution with shape `κ` and scale
`s`, the smoothing parameter given in `bandwidth`. With `modified = true`, the
second estimator of Chen (2000), with shape `u/s` for `u ≥ 2s` and `(u/(2s))² + 1`
below. As for [`BetaKernels`](@ref), the estimate is `f̃` divided by its integral,
and observations exactly at the boundary are rejected. A support `(-∞, b]` is
handled by reflecting the data at `b`.
"""
struct GammaKernels <: BoundaryMethod
  modified::Bool
end

GammaKernels(; modified::Bool = false) = GammaKernels(modified)

"""
    Transformed(transform = :auto)

The transformation estimator (Wand, Marron and Ruppert 1991; Marron and Ruppert
1994): the kernel density estimate `ĝ` of `t(xᵢ)` for a map `t` of the support onto
the real line, transformed back,

    f̂(x) = ĝ(t(x)) |t'(x)|,   F̂(x) = Ĝ(t(x)),

with `t(x) = log(x - a)` for a support `[a, ∞)` (`transform = :log`), `-log(b - x)` for
`(-∞, b]`, and `t(x) = log((x - a)/(b - x))` (`:logit`) or `Φ⁻¹((x - a)/(b - a))`
(`:probit`) for `[a, b]`; `:auto` takes `:log` or `:logit`. The estimate is a
distribution exactly, its quantiles and draws come from those of `ĝ`, and the
bandwidth, chosen on the transformed data and given on that scale, varies with `x`
on the original scale. It suits densities that are infinite or steep at a
boundary. The observations must lie strictly inside the support.
"""
struct Transformed <: BoundaryMethod
  transform::Symbol

  function Transformed(transform::Symbol)
    transform in (:auto, :log, :logit, :probit) ||
      throw(ArgumentError("transform must be :auto, :log, :logit or :probit, got :$transform"))
    new(transform)
  end
end

Transformed(; transform::Symbol = :auto) = Transformed(transform)

"""
    BoundedKDE <: ContinuousUnivariateDistribution

A kernel density estimate on a support `[a, b]` with one or two finite
boundaries, which is a distribution on the support: its density integrates to one
there, and `pdf`, `logpdf`, `cdf`, `ccdf`, `quantile`, `rand`, `mean`, `var`,
`minimum`, `maximum` and `insupport` work as for any distribution of
Distributions.jl. Fitted with [`fit`](@ref)`(BoundedKDE, x; support, method)`, see
[`BoundaryMethod`](@ref) for the methods. [`bandwidth`](@ref)`(f)` returns the
bandwidth, or the smoothing parameter of beta and gamma kernels.

Unlike [`KDE`](@ref), which has no boundary correction, the estimate does not spill
mass across the boundaries. For estimates of the density at a boundary with
robust bias-corrected intervals, see [`LocalPolynomialDensity`](@ref).
"""
struct BoundedKDE{M<:BoundaryMethod,I,D<:SmoothingData,S<:BandwidthSelection} <:
       Distributions.ContinuousUnivariateDistribution
  method::M
  lower::Float64
  upper::Float64
  data::D            # the observations
  h::Float64
  impl::I            # what the method evaluates the estimate with
  selection::S
end

"""
    fit(BoundedKDE, x; support, method = Reflection(), kernel = EpanechnikovKernel(),
        bandwidth = nothing, weights = nothing)
    fit(BoundedKDE, d::SmoothingData; support, method, kernel, bandwidth)

The kernel density estimate of the sample `x` on the `support = (a, b)`, a pair with
`a < b` of which at least one is finite, such as `(0, Inf)`, `(-Inf, 0)` or `(0, 1)`,
with the boundary correction `method`: [`Reflection`](@ref), [`BetaKernels`](@ref),
[`GammaKernels`](@ref) or [`Transformed`](@ref). The observations must lie in the
support. `kernel` is the kernel of reflection and transformation estimates; beta and
gamma kernels have their own.

`bandwidth` is a positive number or a rule. By default it is
[`SheatherJones`](@ref) applied to the observations and their first mirror images
for reflection and to the transformed observations for transformation estimates.
For beta and gamma kernels it is the smoothing parameter, chosen by default by
least-squares cross-validation, `∫ f̃² - 2 ∑ᵢ wᵢ f̃₋ᵢ(uᵢ)/W`, at the global minimum
over the `gridsize` values of an [`LSCV`](@ref) rule (by default 30), spaced evenly on
the log scale from 1/30 to 5 times the parameter whose kernel has the variance of a
normal-reference bandwidth, with a warning when the minimum lies at an end.
"""
function fit(::Type{BoundedKDE}, x::AbstractVector; weights = nothing, kwargs...)
  return fit(BoundedKDE, SmoothingData(x; weights); kwargs...)
end

function fit(::Type{BoundedKDE}, d::SmoothingData; support, method::BoundaryMethod = Reflection(),
             kernel::Union{Nothing,AbstractKernel} = nothing, bandwidth = nothing)
  _noclusters(d, "BoundedKDE")
  d = _withouty(d)
  a, b = _checksupport(support, d)
  return _boundedfit(method, d, a, b, kernel, bandwidth)
end

# the weights in the order of the sorted observations (stored scaled by 2^-wexp,
# which cancels in all uses below)
_weightvector(d::SmoothingData) = d.w === nothing ? ones(nobs(d)) : copy(d.w)

function _checksupport(support, d)
  (support isa Tuple || support isa AbstractVector) && length(support) == 2 && all(v -> v isa Real, support) ||
    throw(ArgumentError("support must be a pair (a, b), such as (0, Inf), got $(repr(support))"))
  a, b = Float64(support[1]), Float64(support[2])
  !isnan(a) && !isnan(b) && a < b || throw(ArgumentError("the support must have a < b, got ($a, $b)"))
  isfinite(a) || isfinite(b) || throw(ArgumentError("the support has no finite boundary; use fit(KDE, x)"))
  a <= d.x[1] && d.x[end] <= b ||
    throw(ArgumentError("the observations must lie in the support ($a, $b); they range from $(d.x[1]) to $(d.x[end])"))
  return a, b
end

function Base.show(io::IO, f::BoundedKDE)
  print(io, "BoundedKDE(", f.method, ", support = (", f.lower, ", ", f.upper, "), bandwidth = ",
        round(f.h; sigdigits = 4), ", ", nobs(f.data), " observations", f.data.w === nothing ? "" : ", weighted", ")")
end

Base.show(io::IO, ::Reflection) = print(io, "Reflection()")
Base.show(io::IO, m::Union{BetaKernels,GammaKernels}) = print(io, nameof(typeof(m)), m.modified ? "(modified = true)" : "()")
Base.show(io::IO, m::Transformed) = print(io, "Transformed(:", m.transform, ")")

bandwidth(f::BoundedKDE) = f.h
nobs(f::BoundedKDE) = nobs(f.data)
insupport(f::BoundedKDE, x::Real) = minimum(f) <= x <= maximum(f)

# a draw of the observation by weight
_drawindex(rng, d::SmoothingData) =
  d.w === nothing ? rand(rng, 1:nobs(d)) : min(searchsortedlast(d.cumw, rand(rng) * d.sumw), nobs(d))

function _checkprobability(p)
  0 <= p <= 1 || throw(DomainError(p, "a probability must be between 0 and 1"))
end

## Reflection

# The estimate is mass times the kernel estimate `inner` of the observations and
# their images, restricted to the support
struct _ReflectionImpl{K<:KDE}
  inner::K
  mass::Float64
end

function _issymmetric(k::AbstractKernel)
  k isa Union{NamedPolynomialKernel,GaussianKernel} && return true
  lo, hi = support(k)
  lo == -hi || return false
  s = isfinite(hi) ? hi : 4.0
  return all(u -> isapprox(k(u), k(-u); rtol = 1e-12, atol = 1e-14), range(0, s; length = 41))
end

# the observations and their mirror images within distance R of [a, b], with their
# weights
function _images(d::SmoothingData, a, b, R)
  xs = Float64[]
  ws = Float64[]
  push_image(z, i) = (push!(xs, z); push!(ws, _weight(d.w, i)))
  if !isfinite(b)   # one boundary below
    for i in 1:nobs(d)
      push_image(d.x[i], i)
      push_image(2a - d.x[i], i)
    end
  elseif !isfinite(a)   # one boundary above
    for i in 1:nobs(d)
      push_image(d.x[i], i)
      push_image(2b - d.x[i], i)
    end
  else
    L = b - a
    M = ceil(Int, R / (2L)) + 1
    M <= 500 || throw(ArgumentError("the bandwidth is hundreds of times the width of the support, which needs " *
      "too many images; use a smaller bandwidth"))
    for i in 1:nobs(d), m in -M:M, z in (d.x[i] + 2m * L, 2a - d.x[i] + 2m * L)
      a - R <= z <= b + R && push_image(z, i)
    end
  end
  return xs, ws
end

function _boundedfit(method::Reflection, d::SmoothingData, a, b, kernel, bw)
  kernel === nothing && (kernel = EpanechnikovKernel())
  _isnonnegative(kernel) && isapprox(moment(kernel, 0), 1; rtol = 1e-10) ||
    throw(ArgumentError("$kernel must be a probability density"))
  _issymmetric(kernel) || throw(ArgumentError("reflection needs a symmetric kernel, got $kernel"))
  bw === nothing && (bw = SheatherJones())
  sel = if bw isa BandwidthSelector
    # the observations and their first images: their kernel estimate is the reflected
    # density, whose bandwidth is that of the density on the support, the factors of
    # the images canceling
    xs = vcat(d.x, isfinite(a) ? 2a .- d.x : Float64[], isfinite(b) ? 2b .- d.x : Float64[])
    ws = repeat(_weightvector(d), length(xs) ÷ nobs(d))
    _selection(bw, SmoothingData(xs; weights = d.w === nothing ? nothing : ws), kernel)
  else
    _selection(bw, d, kernel)
  end
  h = sel.h
  h isa Float64 || throw(ArgumentError("a density estimate needs one bandwidth"))
  # images farther than the reach of the kernel do not contribute; for the Gaussian
  # kernel, beyond 39 standard deviations its density underflows
  hi = last(support(kernel))
  reach = isfinite(hi) ? hi : 39 * sqrt(moment(kernel, 2))
  isfinite(reach) || throw(ArgumentError("reflection needs a kernel with a finite variance, got $kernel"))
  xs, ws = _images(d, a, b, h * reach)
  inner = fit(KDE, SmoothingData(xs; weights = ws); kernel, bandwidth = h)
  return BoundedKDE(Reflection(), a, b, d, h, _ReflectionImpl(inner, sum(ws) / d.sumw), sel)
end

Base.minimum(f::BoundedKDE{Reflection}) = max(f.lower, minimum(f.impl.inner))
Base.maximum(f::BoundedKDE{Reflection}) = min(f.upper, maximum(f.impl.inner))

function pdf(f::BoundedKDE{Reflection}, x::Real)
  isnan(x) && return float(x)
  f.lower <= x <= f.upper || return zero(float(x))
  return f.impl.mass * pdf(f.impl.inner, x)
end

# the logarithm from that of the inner estimate, which stays finite in the tails of
# a Gaussian kernel
function logpdf(f::BoundedKDE{Reflection}, x::Real)
  isnan(x) && return float(x)
  f.lower <= x <= f.upper || return -Inf
  return log(f.impl.mass) + logpdf(f.impl.inner, x)
end

function cdf(f::BoundedKDE{Reflection}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 0.0
  x >= f.upper && return 1.0
  g = f.impl.inner
  isfinite(f.lower) && return clamp(f.impl.mass * (cdf(g, x) - cdf(g, f.lower)), 0.0, 1.0)
  return clamp(f.impl.mass * cdf(g, x), 0.0, 1.0)
end

function ccdf(f::BoundedKDE{Reflection}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 1.0
  x >= f.upper && return 0.0
  g = f.impl.inner
  isfinite(f.upper) && return clamp(f.impl.mass * (ccdf(g, x) - ccdf(g, f.upper)), 0.0, 1.0)
  return clamp(f.impl.mass * ccdf(g, x), 0.0, 1.0)
end

# F̂(x) = mass (G(x) - G(a)) for the distribution function G of the inner estimate,
# which inverts with the quantiles of G
function quantile(f::BoundedKDE{Reflection}, p::Real)
  _checkprobability(p)
  p == 0 && return minimum(f)
  p == 1 && return maximum(f)
  g, m = f.impl.inner, f.impl.mass
  q = if isfinite(f.lower)
    quantile(g, min(cdf(g, f.lower) + p / m, 1.0))
  else
    # one boundary above, where 1 - G(b) = 1 - 1/m: from the upper tail for accuracy
    cquantile(g, min((1 - p) / m + ccdf(g, f.upper), 1.0))
  end
  return clamp(q, minimum(f), maximum(f))
end

# an observation drawn by weight plus h times a draw from the kernel, folded into the
# support
function Base.rand(rng::AbstractRNG, f::BoundedKDE{Reflection})
  d, a, b = f.data, f.lower, f.upper
  z = d.x[_drawindex(rng, d)] + f.h * _randkernel(rng, f.impl.inner.kernel)
  !isfinite(b) && return z < a ? 2a - z : z
  !isfinite(a) && return z > b ? 2b - z : z
  L = b - a
  t = mod(z - a, 2L)
  return a + (t > L ? 2L - t : t)
end

## Transformation

struct _TransformImpl{K<:KDE}
  inner::K
  transform::Symbol
end

# t(x) for the support (a, b): log(x - a), -log(b - x), the logit as a difference of
# logarithms, which never overflows inside the support, or the probit, from the
# nearer end
function _forward(transform::Symbol, a, b, x)
  transform === :log && return isfinite(a) ? log(x - a) : -log(b - x)
  transform === :logit && return log(x - a) - log(b - x)
  u = (x - a) / (b - a)
  return u <= 0.5 ? quantile(Distributions.Normal(), u) : -quantile(Distributions.Normal(), (b - x) / (b - a))
end

function _backward(transform::Symbol, a, b, t)
  transform === :log && return isfinite(a) ? a + exp(t) : b - exp(-t)
  if transform === :logit
    return t <= 0 ? a + (b - a) * exp(t) / (1 + exp(t)) : b - (b - a) / (1 + exp(t))
  end
  return t <= 0 ? a + (b - a) * cdf(Distributions.Normal(), t) : b - (b - a) * cdf(Distributions.Normal(), -t)
end

function _logjacobian(transform::Symbol, a, b, x)
  transform === :log && return isfinite(a) ? -log(x - a) : -log(b - x)
  transform === :logit && return log(b - a) - log(x - a) - log(b - x)
  return -logpdf(Distributions.Normal(), _forward(transform, a, b, x)) - log(b - a)
end

function _boundedfit(method::Transformed, d::SmoothingData, a, b, kernel, bw)
  kernel === nothing && (kernel = EpanechnikovKernel())
  tr = method.transform
  two = isfinite(a) && isfinite(b)
  tr === :auto && (tr = two ? :logit : :log)
  two && tr === :log && throw(ArgumentError("a support with two finite boundaries needs transform = :logit or :probit"))
  !two && tr !== :log && throw(ArgumentError("a support with one finite boundary needs transform = :log"))
  a < d.x[1] && d.x[end] < b || throw(ArgumentError("a transformation estimate needs the observations " *
    "strictly inside the support; widen it or use Reflection()"))
  t = [_forward(tr, a, b, v) for v in d.x]
  all(isfinite, t) || throw(ArgumentError("the transformed observations are not finite"))
  bw === nothing && (bw = SheatherJones())
  inner = fit(KDE, t; kernel, bandwidth = bw, weights = d.w === nothing ? nothing : _weightvector(d))
  return BoundedKDE(Transformed(tr), a, b, d, inner.h, _TransformImpl(inner, tr), inner.selection)
end

Base.minimum(f::BoundedKDE{Transformed}) = max(f.lower, _backward(f.impl.transform, f.lower, f.upper, minimum(f.impl.inner)))
Base.maximum(f::BoundedKDE{Transformed}) = min(f.upper, _backward(f.impl.transform, f.lower, f.upper, maximum(f.impl.inner)))

function logpdf(f::BoundedKDE{Transformed}, x::Real)
  isnan(x) && return float(x)
  f.lower < x < f.upper || return -Inf
  m = f.impl
  return logpdf(m.inner, _forward(m.transform, f.lower, f.upper, x)) + _logjacobian(m.transform, f.lower, f.upper, x)
end

function pdf(f::BoundedKDE{Transformed}, x::Real)
  isnan(x) && return float(x)
  f.lower < x < f.upper || return zero(float(x))
  m = f.impl
  v = pdf(m.inner, _forward(m.transform, f.lower, f.upper, x))
  return v == 0 ? zero(v) : exp(log(v) + _logjacobian(m.transform, f.lower, f.upper, x))
end

function cdf(f::BoundedKDE{Transformed}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 0.0
  x >= f.upper && return 1.0
  return cdf(f.impl.inner, _forward(f.impl.transform, f.lower, f.upper, x))
end

function ccdf(f::BoundedKDE{Transformed}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 1.0
  x >= f.upper && return 0.0
  return ccdf(f.impl.inner, _forward(f.impl.transform, f.lower, f.upper, x))
end

function quantile(f::BoundedKDE{Transformed}, p::Real)
  _checkprobability(p)
  p == 0 && return minimum(f)
  p == 1 && return maximum(f)
  m = f.impl
  return clamp(_backward(m.transform, f.lower, f.upper, quantile(m.inner, p)), f.lower, f.upper)
end

Base.rand(rng::AbstractRNG, f::BoundedKDE{Transformed}) =
  clamp(_backward(f.impl.transform, f.lower, f.upper, rand(rng, f.impl.inner)), f.lower, f.upper)

## Beta and gamma kernels

# The estimate in the variable v of the kernels, v = u/σ with u = (x - a)/(b - a) for
# beta kernels (σ = 1) and u = x - a (or b - x) for gamma kernels, σ their weighted
# mean, so that the computations do not depend on the scale of the data; s is the
# smoothing parameter in v. The panels [e₀, e₁], …, of the quadrature cover where the
# estimate is not negligible, and cum and rcum hold the integrals of f̃ / c to the
# left of each edge and to its right.
struct _AsymmetricImpl{M<:Union{BetaKernels,GammaKernels}}
  method::M
  v::Vector{Float64}         # the observations in v, increasing
  logv::Vector{Float64}
  log1mv::Vector{Float64}    # log(1 - v) for beta kernels
  w::Vector{Float64}         # the weights in the same order, summing to one
  s::Float64
  σ::Float64                 # u = σ v
  scale::Float64             # dx/du
  flip::Bool                 # u = b - x
  edges::Vector{Float64}
  cum::Vector{Float64}
  rcum::Vector{Float64}
  c::Float64                 # the integral of f̃
end

const _GL = _gausslegendre(12)

# ∫ g over [lo, hi] by Gauss–Legendre
function _glintegral(g, lo, hi)
  x, w = _GL
  c, r = (lo + hi) / 2, (hi - lo) / 2
  return r * sum(w[k] * g(c + r * x[k]) for k in eachindex(x))
end

# the parameters of the kernel at v
function _kernelparams(m::BetaKernels, s, v)
  if !m.modified
    return v / s + 1, (1 - v) / s + 1
  elseif v < 2s
    return _chenrho(s, v), (1 - v) / s
  elseif v > 1 - 2s
    return v / s, _chenrho(s, 1 - v)
  end
  return v / s, (1 - v) / s
end

# ρ(v) = 2s² + 2.5 - √(4s⁴ + 6s² + 2.25 - v² - v/s) without cancellation: with
# B = 2s² + 1.5 and q = v² + v/s, ρ = 1 + q / (B + √(B² - q)), which is 1 at v = 0
function _chenrho(s, v)
  B = 2s^2 + 1.5
  q = v^2 + v / s
  return 1 + q / (B + sqrt(B^2 - q))
end

_kernelparams(m::GammaKernels, s, v) = !m.modified ? v / s + 1 : v >= 2s ? v / s : (v / (2s))^2 + 1

# the spread of the kernel at v, about its standard deviation, which sets the window
# of the sums and the widths of the panels
_kernelspread(::BetaKernels, s, v) = sqrt(s * clamp(v, 0, 1) * (1 - clamp(v, 0, 1))) + s
_kernelspread(::GammaKernels, s, v) = sqrt(s * max(v, 0.0)) + s

# a log y with 0 log 0 = 0
_xlogy(a, logy) = a == 0 ? zero(a) : a * logy

# the observations whose kernel at v can matter: within 40 spreads of v
function _asymmetricwindow(impl::_AsymmetricImpl, v)
  r = 40 * _kernelspread(impl.method, impl.s, v)
  return searchsortedfirst(impl.v, v - r):searchsortedlast(impl.v, v + r)
end

# the kernel of observation i at v, and its logarithmic normalization at v
function _kernellognorm(m::BetaKernels, s, v)
  p, q = _kernelparams(m, s, v)
  return p, q, SpecialFunctions.logbeta(p, q)
end

function _kernellognorm(m::GammaKernels, s, v)
  κ = _kernelparams(m, s, v)
  return κ, 0.0, κ * log(s) + SpecialFunctions.loggamma(κ)
end

@inline _logkernel(::BetaKernels, impl, i, p, q, ln) = _xlogy(p - 1, impl.logv[i]) + _xlogy(q - 1, impl.log1mv[i]) - ln
@inline _logkernel(::GammaKernels, impl, i, κ, _, ln) = _xlogy(κ - 1, impl.logv[i]) - impl.v[i] / impl.s - ln

# f̃(v) = ∑ wᵢ Kᵥ(vᵢ), over the window of v, leaving out observation `skip`
function _asymmetricsum(impl::_AsymmetricImpl, v; skip = 0)
  m = impl.method
  p, q, ln = _kernellognorm(m, impl.s, v)
  t = 0.0
  @inbounds for i in _asymmetricwindow(impl, v)
    i == skip && continue
    t += impl.w[i] * exp(_logkernel(m, impl, i, p, q, ln))
  end
  return t
end

# the edges of panels about a quarter of the spread of the kernel wide, over the
# range of v where the estimate is not negligible, with edges at the kinks of the
# modified kernels, where their parameters change formula
function _asymmetricpanels(m, v, s; maxpanels = 50_000)
  lo = max(0.0, v[1] - 40 * _kernelspread(m, s, v[1]))
  hi = v[end] + 40 * _kernelspread(m, s, v[end])
  m isa BetaKernels && (hi = min(hi, 1.0))
  kinks = !m.modified ? Float64[] : m isa BetaKernels ? [2s, 1 - 2s] : [2s]
  kinks = sort(filter(k -> lo < k < hi, kinks))
  edges = [lo]
  while edges[end] < hi
    length(edges) > maxpanels && return nothing
    e = edges[end]
    next = min(hi, e + _kernelspread(m, s, e) / 4)
    k = findfirst(t -> e < t < next, kinks)
    push!(edges, k === nothing ? next : kinks[k])
  end
  return edges
end

function _asymmetricimpl(method, v, w, s, σ, scale, flip; panels = _asymmetricpanels(method, v, s))
  panels === nothing && return nothing
  pre = _AsymmetricImpl(method, v, log.(v), method isa BetaKernels ? log1p.(-v) : Float64[], w, s, σ, scale, flip,
                        Float64[], Float64[], Float64[], 1.0)
  pieces = [_glintegral(t -> _asymmetricsum(pre, t), panels[k], panels[k + 1]) for k in 1:length(panels)-1]
  c = sum(pieces)
  c > 0 || return nothing
  cum = vcat(0.0, cumsum(pieces) ./ c)
  rcum = vcat(reverse(cumsum(reverse(pieces))) ./ c, 0.0)
  return _AsymmetricImpl(method, v, pre.logv, pre.log1mv, w, s, σ, scale, flip, panels, cum, rcum, c)
end

function _boundedfit(method::Union{BetaKernels,GammaKernels}, d::SmoothingData, a, b, kernel, bw)
  isbeta = method isa BetaKernels
  kernel === nothing || throw(ArgumentError("$method uses its own kernels; leave out the keyword kernel"))
  if isbeta
    isfinite(a) && isfinite(b) || throw(ArgumentError("beta kernels need a support with two finite boundaries"))
  else
    isfinite(a) && isfinite(b) && throw(ArgumentError("gamma kernels need a support with one finite boundary; " *
      "use BetaKernels() for two"))
  end
  flip = !isbeta && !isfinite(a)
  scale = isbeta ? b - a : 1.0
  u = isbeta ? (d.x .- a) ./ scale : flip ? b .- d.x : d.x .- a
  w = _weightvector(d)
  if flip
    u, w = reverse(u), reverse(w)
  end
  w ./= sum(w)
  (u[1] > 0 && (!isbeta || u[end] < 1)) || throw(ArgumentError("$method gives no weight to observations at a " *
    "boundary, which would be lost; shift the support slightly or use Reflection()"))
  σ = isbeta ? 1.0 : sum(w .* u)
  v = u ./ σ
  s, sel = _asymmetricbandwidth(method, v, w, d, bw, σ)
  impl = _asymmetricimpl(method, v, w, s, σ, scale, flip)
  impl === nothing && throw(ArgumentError("the smoothing parameter $(s * σ) is too small for the spread of the data"))
  return BoundedKDE(method, a, b, d, s * σ, impl, sel)
end

# the smoothing parameter in v: a number (given in u), or least-squares
# cross-validation over a grid around the parameter whose kernel has the variance of
# a normal-reference bandwidth
function _asymmetricbandwidth(method, v, w, d, bw, σ)
  maxs = method isa BetaKernels && method.modified ? 0.25 : Inf
  if bw isa Real
    s = _bandwidths(bw, 1) / σ
    s < maxs || throw(ArgumentError("the modified beta kernels need a smoothing parameter below 1/4, got $bw"))
    return s, BandwidthSelection(Float64(bw))
  end
  rule = bw === nothing ? LSCV(gridsize = 30) : bw
  rule isa LSCV || throw(ArgumentError("beta and gamma kernels take a positive number or LSCV(), got $(repr(rule))"))
  nobs(d) >= 2 || throw(ArgumentError("cross-validation needs at least two observations"))
  h = bandwidth(NormalReference(), SmoothingData(v; weights = w), GaussianKernel())
  # the variance of the kernel at v is about s v (1 - v) for beta and s v for gamma kernels
  spread = method isa BetaKernels ? sum(w .* v .* (1 .- v)) : sum(w .* v)
  s0 = h^2 / spread
  method isa BetaKernels && (s0 = min(s0, 0.05))
  grid = exp.(range(log(s0 / 30), log(s0 * 5); length = rule.gridsize))
  grid = filter(<(min(maxs * 0.96, method isa BetaKernels ? 1.0 : Inf)), grid)
  length(grid) >= 3 || throw(ArgumentError("too few smoothing parameters to search; increase gridsize"))
  logv, log1mv = log.(v), method isa BetaKernels ? log1p.(-v) : Float64[]
  s, details = _gridminimize(t -> _asymmetriclscv(method, v, logv, log1mv, w, t), grid; largest = false, rtol = 1e-3)
  details.atbound && @warn "the cross-validation criterion of $method is lowest at an end of its grid, s = $(s * σ)"
  return s, BandwidthSelection(s * σ, rule, merge(details, (grid = details.grid .* σ,)))
end

# ∫ f̃² - 2 ∑ᵢ wᵢ f̃₋ᵢ(vᵢ), the cross-validation criterion of the unnormalized estimate,
# with f̃₋ᵢ = ∑_{j≠i} wⱼ Kᵥ(vⱼ) / (1 - wᵢ); Inf where the panels would be too many
function _asymmetriclscv(method, v, logv, log1mv, w, s)
  panels = _asymmetricpanels(method, v, s)
  panels === nothing && return Inf
  impl = _AsymmetricImpl(method, v, logv, log1mv, w, s, 1.0, 1.0, false, Float64[], Float64[], Float64[], 1.0)
  sq = sum(_glintegral(t -> _asymmetricsum(impl, t)^2, panels[k], panels[k + 1]) for k in 1:length(panels)-1)
  loo = sum(w[i] * _asymmetricsum(impl, v[i]; skip = i) / (1 - w[i]) for i in eachindex(v))
  return sq - 2 * loo
end

# u (in the variable of the kernels, before the scaling by σ) of x
_tov(f::BoundedKDE, x) = f.impl.method isa BetaKernels ? (x - f.lower) / f.impl.scale / f.impl.σ :
  (f.impl.flip ? f.upper - x : x - f.lower) / f.impl.σ

Base.minimum(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}) = f.lower
Base.maximum(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}) = f.upper

function pdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real)
  isnan(x) && return float(x)
  f.lower <= x <= f.upper && isfinite(x) || return zero(float(x))
  impl = f.impl
  return _asymmetricsum(impl, _tov(f, x)) / (impl.c * impl.scale * impl.σ)
end

logpdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real) = log(pdf(f, x))

# the integrals of the estimate in v from the left end to v, and from v to the right
function _vcdf(impl::_AsymmetricImpl, v)
  e = impl.edges
  v <= e[1] && return 0.0
  v >= e[end] && return 1.0
  k = searchsortedlast(e, v)
  return clamp(impl.cum[k] + _glintegral(t -> _asymmetricsum(impl, t), e[k], v) / impl.c, 0.0, 1.0)
end

function _vccdf(impl::_AsymmetricImpl, v)
  e = impl.edges
  v <= e[1] && return 1.0
  v >= e[end] && return 0.0
  k = searchsortedlast(e, v)
  return clamp(impl.rcum[k + 1] + _glintegral(t -> _asymmetricsum(impl, t), v, e[k + 1]) / impl.c, 0.0, 1.0)
end

function cdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 0.0
  x >= f.upper && return 1.0
  v = _tov(f, x)
  return f.impl.flip ? _vccdf(f.impl, v) : _vcdf(f.impl, v)
end

function ccdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 1.0
  x >= f.upper && return 0.0
  v = _tov(f, x)
  return f.impl.flip ? _vcdf(f.impl, v) : _vccdf(f.impl, v)
end

# the quantile of the estimate in v, inverted within the panel that holds it
function _vquantile(impl::_AsymmetricImpl, p)
  e, c = impl.edges, impl.cum
  k = clamp(searchsortedlast(c, p), 1, length(e) - 1)
  return _invertcdf(t -> _vcdf(impl, t), t -> _asymmetricsum(impl, t) / impl.c, p, e[k], e[k + 1])
end

function quantile(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, p::Real)
  _checkprobability(p)
  p == 0 && return minimum(f)
  p == 1 && return maximum(f)
  impl = f.impl
  v = _vquantile(impl, impl.flip ? 1 - p : p)
  u = impl.σ * v
  x = impl.method isa BetaKernels ? f.lower + impl.scale * u : impl.flip ? f.upper - u : f.lower + u
  return clamp(x, f.lower, f.upper)
end

Base.rand(rng::AbstractRNG, f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}) = quantile(f, rand(rng))

## Moments

# ∫ g over [lo, hi] by Gauss–Legendre on halves of the interval, halved again where
# the two levels differ by more than rtol of the total, as at the kinks of estimates
# with compactly supported kernels
function _adaptiveintegral(g, lo, hi, whole = _glintegral(g, lo, hi); rtol = 1e-12, total = abs(whole), depth = 0)
  mid = (lo + hi) / 2
  left, right = _glintegral(g, lo, mid), _glintegral(g, mid, hi)
  if abs(left + right - whole) <= rtol * max(total, floatmin()) || depth >= 30 || !(lo < mid < hi)
    return left + right
  end
  return _adaptiveintegral(g, lo, mid, left; rtol, total, depth = depth + 1) +
         _adaptiveintegral(g, mid, hi, right; rtol, total, depth = depth + 1)
end

# the moments ∫ g(x) f̂(x) dx over 64 panels of equal probability between the
# quantiles of order 1e-14 and 1 - 1e-14 (not the ends of the support, which can lie
# far from the mass), each integrated adaptively
function _boundedmoment(f::BoundedKDE, g)
  probs = vcat(1e-14, (1:63) ./ 64, 1 - 1e-14)
  qs = quantile.(Ref(f), probs)
  h(t) = g(t) * pdf(f, t)
  pieces = [_glintegral(h, qs[k], qs[k + 1]) for k in 1:length(qs)-1]
  total = sum(abs, pieces)
  return sum(_adaptiveintegral(h, qs[k], qs[k + 1], pieces[k]; total) for k in eachindex(pieces) if qs[k] < qs[k + 1])
end

mean(f::BoundedKDE) = _boundedmoment(f, identity)
var(f::BoundedKDE) = (μ = mean(f); _boundedmoment(f, t -> (t - μ)^2))
