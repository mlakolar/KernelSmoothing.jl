## Kernel density estimation on a bounded support

"""
    BoundaryMethod

Supertype of the boundary corrections of a [`BoundedKDE`](@ref):
[`Reflection`](@ref), [`BetaKernels`](@ref), [`GammaKernels`](@ref) and
[`Transformation`](@ref).
"""
abstract type BoundaryMethod end

"""
    Reflection()

Reflection at the boundaries (Schuster 1985). With a lower boundary `a`, the
estimate is

    f̂(x) = ∑ᵢ wᵢ (k((x - xᵢ)/h) + k((x + xᵢ - 2a)/h)) / (h W),   x ≥ a,

the kernel estimate of the observations and their mirror images, restricted to the
support; with two boundaries `a < b`, the images are reflected at both, repeatedly,
so that `f̂` is the density of `xᵢ + h ε` folded into `[a, b]`, for an observation
`xᵢ` drawn by weight and `ε` from the kernel. The estimate integrates to one and is
computed exactly, with its distribution function. Its derivative vanishes at a
boundary, so that it has a bias of order `h` there unless the density's does too.
The kernel must be symmetric; a bandwidth rule is applied to the observations and
their first images, which targets the bandwidth of the density on the support.
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
u/s)` for `u < 2s` and the second by `ρ(1 - u)` for `u > 1 - 2s`. Neither integrates
to one; the estimate is `f̃` divided by its integral, computed by quadrature, which
also gives the distribution function.
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
below. As for [`BetaKernels`](@ref), the estimate is `f̃` divided by its integral. A
support `(-∞, b]` is handled by reflecting the data at `b`.
"""
struct GammaKernels <: BoundaryMethod
  modified::Bool
end

GammaKernels(; modified::Bool = false) = GammaKernels(modified)

"""
    Transformation(transform = :auto)

The transformation estimator (Wand, Marron and Ruppert 1991; Marron and Ruppert
1994): the kernel density estimate `ĝ` of `t(xᵢ)` for a map `t` of the support onto
the real line, transformed back,

    f̂(x) = ĝ(t(x)) |t'(x)|,   F̂(x) = Ĝ(t(x)),

with `t(x) = log(x - a)` for a support `[a, ∞)` (`transform = :log`), `log(b - x)` for
`(-∞, b]`, and `t(x) = log((x - a)/(b - x))` (`:logit`) or `Φ⁻¹((x - a)/(b - a))`
(`:probit`) for `[a, b]`; `:auto` takes `:log` or `:logit`. The estimate is a
distribution exactly, its quantiles and draws come from those of `ĝ`, and the
bandwidth, chosen on the transformed data, varies with `x` on the original scale.
It suits densities that are infinite or steep at a boundary. The observations must
lie strictly inside the support.
"""
struct Transformation <: BoundaryMethod
  transform::Symbol

  function Transformation(transform::Symbol)
    transform in (:auto, :log, :logit, :probit) ||
      throw(ArgumentError("transform must be :auto, :log, :logit or :probit, got :$transform"))
    new(transform)
  end
end

Transformation(; transform::Symbol = :auto) = Transformation(transform)

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

The kernel density estimate of the sample `x` on the `support = (a, b)`, a pair with
`a < b` of which at least one is finite, such as `(0, Inf)` or `(0, 1)`, with the
boundary correction `method`: [`Reflection`](@ref), [`BetaKernels`](@ref),
[`GammaKernels`](@ref) or [`Transformation`](@ref). The observations must lie in the
support. `kernel` is the kernel of reflection and transformation estimates.

`bandwidth` is a positive number or a rule. By default it is
[`SheatherJones`](@ref) applied to the observations and their first mirror images
for reflection and to the transformed observations for transformation estimates;
for beta and gamma kernels it is the smoothing parameter, chosen by least-squares
cross-validation by default (`LSCV()`, whose `gridsize` sets the number of values
tried).
"""
function fit(::Type{BoundedKDE}, x::AbstractVector; support, method::BoundaryMethod = Reflection(),
             kernel::AbstractKernel = EpanechnikovKernel(), bandwidth = nothing, weights = nothing)
  d = SmoothingData(x; weights)
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
Base.show(io::IO, m::Transformation) = print(io, "Transformation(:", m.transform, ")")

bandwidth(f::BoundedKDE) = f.h
nobs(f::BoundedKDE) = nobs(f.data)
Base.minimum(f::BoundedKDE) = f.lower
Base.maximum(f::BoundedKDE) = f.upper
insupport(f::BoundedKDE, x::Real) = f.lower <= x <= f.upper

# a draw of the observation by weight
_drawindex(rng, d::SmoothingData) =
  d.w === nothing ? rand(rng, 1:nobs(d)) : min(searchsortedlast(d.cumw, rand(rng) * d.sumw), nobs(d))

logpdf(f::BoundedKDE, x::Real) = log(pdf(f, x))
ccdf(f::BoundedKDE, x::Real) = 1 - cdf(f, x)

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
    for i in 1:nobs(d), m in -M:M, z in (d.x[i] + 2m * L, 2a - d.x[i] + 2m * L)
      a - R <= z <= b + R && push_image(z, i)
    end
  end
  return xs, ws
end

function _boundedfit(method::Reflection, d::SmoothingData, a, b, kernel, bw)
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
  R = h * (isfinite(hi) ? hi : 39 * sqrt(moment(kernel, 2)))
  xs, ws = _images(d, a, b, R)
  inner = fit(KDE, SmoothingData(xs; weights = ws); kernel, bandwidth = h)
  return BoundedKDE(Reflection(), a, b, d, h, _ReflectionImpl(inner, sum(ws) / d.sumw), sel)
end


function pdf(f::BoundedKDE{Reflection}, x::Real)
  isnan(x) && return float(x)
  insupport(f, x) || return zero(float(x))
  return f.impl.mass * pdf(f.impl.inner, x)
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
  0 <= p <= 1 || throw(DomainError(p, "a probability must be between 0 and 1"))
  g, m = f.impl.inner, f.impl.mass
  q = if isfinite(f.lower)
    quantile(g, min(cdf(g, f.lower) + p / m, 1.0))
  else
    # one boundary above, G(b) = 1/m: from the upper tail for accuracy
    cquantile(g, min(1 - p, 1.0) / m + ccdf(g, f.upper))
  end
  return clamp(q, max(f.lower, minimum(g)), min(f.upper, maximum(g)))
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
  decreasing::Bool   # log(b - x) for a support (-∞, b]
end

# t(x), the inverse and log |t'(x)| for the support (a, b)
function _forward(m::_TransformImpl, a, b, x)
  m.transform === :log && return m.decreasing ? log(b - x) : log(x - a)
  u = (x - a) / (b - a)
  return m.transform === :logit ? log(u) - log1p(-u) : quantile(Distributions.Normal(), u)
end

function _backward(m::_TransformImpl, a, b, t)
  m.transform === :log && return m.decreasing ? b - exp(t) : a + exp(t)
  u = m.transform === :logit ? 1 / (1 + exp(-t)) : cdf(Distributions.Normal(), t)
  return a + (b - a) * u
end

function _logjacobian(m::_TransformImpl, a, b, x)
  m.transform === :log && return m.decreasing ? -log(b - x) : -log(x - a)
  u = (x - a) / (b - a)
  m.transform === :logit && return -log(u) - log1p(-u) - log(b - a)
  t = quantile(Distributions.Normal(), u)
  return -logpdf(Distributions.Normal(), t) - log(b - a)
end

function _boundedfit(method::Transformation, d::SmoothingData, a, b, kernel, bw)
  tr = method.transform
  two = isfinite(a) && isfinite(b)
  tr === :auto && (tr = two ? :logit : :log)
  two && tr === :log && throw(ArgumentError("a support with two finite boundaries needs transform = :logit or :probit"))
  !two && tr !== :log && throw(ArgumentError("a support with one finite boundary needs transform = :log"))
  a < d.x[1] && d.x[end] < b || throw(ArgumentError("a transformation estimate needs the observations " *
    "strictly inside the support; widen it or use Reflection()"))
  m = _TransformImpl(fit(KDE, [0.0, 1.0]; bandwidth = 1.0), tr, !isfinite(a))
  t = [_forward(m, a, b, v) for v in d.x]
  all(isfinite, t) || throw(ArgumentError("the transformed observations are not finite; the observations " *
    "are too close to a boundary"))
  bw === nothing && (bw = SheatherJones())
  inner = fit(KDE, t; kernel, bandwidth = bw, weights = d.w === nothing ? nothing : _weightvector(d))
  impl = _TransformImpl(inner, tr, !isfinite(a))
  return BoundedKDE(Transformation(tr), a, b, d, inner.h, impl, inner.selection)
end

function pdf(f::BoundedKDE{Transformation}, x::Real)
  isnan(x) && return float(x)
  f.lower < x < f.upper || return zero(float(x))
  m = f.impl
  t = _forward(m, f.lower, f.upper, x)
  isfinite(t) || return zero(float(x))
  v = pdf(m.inner, t)
  return v == 0 ? zero(v) : exp(log(v) + _logjacobian(m, f.lower, f.upper, x))
end

function logpdf(f::BoundedKDE{Transformation}, x::Real)
  isnan(x) && return float(x)
  f.lower < x < f.upper || return -Inf
  m = f.impl
  t = _forward(m, f.lower, f.upper, x)
  isfinite(t) || return -Inf
  return logpdf(m.inner, t) + _logjacobian(m, f.lower, f.upper, x)
end

function cdf(f::BoundedKDE{Transformation}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 0.0
  x >= f.upper && return 1.0
  m = f.impl
  t = _forward(m, f.lower, f.upper, x)
  return m.decreasing ? ccdf(m.inner, t) : cdf(m.inner, t)
end

function ccdf(f::BoundedKDE{Transformation}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 1.0
  x >= f.upper && return 0.0
  m = f.impl
  t = _forward(m, f.lower, f.upper, x)
  return m.decreasing ? cdf(m.inner, t) : ccdf(m.inner, t)
end

function quantile(f::BoundedKDE{Transformation}, p::Real)
  0 <= p <= 1 || throw(DomainError(p, "a probability must be between 0 and 1"))
  m = f.impl
  t = m.decreasing ? cquantile(m.inner, p) : quantile(m.inner, p)
  return clamp(_backward(m, f.lower, f.upper, t), f.lower, f.upper)
end

Base.rand(rng::AbstractRNG, f::BoundedKDE{Transformation}) =
  clamp(_backward(f.impl, f.lower, f.upper, rand(rng, f.impl.inner)), f.lower, f.upper)

## Beta and gamma kernels

# The estimate f̃ / c in the variable u of the kernels, u = (x - a)/(b - a) for beta
# kernels and u = x - a (or b - x) for gamma kernels, with s the smoothing parameter;
# the panels [e₀, e₁], …, of the quadrature and the integrals of f̃ / c up to their
# right ends
struct _AsymmetricImpl{M<:Union{BetaKernels,GammaKernels}}
  method::M
  u::Vector{Float64}         # the observations in the variable u
  logu::Vector{Float64}
  log1mu::Vector{Float64}    # log(1 - u) for beta kernels
  w::Vector{Float64}         # the weights divided by their sum
  s::Float64
  scale::Float64             # dx/du
  flip::Bool                 # u = b - x
  edges::Vector{Float64}
  cum::Vector{Float64}       # the integral of f̃ / c from e₀ to each edge
  c::Float64                 # the integral of f̃
end

const _GL = _gausslegendre(12)

# ∫ g over [lo, hi] by Gauss–Legendre
function _glintegral(g, lo, hi)
  x, w = _GL
  c, r = (lo + hi) / 2, (hi - lo) / 2
  return r * sum(w[k] * g(c + r * x[k]) for k in eachindex(x))
end

# the parameters of the kernel at u
function _kernelparams(m::BetaKernels, s, u)
  if !m.modified
    return u / s + 1, (1 - u) / s + 1
  elseif u < 2s
    return _chenrho(s, u), (1 - u) / s
  elseif u > 1 - 2s
    return u / s, _chenrho(s, 1 - u)
  end
  return u / s, (1 - u) / s
end

_chenrho(s, u) = 2s^2 + 2.5 - sqrt(4s^4 + 6s^2 + 2.25 - u^2 - u / s)

_kernelparams(m::GammaKernels, s, u) = !m.modified ? u / s + 1 : u >= 2s ? u / s : (u / (2s))^2 + 1

# f̃(u) = ∑ wᵢ K_u(uᵢ), with the weights summing to one
function _asymmetricsum(m::BetaKernels, impl, u)
  p, q = _kernelparams(m, impl.s, u)
  lb = SpecialFunctions.logbeta(p, q)
  t = zero(float(u))
  @inbounds for i in eachindex(impl.u)
    v = _xlogy(p - 1, impl.logu[i]) + _xlogy(q - 1, impl.log1mu[i]) - lb
    t += impl.w[i] * exp(v)
  end
  return t
end

function _asymmetricsum(m::GammaKernels, impl, u)
  κ, s = _kernelparams(m, impl.s, u), impl.s
  ln = -κ * log(s) - SpecialFunctions.loggamma(κ)
  t = zero(float(u))
  @inbounds for i in eachindex(impl.u)
    v = _xlogy(κ - 1, impl.logu[i]) - impl.u[i] / s + ln
    t += impl.w[i] * exp(v)
  end
  return t
end

# a log y with 0 log 0 = 0, so that a kernel with parameter one has a finite value at 0
_xlogy(a, logy) = a == 0 ? zero(a) : a * logy

function _boundedfit(method::Union{BetaKernels,GammaKernels}, d::SmoothingData, a, b, kernel, bw)
  isbeta = method isa BetaKernels
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
  w ./= sum(w)
  s, sel = _asymmetricbandwidth(method, u, w, d, bw)
  impl = _asymmetricimpl(method, u, w, s, scale, flip)
  return BoundedKDE(method, a, b, d, s, impl, sel)
end

function _asymmetricimpl(method, u, w, s, scale, flip)
  isbeta = method isa BetaKernels
  logu = log.(u)
  log1mu = isbeta ? log1p.(-u) : Float64[]
  pre = _AsymmetricImpl(method, u, logu, log1mu, w, s, scale, flip, Float64[], Float64[], 1.0)
  edges = _asymmetricpanels(method, u, s)
  g(t) = _asymmetricsum(method, pre, t)
  pieces = [_glintegral(g, edges[k], edges[k + 1]) for k in 1:length(edges)-1]
  c = sum(pieces)
  cum = vcat(0.0, cumsum(pieces) ./ c)
  return _AsymmetricImpl(method, u, logu, log1mu, w, s, scale, flip, edges, cum, c)
end

# panels of width s/4 on [0, 1] for beta kernels; for gamma kernels, of width
# max(s, √(s u))/4, about a quarter of the spread of the kernel at u, up to where the
# estimate is negligible
function _asymmetricpanels(::BetaKernels, u, s)
  m = clamp(ceil(Int, 4 / s), 8, 4000)
  return collect(range(0.0, 1.0; length = m + 1))
end

function _asymmetricpanels(::GammaKernels, u, s)
  umax = maximum(u)
  top = umax + 40s + 12 * sqrt(s * (umax + s))
  edges = [0.0]
  while edges[end] < top && length(edges) < 4000
    e = edges[end]
    push!(edges, min(top, e + max(s, sqrt(s * e)) / 4))
  end
  return edges
end

# the smoothing parameter: a number, or least-squares cross-validation over a grid
# around the parameter whose kernel has the variance of a normal-reference bandwidth
function _asymmetricbandwidth(method, u, w, d, bw)
  bw isa Real && return (_bandwidths(bw, 1), BandwidthSelection(bw))
  rule = bw === nothing ? LSCV(gridsize = 30) : bw
  rule isa LSCV || throw(ArgumentError("beta and gamma kernels take a positive number or LSCV(), got $(repr(rule))"))
  nobs(d) >= 2 || throw(ArgumentError("cross-validation needs at least two observations"))
  h = bandwidth(NormalReference(), SmoothingData(u; weights = d.w === nothing ? nothing : w), GaussianKernel())
  # the variance of the kernel at u is about s u (1 - u) for beta and s u for gamma kernels
  spread = method isa BetaKernels ? sum(w .* u .* (1 .- u)) : sum(w .* u)
  s0 = h^2 / max(spread, eps())
  method isa BetaKernels && (s0 = min(s0, 0.05))
  grid = exp.(range(log(s0 / 30), log(s0 * 5); length = rule.gridsize))
  method isa BetaKernels && (grid = filter(<(0.24), grid))
  s, details = _gridminimize(t -> _asymmetriclscv(method, u, w, t), grid; largest = false, rtol = 1e-3)
  return s, BandwidthSelection(s, rule, details)
end

# ∫ f̃² - 2 ∑ᵢ wᵢ f̃₋ᵢ(uᵢ), the cross-validation criterion of the unnormalized estimate
function _asymmetriclscv(method, u, w, s)
  impl = _AsymmetricImpl(method, u, log.(u), method isa BetaKernels ? log1p.(-u) : Float64[], w, s, 1.0, false,
                         Float64[], Float64[], 1.0)
  edges = _asymmetricpanels(method, u, s)
  sq = sum(_glintegral(t -> _asymmetricsum(method, impl, t)^2, edges[k], edges[k + 1]) for k in 1:length(edges)-1)
  loo = 0.0
  for i in eachindex(u)
    # f̃₋ᵢ(uᵢ) = (f̃(uᵢ) - wᵢ Kᵢ(uᵢ)) / (1 - wᵢ)
    full = _asymmetricsum(method, impl, u[i])
    loo += w[i] * (full - w[i] * _kernelvalue(method, impl, i, u[i])) / (1 - w[i])
  end
  return sq - 2 * loo
end

# the kernel of observation i at u
function _kernelvalue(m::BetaKernels, impl, i, u)
  p, q = _kernelparams(m, impl.s, u)
  return exp(_xlogy(p - 1, impl.logu[i]) + _xlogy(q - 1, impl.log1mu[i]) - SpecialFunctions.logbeta(p, q))
end

function _kernelvalue(m::GammaKernels, impl, i, u)
  κ, s = _kernelparams(m, impl.s, u), impl.s
  return exp(_xlogy(κ - 1, impl.logu[i]) - impl.u[i] / s - κ * log(s) - SpecialFunctions.loggamma(κ))
end

_tou(f::BoundedKDE, x) = f.impl.method isa BetaKernels ? (x - f.lower) / f.impl.scale :
  f.impl.flip ? f.upper - x : x - f.lower

function pdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real)
  isnan(x) && return float(x)
  insupport(f, x) || return zero(float(x))
  impl = f.impl
  return _asymmetricsum(impl.method, impl, _tou(f, x)) / (impl.c * impl.scale)
end

# the integral of the estimate in u from 0 to v
function _ucdf(impl::_AsymmetricImpl, v)
  e = impl.edges
  v <= 0 && return 0.0
  v >= e[end] && return 1.0
  k = searchsortedlast(e, v)
  part = _glintegral(t -> _asymmetricsum(impl.method, impl, t), e[k], v) / impl.c
  return clamp(impl.cum[k] + part, 0.0, 1.0)
end

function cdf(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, x::Real)
  isnan(x) && return float(x)
  x <= f.lower && return 0.0
  x >= f.upper && return 1.0
  F = _ucdf(f.impl, _tou(f, x))
  return f.impl.flip ? 1 - F : F
end

# the quantile of the estimate in u, inverted within the panel that holds it
function _uquantile(impl::_AsymmetricImpl, p)
  e, c = impl.edges, impl.cum
  k = clamp(searchsortedlast(c, p), 1, length(e) - 1)
  F(v) = _ucdf(impl, v)
  f(v) = _asymmetricsum(impl.method, impl, v) / impl.c
  return _invertcdf(F, f, p, e[k], e[k + 1])
end

function quantile(f::BoundedKDE{<:Union{BetaKernels,GammaKernels}}, p::Real)
  0 <= p <= 1 || throw(DomainError(p, "a probability must be between 0 and 1"))
  impl = f.impl
  u = _uquantile(impl, impl.flip ? 1 - p : p)
  x = impl.method isa BetaKernels ? f.lower + impl.scale * u : impl.flip ? f.upper - u : f.lower + u
  return clamp(x, f.lower, f.upper)
end

Base.rand(rng::AbstractRNG, f::BoundedKDE) = quantile(f, rand(rng))

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

# the moments ∫ g(x) f̂(x) dx over 64 panels of equal probability, between the
# quantiles of order 1e-14 and 1 - 1e-14 where the support is unbounded, each
# integrated adaptively
function _boundedmoment(f::BoundedKDE, g)
  qs = quantile.(Ref(f), vcat(isfinite(f.lower) ? 0.0 : 1e-14, (1:63) ./ 64, isfinite(f.upper) ? 1.0 : 1 - 1e-14))
  h(t) = g(t) * pdf(f, t)
  pieces = [_glintegral(h, qs[k], qs[k + 1]) for k in 1:length(qs)-1]
  total = sum(abs, pieces)
  return sum(_adaptiveintegral(h, qs[k], qs[k + 1], pieces[k]; total) for k in eachindex(pieces) if qs[k] < qs[k + 1])
end

mean(f::BoundedKDE) = _boundedmoment(f, identity)
var(f::BoundedKDE) = (μ = mean(f); _boundedmoment(f, t -> (t - μ)^2))
