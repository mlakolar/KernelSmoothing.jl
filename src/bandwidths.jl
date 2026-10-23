"""
    BandwidthSelector

Supertype of the rules that choose a bandwidth from the data, such as
[`NormalReference`](@ref) and [`SilvermanRule`](@ref). [`selectbandwidth`](@ref)
applies a rule and records what it computed on the way, and
[`bandwidth`](@ref) returns only the bandwidth. Estimators accept a rule wherever
they accept a bandwidth.

A new rule `R <: BandwidthSelector` implements
`selectbandwidth(rule::R, d::SmoothingData, k::AbstractKernel)`.
"""
abstract type BandwidthSelector end

"""
    BandwidthSelection

The result of [`selectbandwidth`](@ref): the bandwidth `h`, a number or a vector
with one bandwidth per evaluation point, the `rule` that chose it (`nothing` for a
bandwidth given by the user) and the named tuple `details` with what the rule
computed on the way, such as the scale estimate of a rule of thumb.
"""
struct BandwidthSelection{H<:Union{Float64,Vector{Float64}},R,T<:NamedTuple}
  h::H
  rule::R
  details::T
end

BandwidthSelection(h::Real) = BandwidthSelection(_bandwidths(h, 1), nothing, NamedTuple())

function Base.show(io::IO, s::BandwidthSelection)
  print(io, "BandwidthSelection(", s.h)
  s.rule === nothing || print(io, ", ", s.rule)
  print(io, ")")
end

"""
    selectbandwidth(rule::BandwidthSelector, d::SmoothingData, k::AbstractKernel)

Applies the bandwidth selector `rule` to the sample `d` for the kernel `k` and
returns a [`BandwidthSelection`](@ref). Bandwidths follow the convention of
[`AbstractKernel`](@ref): a half-width for the compactly supported kernels and a
standard deviation for the Gaussian kernel.
"""
function selectbandwidth end

"""
    bandwidth(rule::BandwidthSelector, d::SmoothingData, k::AbstractKernel; kwargs...)
    bandwidth(s::BandwidthSelection)
    bandwidth(f)

The bandwidth that `rule` chooses for the sample `d` and the kernel `k`, with the
keywords of [`selectbandwidth`](@ref) for the rule, such as `degree` for
regression; the bandwidth of a selection `s`; or the bandwidth of a fitted
estimate `f`.
"""
bandwidth(s::BandwidthSelection) = s.h
bandwidth(rule::BandwidthSelector, d::SmoothingData, k::AbstractKernel; kwargs...) =
  bandwidth(selectbandwidth(rule, d, k; kwargs...))

# a bandwidth given as a number, or chosen by a rule
_selection(h::Real, d, k) = BandwidthSelection(h)
_selection(rule::BandwidthSelector, d, k) = selectbandwidth(rule, d, k)
_selection(h, d, k) = throw(ArgumentError(
  "bandwidth must be a positive number or a BandwidthSelector, got $(repr(h))"))


## Rules of thumb

"""
    NormalReference(; robust = true)

The bandwidth that minimizes the asymptotic mean integrated squared error when
the density is normal,

    h = σ̂ (8 √π R(k) / (3 μ₂(k)² n))^(1/5),

where `R(k) = roughness(k)`, `μ₂(k) = moment(k, 2)` and `n` is the
[`effective_nobs`](@ref). The scale `σ̂` is the standard deviation or, with
`robust = true`, the smaller of the standard deviation and the interquartile
range divided by 1.349, which guards against heavy tails and outliers. For the
Gaussian kernel the rule gives `1.06 σ̂ n^(-1/5)` and for the Epanechnikov kernel
`2.34 σ̂ n^(-1/5)`.

The rule oversmooths densities that are far from normal, multimodal ones in
particular. It is a quick default and a pilot for better selectors. It needs a
second-order kernel, with `moment(k, 2) > 0`.
"""
struct NormalReference <: BandwidthSelector
  robust::Bool
end

NormalReference(; robust::Bool = true) = NormalReference(robust)

Base.show(io::IO, r::NormalReference) = print(io, "NormalReference(robust = ", r.robust, ")")

function selectbandwidth(rule::NormalReference, d::SmoothingData, k::AbstractKernel)
  sd, iqr = _sdiqr(d)
  σ = rule.robust ? _robustscale(sd, iqr / 1.349) : sd
  _check_scale(σ, rule)
  n = effective_nobs(d)
  h = σ * (8 * sqrt(π) * roughness(k) / (3 * _mu2centred(k)^2 * n))^(1 / 5)
  return BandwidthSelection(h, rule, (scale = σ, sd = sd, iqr = iqr, n = n))
end

"""
    SilvermanRule()

Silverman's rule of thumb, `0.9 min(σ̂, IQR / 1.34) n^(-1/5)` for the Gaussian
kernel, where `σ̂` is the standard deviation, `IQR` the interquartile range and
`n` the [`effective_nobs`](@ref). It is the default bandwidth of R's `density`
(`bw.nrd0`) and of KernelDensity.jl. Other kernels get the bandwidth that R's
`density` uses for them, the Gaussian one divided by the standard deviation of
the kernel, `sqrt(moment(k, 2))`. Compared with [`NormalReference`](@ref), the
factor 0.9 instead of 1.06 suits densities with several modes better.
"""
struct SilvermanRule <: BandwidthSelector end

function selectbandwidth(rule::SilvermanRule, d::SmoothingData, k::AbstractKernel)
  sd, iqr = _sdiqr(d)
  σ = _robustscale(sd, iqr / 1.34)
  _check_scale(σ, rule)
  n = effective_nobs(d)
  h = 0.9 * σ * n^(-1 / 5) / sqrt(_mu2centred(k))
  return BandwidthSelection(h, rule, (scale = σ, sd = sd, iqr = iqr, n = n))
end

# min(sd, s), unless s is zero because more than half of the data are tied
_robustscale(sd, s) = s > 0 ? min(sd, s) : sd

_check_scale(σ, rule) = σ > 0 || throw(ArgumentError(
  "the data have no spread, so $rule cannot choose a bandwidth; pass a number instead"))
_check_spread(d, rule) = _check_scale(d.x[end] - d.x[1], rule)

function _mu2(k)
  μ2 = moment(k, 2)
  μ2 > 0 || throw(ArgumentError("rules of thumb need a second-order kernel, but moment(k, 2) = $μ2 for $k"))
  return μ2
end

# The second moment of a kernel whose first moment vanishes: the rules of thumb
# assume a bias of order h², while a kernel with moment(k, 1) ≠ 0 has a bias of
# order h
function _mu2centred(k)
  μ1 = moment(k, 1)
  abs(μ1) <= 1e-10 * sqrt(abs(moment(k, 2))) || throw(ArgumentError(
    "rules of thumb need a kernel with moment(k, 1) = 0, whose bias is of order h², got $μ1 for $k"))
  return _mu2(k)
end

# The weighted standard deviation and interquartile range. With equal weights
# they are the usual sample standard deviation and the interquartile range of R's
# default quantile definition (type 7).
function _sdiqr(d::SmoothingData)
  return _std(d), _quantile(d, 0.75) - _quantile(d, 0.25)
end

function _mean(d::SmoothingData)
  x, w = d.x, d.w
  s = 0.0
  for i in eachindex(x)
    s += _weight(w, i) * x[i]
  end
  return s / d.sumw
end

# the variance of the empirical distribution, ∑ wᵢ (xᵢ - x̄)² / ∑ wᵢ
function _variance(d::SmoothingData)
  x, w = d.x, d.w
  μ = _mean(d)
  s = 0.0
  for i in eachindex(x)
    s += _weight(w, i) * (x[i] - μ)^2
  end
  return s / d.sumw
end

# corrected with Kish's effective sample size, n / (n - 1) for equal weights
function _std(d::SmoothingData)
  n = effective_nobs(d)
  return n > 1 ? sqrt(_variance(d) * n / (n - 1)) : 0.0
end

_weight(::Nothing, i) = 1.0
_weight(w::Vector{Float64}, i) = @inbounds w[i]

# The p-quantile by linear interpolation between the sorted observations, placed
# at τᵢ = (mᵢ - m₁) / (mₙ - m₁), where mᵢ is the weight before observation i plus
# half its own. Tied observations share their total weight equally, so that their
# order does not matter. With equal weights τᵢ = (i - 1) / (n - 1), R's type 7.
function _quantile(d::SmoothingData, p::Real)
  x, c, n = d.x, d.cumw, nobs(d)
  n == 1 && return x[1]
  function m(i)
    g = d.valueindex[i]
    s, e = d.starts[g], d.starts[g + 1]
    return c[s] + (i - s + 0.5) * (c[e] - c[s]) / (e - s)
  end
  t = m(1) + p * (m(n) - m(1))
  i = clamp(_firsttrue(i -> m(i) > t, n) - 1, 1, n - 1)
  θ = clamp((t - m(i)) / (m(i + 1) - m(i)), 0.0, 1.0)
  return x[i] + θ * (x[i + 1] - x[i])
end
