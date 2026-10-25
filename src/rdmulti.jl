## Regression discontinuity designs with several cutoffs

"""
    MultipleCutoffs

Regression discontinuity estimates for a design in which different units face
different cutoffs (Cattaneo, Keele, Titiunik and Vazquez-Bare 2016), as rdmulti's
`rdmc`: at each cutoff, the estimate of [`RegressionDiscontinuity`](@ref) from the
units that face it; their weighted average, with weights proportional to the
number of observations in the windows; and the pooled estimate, from all units with
the running variable normalized to `x - c` and the cutoff at zero. The pooled
estimate is a weighted average of the effects at the cutoffs, with weights that
depend on the densities of the running variable at the cutoffs.

`m.fits` holds the fit at each cutoff, `m.pooled` the pooled fit, and `m.weights` the
weights of the average; `coeftable(m)` reports all of them, `coef` and `confint` the
weighted average.
"""
struct MultipleCutoffs{F<:RegressionDiscontinuity,P<:RegressionDiscontinuity}
  cutoffs::Vector{Float64}
  fits::Vector{F}
  pooled::P
  weights::Vector{Float64}
  estimate::Float64
  estimate_bc::Float64
  se::Float64
  se_rbc::Float64
end

"""
    CumulativeCutoffs

Regression discontinuity estimates at several cutoffs of one running variable,
where the treatment changes at each cutoff (cumulative cutoffs), as rdmulti's `rdms`
with one score: at each cutoff, the estimate of [`RegressionDiscontinuity`](@ref)
from the observations in a range of the running variable, by default all of them.
`m.fits` holds the fits; `coeftable(m)` reports them.
"""
struct CumulativeCutoffs{F<:RegressionDiscontinuity}
  cutoffs::Vector{Float64}
  ranges::Vector{NTuple{2,Float64}}
  fits::Vector{F}
end

# the options of fit(RegressionDiscontinuity, ...) that hold one value per observation
const _RDPEROBS = (:treatment, :covariates, :weights, :cluster)

# the keyword arguments restricted to the observations idx
function _subsetkwargs(kwargs, idx)
  return (; (k => (k in _RDPEROBS && v !== nothing ? (v isa AbstractMatrix ? v[idx, :] : v[idx]) : v)
              for (k, v) in pairs(kwargs))...)
end

function _checkrdkwargs(kwargs)
  haskey(kwargs, :cutoff) && throw(ArgumentError("the cutoffs are given as an argument, not as the keyword cutoff"))
end

"""
    fit(MultipleCutoffs, x, y, c; kwargs...)

The estimates for a design with several cutoffs, where `c` holds the cutoff that
each unit faces, see [`MultipleCutoffs`](@ref). The keywords are those of
[`fit`](@ref)`(RegressionDiscontinuity, x, y)` and apply to every fit: options with
one value per observation (`treatment`, `covariates`, `weights` and `cluster`) are
restricted to the units of each cutoff, and the bandwidths are chosen at each
cutoff and for the pooled fit separately.
"""
function fit(::Type{MultipleCutoffs}, x::AbstractVector, y::AbstractVector, c::AbstractVector; kwargs...)
  _checkrdkwargs(kwargs)
  n = length(x)
  length(y) == n && length(c) == n || throw(DimensionMismatch("x, y and c must have the same length"))
  _check_values(c, "c")
  cuts = sort!(unique(Float64.(c)))
  length(cuts) >= 2 || throw(ArgumentError("a design with several cutoffs needs at least two; use RegressionDiscontinuity"))
  fits = [begin
            idx = findall(==(cc), c)
            fit(RegressionDiscontinuity, x[idx], y[idx]; cutoff = cc, _subsetkwargs(kwargs, idx)...)
          end for cc in cuts]
  pooled = fit(RegressionDiscontinuity, Float64.(x) .- Float64.(c), y; cutoff = 0, kwargs...)
  N = [sum(f.nh) for f in fits]
  W = N ./ sum(N)
  est = sum(W .* [f.estimate for f in fits])
  bc = sum(W .* [f.estimate_bc for f in fits])
  se = sqrt(sum(W .^ 2 .* [f.se^2 for f in fits]))
  serbc = sqrt(sum(W .^ 2 .* [f.se_rbc^2 for f in fits]))
  return MultipleCutoffs(cuts, fits, pooled, W, est, bc, se, serbc)
end

"""
    fit(CumulativeCutoffs, x, y; cutoffs, ranges = nothing, kwargs...)

The estimates at the `cutoffs` of one running variable, see
[`CumulativeCutoffs`](@ref). `ranges`, one pair `(lo, hi)` per cutoff, restricts the
fit at each cutoff to the observations with `lo ≤ x ≤ hi`, for example to those
between the neighbouring cutoffs; by default every fit uses all observations, as
`rdms`. The other keywords are those of [`fit`](@ref)`(RegressionDiscontinuity, x, y)`.
"""
function fit(::Type{CumulativeCutoffs}, x::AbstractVector, y::AbstractVector; cutoffs::AbstractVector{<:Real},
             ranges = nothing, kwargs...)
  _checkrdkwargs(kwargs)
  length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
  cuts = Float64.(cutoffs)
  issorted(cuts) && allunique(cuts) || throw(ArgumentError("the cutoffs must be increasing"))
  rs = ranges === nothing ? fill((-Inf, Inf), length(cuts)) :
       [(Float64(r[1]), Float64(r[2])) for r in ranges]
  length(rs) == length(cuts) || throw(DimensionMismatch("$(length(rs)) ranges for $(length(cuts)) cutoffs"))
  all(k -> rs[k][1] < cuts[k] < rs[k][2], eachindex(cuts)) ||
    throw(ArgumentError("each range must contain its cutoff in its interior"))
  fits = [begin
            idx = findall(t -> rs[k][1] <= t <= rs[k][2], x)
            fit(RegressionDiscontinuity, x[idx], y[idx]; cutoff = cuts[k], _subsetkwargs(kwargs, idx)...)
          end for k in eachindex(cuts)]
  return CumulativeCutoffs(cuts, rs, fits)
end

"""
    coef(m::MultipleCutoffs; method = Conventional())

The weighted average of the estimates at the cutoffs as a one-element vector, or
with `method = RobustBiasCorrected()` the average of the bias-corrected estimates.
"""
StatsAPI.coef(m::MultipleCutoffs; method::InferenceMethod = Conventional()) =
  [_usesbc(method) ? m.estimate_bc : m.estimate]

"""
    stderror(m::MultipleCutoffs; method = Conventional())

The standard error of the weighted average, `√(∑ₖ Wₖ² seₖ²)` from the independent
estimates at the cutoffs, or with `method = RobustBiasCorrected()` the robust one.
"""
stderror(m::MultipleCutoffs; method::InferenceMethod = Conventional()) = [_usesbc(method) ? m.se_rbc : m.se]

function StatsAPI.confint(m::MultipleCutoffs; level::Real = 0.95, method::InferenceMethod = RobustBiasCorrected())
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  method isa Union{Conventional,RobustBiasCorrected} ||
    throw(ArgumentError("intervals for several cutoffs are Conventional() or RobustBiasCorrected()"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  est, se = coef(m; method)[1], stderror(m; method)[1]
  return [est - z * se est + z * se]
end

# a row of results: the estimate, the robust bias-corrected p-value and interval
function _rdrow(est, bc, serbc, q)
  z = bc / serbc
  return est, 2 * Distributions.ccdf(Distributions.Normal(), abs(z)), bc - q * serbc, bc + q * serbc
end

"""
    coeftable(m::MultipleCutoffs; level = 0.95)
    coeftable(m::CumulativeCutoffs; level = 0.95)

The results as rdmulti reports them, a named tuple of columns with one row per
cutoff, and for several cutoffs the rows `"weighted"` and `"pooled"`: `cutoff`,
`estimate` (conventional), `pvalue`, `lower` and `upper` (robust bias-corrected),
the bandwidths `hleft` and `hright`, the observations in the windows `nh`, and for
several cutoffs the `weight` of each cutoff in the average.
"""
function StatsAPI.coeftable(m::MultipleCutoffs; level::Real = 0.95)
  t = _cutofftable(m.cutoffs, m.fits, level)
  q = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  w = _rdrow(m.estimate, m.estimate_bc, m.se_rbc, q)
  p = m.pooled
  pr = _rdrow(p.estimate, p.estimate_bc, p.se_rbc, q)
  return (cutoff = vcat(t.cutoff, "weighted", "pooled"), estimate = vcat(t.estimate, w[1], pr[1]),
          pvalue = vcat(t.pvalue, w[2], pr[2]), lower = vcat(t.lower, w[3], pr[3]), upper = vcat(t.upper, w[4], pr[4]),
          hleft = vcat(t.hleft, NaN, p.h[1]), hright = vcat(t.hright, NaN, p.h[2]),
          nh = vcat(t.nh, sum(t.nh), sum(p.nh)), weight = vcat(m.weights, NaN, NaN))
end

StatsAPI.coeftable(m::CumulativeCutoffs; level::Real = 0.95) = _cutofftable(m.cutoffs, m.fits, level)

function _cutofftable(cuts, fits, level)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  q = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  rows = [_rdrow(f.estimate, f.estimate_bc, f.se_rbc, q) for f in fits]
  return (cutoff = string.(cuts), estimate = [r[1] for r in rows], pvalue = [r[2] for r in rows],
          lower = [r[3] for r in rows], upper = [r[4] for r in rows], hleft = [f.h[1] for f in fits],
          hright = [f.h[2] for f in fits], nh = [sum(f.nh) for f in fits])
end

function Base.show(io::IO, m::MultipleCutoffs)
  print(io, "MultipleCutoffs(", length(m.cutoffs), " cutoffs, ", sum(nobs, m.fits), " observations)")
end

function Base.show(io::IO, m::CumulativeCutoffs)
  print(io, "CumulativeCutoffs(", length(m.cutoffs), " cutoffs)")
end

function Base.show(io::IO, ::MIME"text/plain", m::Union{MultipleCutoffs,CumulativeCutoffs})
  println(io, m isa MultipleCutoffs ? "Regression discontinuity with several cutoffs" :
              "Regression discontinuity with cumulative cutoffs")
  t = coeftable(m)
  println(io, "  ", rpad("cutoff", 10), rpad("estimate", 11), rpad("p-value", 10), rpad("95% robust interval", 24),
          rpad("h left", 10), rpad("h right", 10), rpad("nh", 7), m isa MultipleCutoffs ? "weight" : "")
  for j in eachindex(t.cutoff)
    print(io, "  ", rpad(t.cutoff[j], 10), rpad(_fmt(t.estimate[j]), 11), rpad(_fmt(t.pvalue[j]), 10),
          rpad(string("[", _fmt(t.lower[j]), ", ", _fmt(t.upper[j]), "]"), 24),
          rpad(isnan(t.hleft[j]) ? "" : _fmt(t.hleft[j]), 10), rpad(isnan(t.hright[j]) ? "" : _fmt(t.hright[j]), 10),
          rpad(t.nh[j], 7))
    m isa MultipleCutoffs && !isnan(t.weight[j]) && print(io, _fmt(t.weight[j]))
    println(io)
  end
end
