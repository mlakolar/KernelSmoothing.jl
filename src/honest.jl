## Honest confidence intervals (Armstrong and Kolesár 2020)

"""
    Honest(M; biascorrected = false)

Honest confidence intervals for local polynomial regression (Armstrong and Kolesár
2018, 2020). Under regularity conditions, they have asymptotic coverage at least
`level` uniformly over the Taylor class whose remainder about the point is bounded by
`M |x - x₀|^(p+1) / (p + 1)!` throughout the smoothing windows,

    estimate ± cv(B / se) se,   B = M / (p + 1)! ∑ᵢ |ℓᵢ| |xᵢ - x₀|^(p+1),

where `ℓᵢ` are the weights of the estimate, `B` is its worst-case bias over the
class and `cv(t)` is the `level` quantile of `|N(t, 1)|`. The intervals account
for the bias instead of removing it. A bound `M` on the derivative of order
`p + 1` throughout the windows is sufficient for the remainder bound; over this
smaller class, `B` can be conservative. Coverage uses a normal approximation with
estimated standard errors. With fixed weights, Gaussian errors and the true
standard error, the coverage bound is finite-sample. The intervals need a bound on
the smoothness: adapting to unknown smoothness is impossible over such classes. The bound `M` is
therefore required, chosen before seeing the estimates. With `M = :rot`, it is the
largest absolute derivative of order `p + 1` of a global polynomial of order
`p + 3` over the range of the data (of lower order when `x` has fewer than `p + 4`
distinct values), the rule of thumb of Armstrong and Kolesár for local linear fits,
which they offer as a starting point for a sensitivity analysis, not as a bound. With `biascorrected = true`, the interval is built
around the bias-corrected estimate and its weights.
"""
struct Honest <: InferenceMethod
  M::Union{Float64,Symbol}
  biascorrected::Bool

  function Honest(M::Union{Real,Symbol}, biascorrected::Bool)
    M isa Symbol && M != :rot && throw(ArgumentError("M must be a nonnegative number or :rot, got :$M"))
    Mv = M isa Real ? Float64(M) : M
    M isa Real && !(M >= 0 && isfinite(Mv)) && throw(ArgumentError("M must be a nonnegative number, got $M"))
    new(Mv, biascorrected)
  end
end

Honest(M::Union{Real,Symbol}; biascorrected::Bool = false) = Honest(M, biascorrected)
Honest(; biascorrected::Bool = false) = throw(ArgumentError("Honest needs a bound M on the " *
  "derivative of order p + 1, chosen in advance, such as Honest(40), or Honest(:rot) for the rule of thumb"))

Base.show(io::IO, m::Honest) = print(io, "Honest(", repr(m.M), m.biascorrected ? "; biascorrected = true" : "", ")")

"""
    worstcasebias(m::LocalPolynomialRegression, M; biascorrected = false)

The worst-case bias `M / (p + 1)! ∑ᵢ |ℓᵢ| |xᵢ - x₀|^(p+1)` of the estimates at the
points over the Taylor class with bound `M` on the derivative of order `p + 1`, as
in [`Honest`](@ref); `M = :rot` uses the rule of thumb.
"""
function worstcasebias(m::LocalPolynomialRegression, M; biascorrected::Bool = false)
  Honest(M)   # checks M
  Mv = M === :rot ? _honestM(m.data, m.degree) : Float64(M)::Float64
  p, ν = m.degree, m.derivative
  return _pointwise(eachindex(m.points)) do j
    Mv * _absmoment(m.data, m.kernel, m.points[j], m.h[j], m.b[j], p, ν, biascorrected) / factorial(p + 1)
  end
end

# ∑ᵢ |ℓᵢ| |xᵢ - x₀|^(p+1) for the weights of the estimate, or of the bias-corrected
# estimate, at x0, from the setup f of the local fits there
function _absmoment(d, k, x0, h, b, p, ν, bc, f = _localsetup(d, k, x0, h, b, p, ν))
  f.ok || return NaN
  s = 0.0
  for i in f.a:f.e
    t = d.x[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    ℓ = wi * k(u) * evalpoly(u, f.g)
    bc && (ℓ -= f.ρ * f.s * wi * k(v) * evalpoly(v, f.c))
    s += abs(f.scale * ℓ) * abs(t)^(p + 1)
  end
  return s
end

# the largest |m^(p+1)| of a global polynomial of order p + 3 over the range of x,
# or of the highest order that the distinct values identify
function _honestM(d::SmoothingData, p)
  d.y === nothing && throw(ArgumentError("the rule of thumb for M needs responses"))
  P = min(p + 3, nunique(d) - 1)
  P >= p + 1 || throw(ArgumentError("the rule of thumb for M fits a global polynomial of " *
    "order at least $(p + 1), which needs $(p + 2) distinct values of x, got $(nunique(d)); pass M"))
  x̄, s = _mean(d), sqrt(_variance(d))
  z = (d.x .- x̄) ./ s
  sw = d.w === nothing ? ones(nobs(d)) : sqrt.(d.w)
  # (the solve does not infer, so its type is asserted, and with it the result's)
  α = (LinearAlgebra.qr(sw .* [zi^j for zi in z, j in 0:P]) \ (sw .* d.y))::Vector{Float64}
  r = p + 1
  deriv(zz) = sum(α[j + 1] * prod(j-r+1:j) * zz^(j - r) for j in r:P; init = 0.0) / s^r
  # a polynomial of order 2 attains its largest absolute value at an end or its vertex
  zs = range(z[1], z[end]; length = 1001)
  return maximum(abs ∘ deriv, zs)::Float64
end

# the level quantile of |N(t, 1)|: the c with Φ(c - t) - Φ(-c - t) = level, which
# increases in c from -level at 0; for levels below 1/2 it can be below t
function _cvb(t, level)
  N = Distributions.Normal()
  t = abs(t)
  F(c) = Distributions.cdf(N, c - t) - Distributions.cdf(N, -c - t) - level
  return _findroot(F, 0.0, t + Distributions.quantile(N, (1 + level) / 2) + 1; rtol = 1e-12)[1]
end

function _honestconfint(m::LocalPolynomialRegression, level, method::Honest)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  bc = method.biascorrected
  B = worstcasebias(m, method.M; biascorrected = bc)
  est, se = bc ? (m.estimate_bc, m.se_rbc) : (m.estimate, m.se)
  # cv(B / se) se tends to B as se tends to 0
  half = [se[j] == 0 ? B[j] : _cvb(B[j] / se[j], level) * se[j] for j in eachindex(se)]
  return hcat(est .- half, est .+ half)
end

"""
    FLCIOptimal(M; gridsize = 40, level = 0.95)

The bandwidth at each point that minimizes the length of the [`Honest`](@ref)
interval with bound `M` and coverage `level`, `cv(B(h) / se(h)) se(h)`, as Armstrong
and Kolesár (2020) recommend for honest inference. `M` is required, as for
[`Honest`](@ref); `:rot` selects the rule of thumb. The length is evaluated with the standard errors
of the fit's variance estimator on a grid of bandwidths spaced evenly on the log
scale, from the distance of the `max(minobs, 21)`-th nearest observation to the
range of the data, and the best one is refined by golden-section search. The lower
end does not go below 21 observations, since residual-based standard errors shrink
with the window: with HC0 variances and no floor, the search ended at windows of
about three observations, where the intervals covered 3% of the time.
"""
struct FLCIOptimal <: BandwidthSelector
  M::Union{Float64,Symbol}
  gridsize::Int
  level::Float64

  function FLCIOptimal(M::Union{Real,Symbol}, gridsize::Integer, level::Real)
    Mv = Honest(M).M   # checks the stored bound
    gridsize >= 3 || throw(ArgumentError("gridsize must be at least 3, got $gridsize"))
    lv = Float64(level)
    0 < lv < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
    new(Mv, gridsize, lv)
  end
end

FLCIOptimal(M::Union{Real,Symbol}; gridsize::Integer = 40, level::Real = 0.95) =
  FLCIOptimal(M, gridsize, level)
FLCIOptimal(; kwargs...) = throw(ArgumentError("FLCIOptimal needs a bound M on the derivative " *
  "of order p + 1, chosen in advance, such as FLCIOptimal(40), or FLCIOptimal(:rot) for the rule of thumb"))

function Base.show(io::IO, r::FLCIOptimal)
  print(io, "FLCIOptimal(", repr(r.M))
  _showkeywords(io, r, FLCIOptimal(r.M), "; ")
  print(io, ")")
end

function selectbandwidth(rule::FLCIOptimal, d::SmoothingData, k::AbstractKernel; derivative::Integer = 0,
                         degree::Integer = derivative + 1, points = nothing,
                         vce::Union{Nothing,VarianceEstimator} = nothing, minobs::Integer = 21)
  d.y === nothing && throw(ArgumentError("FLCIOptimal needs responses; use SmoothingData(x, y)"))
  vce = _variance(vce, d)
  _check_degree(degree, derivative)
  _check_spread(d, rule)
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  p, ν = Int(degree), Int(derivative)
  pts = points === nothing ? collect(range(d.x[1], d.x[end]; length = 30)) : _checkpoints(points)
  M = rule.M === :rot ? _honestM(d, p) : rule.M::Float64
  hmax = d.x[end] - d.x[1]
  hs = _pointwise(pts) do c
    hmin = max(knn_distance(d, c, min(max(minobs, 21, p + 2), nobs(d))), hmax / 1e4)
    # one setup of the local fits per bandwidth, for the standard error and the bias
    function halflength(h)
      s = _localsetup(d, k, c, h, h, p, ν)
      f = _localfit(d, k, c, h, h, p, ν, vce; setup = s)
      isfinite(f.se) && f.se > 0 || return Inf
      B = M * _absmoment(d, k, c, h, h, p, ν, false, s) / factorial(p + 1)
      return _cvb(B / f.se, rule.level) * f.se
    end
    grid = exp.(range(log(hmin), log(hmax); length = rule.gridsize))
    vals = halflength.(grid)
    i = argmin(vals)
    1 < i < length(grid) ? _goldenmin(halflength, grid[i - 1], grid[i + 1]; rtol = 1e-4) : grid[i]
  end
  return BandwidthSelection(hs, rule, (M = M, points = pts))
end

_usesbc(method::Honest) = method.biascorrected
_confint(m::LocalPolynomialRegression, level, method::Honest) = _honestconfint(m, level, method)
_confint(m::PointwiseDensity, level, method::Honest) =
  throw(ArgumentError("honest intervals are implemented for local polynomial regression"))
_densitymethod(::Honest) =
  throw(ArgumentError("honest intervals are implemented for local polynomial regression"))
