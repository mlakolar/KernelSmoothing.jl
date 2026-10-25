## Local likelihood

"""
    LocalLikelihood <: RegressionModel

Local polynomial likelihood estimates of a regression function for binary or count
responses (Tibshirani and Hastie 1987; Fan, Heckman and Wand 1995), fitted with
[`fit`](@ref)`(LocalLikelihood, x, y; family)`. At a point `x₀`, a polynomial of
order `degree` in the linear predictor maximizes the kernel-weighted log likelihood

    ∑ᵢ wᵢ k((xᵢ - x₀)/h) ℓ(yᵢ; β₀ + β₁ (xᵢ - x₀) + ⋯ + βₚ (xᵢ - x₀)^p),

for `family = :logit` (binary responses, logistic link) or `:poisson` (counts, log
link). `β̂₀` estimates the linear predictor at `x₀` and its inverse link the mean.
Its standard error is the sandwich one of the local fit, which does not rely on the
variance of the family, and intervals for the mean transform those for the linear
predictor, so that they stay within its range.

`coef(m)` and `stderror(m)` describe the linear predictor at the points,
`predict(m)` gives the means, and `confint(m; level, scale)` the intervals, for the
mean by default and with `scale = :linear` for the linear predictor. Points where
the local likelihood has no maximum, as where all responses in the window are equal
for `:logit`, are `NaN`.
"""
struct LocalLikelihood{K<:AbstractKernel,S<:BandwidthSelection} <: StatsAPI.RegressionModel
  family::Symbol
  kernel::K
  degree::Int
  selection::S
  nobs::Int
  points::Vector{Float64}
  h::Vector{Float64}
  eta::Vector{Float64}
  se::Vector{Float64}
end

"""
    fit(LocalLikelihood, x, y; family = :logit, points = nothing, bandwidth = CrossValidation(),
        degree = 1, kernel = EpanechnikovKernel(), weights = nothing)

Local likelihood estimates of the regression of the binary (`family = :logit`) or
count (`:poisson`) responses `y` on `x` at `points`, by default as for
[`LocalPolynomialRegression`](@ref). `bandwidth` is a number, one per point, or
[`CrossValidation`](@ref), which maximizes the leave-one-out log likelihood, with the
fits without each observation approximated by one Newton step from the fit with it,
over bandwidths whose windows hold at least 21 other observations.
"""
function fit(::Type{LocalLikelihood}, x::AbstractVector, y::AbstractVector; family::Symbol = :logit,
             points = nothing, bandwidth = CrossValidation(), degree::Integer = 1,
             kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing)
  family in (:logit, :poisson) || throw(ArgumentError("family must be :logit or :poisson, got $(repr(family))"))
  p = Int(degree)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  if family === :logit
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("the logit family needs responses 0 or 1"))
  else
    all(v -> v >= 0 && isinteger(v), y) || throw(ArgumentError("the Poisson family needs nonnegative integer responses"))
  end
  d = SmoothingData(x, y; weights)
  pts = points === nothing ? _regressionpoints(d, false) : _checkpoints(points)
  sel = if bandwidth isa CrossValidation
    _llselect(bandwidth, d, kernel, p, family)
  elseif bandwidth isa Union{Real,AbstractVector{<:Real}}
    BandwidthSelection(bandwidth isa Real ? _bandwidths(bandwidth, 1) : _bandwidths(bandwidth, length(pts)), nothing, NamedTuple())
  else
    throw(ArgumentError("bandwidth must be positive numbers or CrossValidation(), got $(repr(bandwidth))"))
  end
  h = _pointbandwidths(sel.h, length(pts))
  fits = _pointwise(j -> _llfit(d, kernel, pts[j], h[j], p, family), eachindex(pts))
  return LocalLikelihood(family, kernel, p, _anyselection(sel), nobs(d), pts, h,
                         [f.eta for f in fits], [f.se for f in fits])
end

_lllinkinv(family, η) = family === :logit ? 1 / (1 + exp(-η)) : exp(η)
_llvariance(family, μ) = family === :logit ? μ * (1 - μ) : μ

# The local likelihood fit of order p at x0 by Newton's method from the local
# constant fit: the linear predictor and its sandwich standard error, or with
# `loo = i` the leave-one-out linear predictor at observation i by one Newton step
function _llfit(d, k, x0, h, p, family; loo::Int = 0)
  P = p + 1
  r = _poswindow(d, k, x0, h)
  nan = loo > 0 ? (eta = NaN,) : (eta = NaN, se = NaN)
  isempty(r) && return nan
  kw = [_weight(d.w, i) * k((d.x[i] - x0) / h) for i in r]
  Z = [((d.x[i] - x0) / h)^l for i in r, l in 0:p]
  y = d.y[r]
  # start from the local constant fit, the link of the weighted mean
  ȳ = sum(kw .* y) / sum(kw)
  (family === :logit ? 0 < ȳ < 1 : ȳ > 0) || return nan
  β = zeros(P)
  β[1] = family === :logit ? log(ȳ / (1 - ȳ)) : log(ȳ)
  A = zeros(P, P)
  ok = false
  for _ in 1:100
    η = Z * β
    μ = _lllinkinv.(family, η)
    v = _llvariance.(family, μ)
    A = Z' * ((kw .* v) .* Z)
    g = Z' * (kw .* (y .- μ))
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
    LinearAlgebra.issuccess(C) && minimum(j -> C.factors[j, j]^2 / A[j, j], 1:P) > 1e-12 || return nan
    δ = C \ g
    β .+= δ
    all(isfinite, β) || return nan
    if maximum(abs, δ) <= 1e-12 * (1 + maximum(abs, β))
      ok = true
      break
    end
  end
  ok || return nan
  η = Z * β
  μ = _lllinkinv.(family, η)
  A = Z' * ((kw .* _llvariance.(family, μ)) .* Z)
  if loo > 0
    # one Newton step without observation loo, whose design vector at x0 = x[loo] is e₁
    l = loo - first(r) + 1
    a11 = (A \ [j == 1 ? 1.0 : 0.0 for j in 1:P])[1]
    hii = kw[l] * _llvariance(family, μ[l]) * a11
    return (eta = β[1] - a11 * kw[l] * (y[l] - μ[l]) / (1 - hii),)
  end
  B = Z' * ((kw .* (y .- μ)) .^ 2 .* Z)
  V = A \ (A \ B)'
  return (eta = β[1], se = sqrt(max(V[1, 1], 0.0)))
end

# The bandwidth that maximizes the leave-one-out log likelihood over a grid spaced
# evenly on the log scale, from the distance at which every observation has 21 others
# and p + 2 distinct values in its window, to the range of x. In smaller windows the
# local likelihood is unstable and the one-step leave-one-out fits are poor: for
# Poisson counts at n = 500 the criterion had a spurious minimum at windows of a few
# observations.
function _llselect(rule, d, k, p, family)
  nunique(d) > p + 1 || throw(ArgumentError("cross-validation needs more than $(p + 1) distinct values of x"))
  grid = bandwidthgrid(d; minobs = 22, mindistinct = p + 2, gridsize = rule.gridsize)
  crit(h) = _llcriterion(d, k, h, p, family)
  h, details = _gridminimize(crit, grid; largest = false, rtol = 1e-3)
  return BandwidthSelection(h, rule, (grid = details.grid, criterion = details.criterion))
end

# minus the weighted leave-one-out log likelihood per unit of weight
function _llcriterion(d, k, h, p, family)
  parts = _pointwise(i -> begin
    η = _llfit(d, k, d.x[i], h, p, family; loo = i).eta
    wi, yi = _weight(d.w, i), d.y[i]
    ll = family === :logit ? yi * η - log1p(exp(η)) : yi * η - exp(η)
    (-wi * ll, wi)
  end, 1:nobs(d))
  s = sum(first, parts)
  return isfinite(s) ? s / sum(last, parts) : Inf
end

nobs(m::LocalLikelihood) = m.nobs
bandwidth(m::LocalLikelihood) = m.h
StatsAPI.coef(m::LocalLikelihood) = copy(m.eta)
stderror(m::LocalLikelihood) = copy(m.se)
StatsAPI.predict(m::LocalLikelihood) = _lllinkinv.(m.family, m.eta)

"""
    confint(m::LocalLikelihood; level = 0.95, scale = :mean)

Pointwise intervals for the mean, the inverse link of the intervals for the linear
predictor, or with `scale = :linear` for the linear predictor, as a matrix with the
lower bounds in the first column.
"""
function StatsAPI.confint(m::LocalLikelihood; level::Real = 0.95, scale::Symbol = :mean)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  scale in (:mean, :linear) || throw(ArgumentError("scale must be :mean or :linear, got $(repr(scale))"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  lo, hi = m.eta .- z .* m.se, m.eta .+ z .* m.se
  scale === :mean && ((lo, hi) = (_lllinkinv.(m.family, lo), _lllinkinv.(m.family, hi)))
  return hcat(lo, hi)
end

function Base.show(io::IO, m::LocalLikelihood)
  print(io, "LocalLikelihood(", m.family, ", degree ", m.degree, ", ", m.kernel, ", ", length(m.points),
        " points, ", nobs(m), " observations)")
end
