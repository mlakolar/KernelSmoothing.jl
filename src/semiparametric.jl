## Varying-coefficient and partially linear models

"""
    VaryingCoefficient <: RegressionModel

The varying-coefficient model `y = β₁(u) x₁ + ⋯ + β_q(u) x_q + ε` (Hastie and
Tibshirani 1993; Fan and Zhang 1999), whose coefficients are smooth functions of an
index variable `u`, fitted with [`fit`](@ref)`(VaryingCoefficient, u, X, y)`. At a
point `u₀`, each coefficient is a local polynomial of order `degree` in `u`,

    ∑ᵢ wᵢ k((uᵢ - u₀)/h) (yᵢ - ∑ⱼ xᵢⱼ ∑ₗ βⱼₗ (uᵢ - u₀)^l)²,

and `β̂ⱼ₀` estimates `βⱼ(u₀)`. A column of ones in `X` gives an intercept that varies
with `u`. The standard errors are the sandwich ones of the local weighted least
squares fit, with its residuals (HC0, or HC1 with `vce = HCVariance(1)`).

`coef(m)` is the matrix of the estimates, one row per point and one column per
coefficient, `stderror(m)` the matrix of their standard errors, and
`confint(m; level)` a named tuple of the matrices `lower` and `upper` of pointwise
intervals.
"""
struct VaryingCoefficient{K<:AbstractKernel,S<:BandwidthSelection} <: StatsAPI.RegressionModel
  kernel::K
  degree::Int
  vce::HCVariance
  selection::S
  nobs::Int
  points::Vector{Float64}
  h::Float64
  coef::Matrix{Float64}
  se::Matrix{Float64}
end

"""
    fit(VaryingCoefficient, u, X, y; points = nothing, bandwidth = CrossValidation(),
        degree = 1, kernel = EpanechnikovKernel(), weights = nothing, vce = HCVariance(0))

Fits the varying-coefficient model of `y` on the columns of `X` (a matrix with one
row per observation, or a vector for one coefficient) with coefficients that vary
with `u`, at `points`, by default 30 equally spaced points between the quantiles of
order 0.025 and 0.975 of `u`. `bandwidth` is a number or
[`CrossValidation`](@ref), which minimizes the leave-one-out prediction error of
the fits at the observations, from the deletion formula.
"""
function fit(::Type{VaryingCoefficient}, u::AbstractVector, X::AbstractVecOrMat, y::AbstractVector;
             points = nothing, bandwidth = CrossValidation(), degree::Integer = 1,
             kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing, vce::HCVariance = HCVariance(0))
  p = Int(degree)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  vce.type <= 1 || throw(ArgumentError("varying-coefficient fits have HC0 and HC1 standard errors"))
  d, Xs = _semidata(u, X, y, weights)
  pts = points === nothing ? _regressionpoints(d, false) : _checkpoints(points)
  sel = if bandwidth isa CrossValidation
    _vcselect(bandwidth, d, Xs, kernel, p)
  elseif bandwidth isa Real
    BandwidthSelection(_bandwidths(bandwidth, 1), nothing, NamedTuple())
  else
    throw(ArgumentError("bandwidth must be a positive number or CrossValidation(), got $(repr(bandwidth))"))
  end
  h = sel.h isa Float64 ? sel.h : sel.h[1]
  fits = _pointwise(j -> _vcfit(d, Xs, kernel, pts[j], h, p, vce), eachindex(pts))
  q = size(Xs, 2)
  coefs = [fits[j].β[l] for j in eachindex(pts), l in 1:q]
  ses = [fits[j].se[l] for j in eachindex(pts), l in 1:q]
  return VaryingCoefficient(kernel, p, vce, _anyselection(sel), nobs(d), pts, h, coefs, ses)
end

# u, X and y as SmoothingData of u and y, and X with its rows in the order of u
function _semidata(u, X, y, weights)
  n = length(u)
  Xm = X isa AbstractVector ? reshape(X, :, 1) : X
  size(Xm, 1) == n || throw(DimensionMismatch("X has $(size(Xm, 1)) rows for $n observations"))
  _check_values(vec(Xm), "X")
  d = SmoothingData(u, y; weights)
  return d, Float64[Xm[i, j] for i in d.perm, j in axes(Xm, 2)]
end

# The local fit of the varying-coefficient model at u0: the estimates of the
# coefficients and their standard errors, or with `loo = i` the fitted value at the
# observation i and its weight there, for leave-one-out cross-validation
function _vcfit(d, X, k, u0, h, p, vce; loo::Int = 0)
  q = size(X, 2)
  P = p + 1
  D = q * P
  r = _poswindow(d, k, u0, h)
  A, c = zeros(D, D), zeros(D)
  z = zeros(D)
  function design!(z, i)
    v = (d.x[i] - u0) / h
    for j in 1:q
      t = X[i, j]
      for l in 1:P
        z[(j - 1) * P + l] = t
        t *= v
      end
    end
    return z
  end
  for i in r
    kw = _weight(d.w, i) * k((d.x[i] - u0) / h)
    kw == 0 && continue
    design!(z, i)
    for a in 1:D
      c[a] += kw * z[a] * d.y[i]
      for b in a:D
        A[a, b] += kw * z[a] * z[b]
      end
    end
  end
  for a in 1:D, b in 1:a-1
    A[a, b] = A[b, a]
  end
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
  ok = !isempty(r) && LinearAlgebra.issuccess(C) && minimum(j -> C.factors[j, j]^2 / A[j, j], 1:D) > 1e-10
  idx = [(j - 1) * P + 1 for j in 1:q]
  if loo > 0
    ok || return (fitted = NaN, self = NaN)
    θ = C \ c
    x̃ = zeros(D)
    x̃[idx] = X[loo, :]
    return (fitted = θ' * x̃, self = _weight(d.w, loo) * k(0.0) * (x̃' * (C \ x̃)))
  end
  ok || return (β = fill(NaN, q), se = fill(NaN, q))
  θ = C \ c
  meat = zeros(D, D)
  m = 0
  for i in r
    kw = _weight(d.w, i) * k((d.x[i] - u0) / h)
    kw == 0 && continue
    m += 1
    design!(z, i)
    e = d.y[i] - z' * θ
    meat .+= (kw * e)^2 .* (z * z')
  end
  vce.type == 1 && (meat .*= m > D ? m / (m - D) : NaN)
  V = C \ (C \ meat)'
  return (β = θ[idx], se = sqrt.(max.(LinearAlgebra.diag(V)[idx], 0.0)))
end

# leave-one-out cross-validation over a grid of bandwidths spaced evenly on the log
# scale, from the distance at which every observation has enough distinct values of u
# in its window to the range of u
function _vcselect(rule, d, X, k, p)
  D = size(X, 2) * (p + 1)
  nunique(d) > D || throw(ArgumentError("cross-validation needs more than $D distinct values of u"))
  hmin = maximum(i -> knn_distance(d, d.x[i], D + 1; distinct = true), 1:nobs(d)) * 1.01
  hmax = d.x[end] - d.x[1]
  hmin < hmax || (hmin = hmax / 2)
  crit(h) = _vccriterion(d, X, k, h, p)
  h, details = _gridminimize(crit, exp.(range(log(hmin), log(hmax); length = rule.gridsize)); largest = false, rtol = 1e-3)
  return BandwidthSelection(h, rule, (grid = details.grid, criterion = details.criterion))
end

function _vccriterion(d, X, k, h, p)
  parts = _pointwise(i -> begin
    f = _vcfit(d, X, k, d.x[i], h, p, HCVariance(0); loo = i)
    wi = _weight(d.w, i)
    pred = (f.fitted - f.self * d.y[i]) / (1 - f.self)
    (wi * (d.y[i] - pred)^2, wi)
  end, 1:nobs(d))
  s = sum(first, parts)
  return isfinite(s) ? s / sum(last, parts) : Inf
end

nobs(m::VaryingCoefficient) = m.nobs
bandwidth(m::VaryingCoefficient) = m.h
StatsAPI.coef(m::VaryingCoefficient) = copy(m.coef)
stderror(m::VaryingCoefficient) = copy(m.se)

function StatsAPI.confint(m::VaryingCoefficient; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  return (lower = m.coef .- z .* m.se, upper = m.coef .+ z .* m.se)
end

function Base.show(io::IO, m::VaryingCoefficient)
  print(io, "VaryingCoefficient(", size(m.coef, 2), " coefficients, degree ", m.degree, ", ", m.kernel, ", ",
        length(m.points), " points, ", nobs(m), " observations)")
end


"""
    PartiallyLinear <: RegressionModel

The partially linear model `y = x'β + g(u) + ε` (Robinson 1988; Speckman 1988),
fitted with [`fit`](@ref)`(PartiallyLinear, u, X, y)`. The parametric part is the
weighted least squares regression of `y - m̂_y(u)` on `X - m̂_X(u)`, where `m̂_y` and
`m̂_X` are local polynomial smooths in `u` of `y` and of each column of `X` with the
same bandwidth, which removes `g` without estimating it first; `β̂` is then
`√n`-consistent and its standard errors are the sandwich ones of that regression
(HC0, or HC1 with `vce = HCVariance(1)`). The smooth part `ĝ` is the smooth of
`y - x'β̂` in `u`. `X` must not contain a constant column, which `g` absorbs.

`coef(m)`, `stderror(m)`, `vcov(m)`, `confint(m)` and `coeftable(m)` describe `β̂`;
`m.points` and `m.smooth` hold `ĝ` at the points.
"""
struct PartiallyLinear{K<:AbstractKernel,S<:BandwidthSelection} <: StatsAPI.RegressionModel
  kernel::K
  degree::Int
  vce::HCVariance
  selection::S
  nobs::Int
  h::Float64
  coef::Vector{Float64}
  vcov::Matrix{Float64}
  points::Vector{Float64}
  smooth::Vector{Float64}
end

"""
    fit(PartiallyLinear, u, X, y; points = nothing, bandwidth = CrossValidation(),
        degree = 1, kernel = EpanechnikovKernel(), weights = nothing, vce = HCVariance(0))

Fits the partially linear model of `y` on the columns of `X` and a smooth function
of `u`. `bandwidth` is a number or [`CrossValidation`](@ref), which chooses the
bandwidth of the smooth of `y` in `u` by leave-one-out cross-validation; the
estimate of `β` is not sensitive to it within a wide range. `points` are those of
the smooth part, by default as for [`LocalPolynomialRegression`](@ref).
"""
function fit(::Type{PartiallyLinear}, u::AbstractVector, X::AbstractVecOrMat, y::AbstractVector;
             points = nothing, bandwidth = CrossValidation(), degree::Integer = 1,
             kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing, vce::HCVariance = HCVariance(0))
  p = Int(degree)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  vce.type <= 1 || throw(ArgumentError("partially linear fits have HC0 and HC1 standard errors"))
  d, Xs = _semidata(u, X, y, weights)
  pts = points === nothing ? _regressionpoints(d, false) : _checkpoints(points)
  sel = if bandwidth isa CrossValidation
    selectbandwidth(bandwidth, d, kernel; degree = p)
  elseif bandwidth isa Real
    BandwidthSelection(_bandwidths(bandwidth, 1), nothing, NamedTuple())
  else
    throw(ArgumentError("bandwidth must be a positive number or CrossValidation(), got $(repr(bandwidth))"))
  end
  h = sel.h isa Float64 ? sel.h : sel.h[1]
  smooth(v) = localpolynomial(_withy(d, v), d.x; bandwidth = h, degree = p, kernel)
  ỹ = d.y .- smooth(d.y)
  X̃ = hcat([Xs[:, j] .- smooth(Xs[:, j]) for j in axes(Xs, 2)]...)
  all(isfinite, ỹ) && all(isfinite, X̃) || throw(ArgumentError("a smooth in u is not identified at some observation; use a larger bandwidth"))
  w = d.w === nothing ? ones(nobs(d)) : d.w
  F = LinearAlgebra.qr(sqrt.(w) .* X̃, LinearAlgebra.ColumnNorm())
  R = F.R
  rank = count(j -> abs(R[j, j]) > 1e-10 * abs(R[1, 1]), 1:min(size(R)...))
  rank == size(X̃, 2) || throw(ArgumentError("the columns of X are collinear after removing their smooths in u; " *
    "drop constant columns, which the smooth part absorbs"))
  β = F \ (sqrt.(w) .* ỹ)
  e = ỹ .- X̃ * β
  G = X̃' * (w .* X̃)
  meat = X̃' * ((w .* e) .^ 2 .* X̃)
  n, q = nobs(d), size(X̃, 2)
  vce.type == 1 && (meat .*= n / (n - q))
  V = G \ (G \ meat)'
  g = localpolynomial(_withy(d, d.y .- Xs * β), pts; bandwidth = h, degree = p, kernel)
  return PartiallyLinear(kernel, p, vce, _anyselection(sel), n, h, β, (V + V') / 2, pts, g)
end

nobs(m::PartiallyLinear) = m.nobs
bandwidth(m::PartiallyLinear) = m.h
StatsAPI.coef(m::PartiallyLinear) = copy(m.coef)
StatsAPI.vcov(m::PartiallyLinear) = copy(m.vcov)
stderror(m::PartiallyLinear) = sqrt.(max.(LinearAlgebra.diag(m.vcov), 0.0))

function StatsAPI.confint(m::PartiallyLinear; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  se = stderror(m)
  return hcat(m.coef .- z .* se, m.coef .+ z .* se)
end

"""
    coeftable(m::PartiallyLinear; level = 0.95)

The coefficients of the parametric part, a named tuple of columns `estimate`, `se`,
`z`, `pvalue`, `lower` and `upper`.
"""
function StatsAPI.coeftable(m::PartiallyLinear; level::Real = 0.95)
  ci = confint(m; level)
  se = stderror(m)
  z = m.coef ./ se
  return (estimate = copy(m.coef), se = se, z = z, pvalue = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z)),
          lower = ci[:, 1], upper = ci[:, 2])
end

function Base.show(io::IO, m::PartiallyLinear)
  print(io, "PartiallyLinear(", length(m.coef), " coefficients, degree ", m.degree, ", ", m.kernel, ", ",
        nobs(m), " observations)")
end
