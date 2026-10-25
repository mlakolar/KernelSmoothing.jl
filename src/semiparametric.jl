## Partially linear models

# u, X and y as SmoothingData of u and y, and X with its rows in the order of u
function _semidata(u, X, y, weights)
  n = length(u)
  Xm = X isa AbstractVector ? reshape(X, :, 1) : X
  size(Xm, 1) == n || throw(DimensionMismatch("X has $(size(Xm, 1)) rows for $n observations"))
  _check_values(vec(Xm), "X")
  d = SmoothingData(u, y; weights)
  return d, Float64[Xm[i, j] for i in d.perm, j in axes(Xm, 2)]
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
    select_bandwidth(bandwidth, d, kernel; degree = p)
  elseif bandwidth isa Real
    BandwidthSelection(_bandwidths(bandwidth, 1), nothing, NamedTuple())
  else
    throw(ArgumentError("bandwidth must be a positive number or CrossValidation(), got $(repr(bandwidth))"))
  end
  h = sel.h isa Float64 ? sel.h : sel.h[1]
  smooth(v) = local_polynomial(_withy(d, v), d.x; bandwidth = h, degree = p, kernel)
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
  g = local_polynomial(_withy(d, d.y .- Xs * β), pts; bandwidth = h, degree = p, kernel)
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
