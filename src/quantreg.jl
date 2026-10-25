## Local polynomial quantile regression

# Weighted linear quantile regression,
#
#     β̂ = argmin ∑ᵢ wᵢ ρ_τ(yᵢ - zᵢᵀβ),   ρ_τ(u) = u (τ - 1{u < 0}),
#
# solved exactly by the simplex method on the dual linear program
#
#     maximize ∑ᵢ wᵢ yᵢ aᵢ   subject to   ∑ᵢ wᵢ zᵢ aᵢ = (1 - τ) ∑ᵢ wᵢ zᵢ,   0 ≤ aᵢ ≤ 1,
#
# whose multipliers of the equality constraints are β̂: at the optimum the basic
# observations are interpolated, aᵢ = 1 where yᵢ > zᵢᵀβ̂ and aᵢ = 0 where yᵢ < zᵢᵀβ̂.
# Z has one row per observation; the observations with wᵢ = 0 are left out.
# Returns β̂ and the indices of the interpolated observations.
function _rqsimplex(Z::AbstractMatrix{Float64}, y::AbstractVector{Float64}, w::AbstractVector{Float64}, τ::Float64)
  keep = findall(>(0), w)
  n, m = length(keep), size(Z, 2)
  n >= m || return fill(NaN, m), Int[]
  A = Matrix{Float64}(undef, m, n)
  c = Vector{Float64}(undef, n)
  for (k, i) in enumerate(keep)
    for j in 1:m
      A[j, k] = w[i] * Z[i, j]
    end
    c[k] = w[i] * y[i]
  end
  scale = maximum(abs, c) + eps()
  # a basic feasible solution from the interior point a = 1 - τ: move within the null
  # space of the columns of m + 1 free variables, never lowering the objective,
  # until one reaches a bound, as long as more than m are free or their columns are
  # dependent, as for tied observations
  a = fill(1 - τ, n)
  free = collect(1:n)
  while !isempty(free)
    S = length(free) > m ? free[1:m+1] : copy(free)
    F = LinearAlgebra.svd(A[:, S]; full = true)
    length(S) <= m && F.S[end] > 1e-12 * F.S[1] && break   # independent columns
    d = F.V[:, end]
    LinearAlgebra.dot(view(c, S), d) < 0 && (d = -d)
    t, hit = Inf, 0
    for (k, j) in enumerate(S)
      tk = d[k] > 0 ? (1 - a[j]) / d[k] : d[k] < 0 ? -a[j] / d[k] : Inf
      tk < t && ((t, hit) = (tk, k))
    end
    for (k, j) in enumerate(S)
      a[j] = clamp(a[j] + t * d[k], 0.0, 1.0)
    end
    a[S[hit]] = d[hit] > 0 ? 1.0 : 0.0
    filter!(j -> 0 < a[j] < 1, free)
  end
  basis = copy(free)
  # complete the basis with columns that keep it nonsingular
  if length(basis) < m
    for j in 1:n
      length(basis) == m && break
      j in basis && continue
      B = A[:, vcat(basis, j)]
      LinearAlgebra.rank(B; rtol = 1e-10) == length(basis) + 1 && push!(basis, j)
    end
    length(basis) < m && return fill(NaN, m), Int[]   # the design does not identify β
  end
  isbasic = falses(n)
  isbasic[basis] .= true
  degenerate = 0
  π = zeros(m)
  for _ in 1:50*(n+m)
    B = A[:, basis]
    F = LinearAlgebra.lu(B; check = false)
    LinearAlgebra.issuccess(F) || return fill(NaN, m), Int[]
    π = F' \ c[basis]
    # the entering variable: the largest improving reduced cost, or the first one
    # (Bland's rule) after a run of degenerate pivots
    enter, best, s = 0, 0.0, 0
    for j in 1:n
      isbasic[j] && continue
      dj = c[j] - LinearAlgebra.dot(view(A, :, j), π)
      tol = 1e-11 * scale
      sj = a[j] == 0 ? (dj > tol ? 1 : 0) : (dj < -tol ? -1 : 0)
      sj == 0 && continue
      if degenerate > 30
        enter, s = j, sj
        break
      elseif abs(dj) > best
        enter, best, s = j, abs(dj), sj
      end
    end
    enter == 0 && return π, keep[basis]
    u = F \ A[:, enter]
    # the ratio test: the entering variable moves by t s, the basic ones by -t s u
    t, leave = 1.0, 0
    for k in 1:m
      δ = -s * u[k]
      j = basis[k]
      tk = δ > 1e-14 ? (1 - a[j]) / δ : δ < -1e-14 ? a[j] / -δ : Inf
      if tk < t || (degenerate > 30 && tk == t && leave > 0 && j < basis[leave])
        t, leave = tk, k
      end
    end
    t = max(t, 0.0)
    for k in 1:m
      a[basis[k]] = clamp(a[basis[k]] - t * s * u[k], 0.0, 1.0)
    end
    degenerate = t == 0 ? degenerate + 1 : 0
    if leave == 0   # the entering variable reaches its other bound
      a[enter] = s > 0 ? 1.0 : 0.0
    else
      j = basis[leave]
      a[j] = -s * u[leave] > 0 ? 1.0 : 0.0
      a[enter] = clamp(a[enter] + t * s, 0.0, 1.0)
      isbasic[j] = false
      isbasic[enter] = true
      basis[leave] = enter
    end
  end
  return π, keep[basis]
end

"""
    LocalQuantileRegression

Local polynomial estimates of a conditional quantile of `y` given `x` and of its
derivatives (Yu and Jones 1998): at a point `x₀`, the fit of order `degree`
minimizes the kernel-weighted check loss

    ∑ᵢ wᵢ k((xᵢ - x₀)/h) ρ_τ(yᵢ - β₀ - β₁ (xᵢ - x₀) - ⋯ - β_p (xᵢ - x₀)^p),
    ρ_τ(u) = u (τ - 1{u < 0}),

and `ν! β̂_ν` estimates the derivative of order `ν` of the quantile function
`q_τ(x)`. Each local fit is a linear program, solved exactly by the simplex method,
so that the fit interpolates `degree + 1` observations of its window. Fitted with
[`fit`](@ref)`(LocalQuantileRegression, x, y)`; `coef(m)` returns the estimates at
the points, `stderror(m)` and `confint(m; level)` the standard errors and
intervals of [`stderror`](@ref).
"""
struct LocalQuantileRegression{D<:SmoothingData,K<:AbstractKernel,S<:BandwidthSelection}
  data::D
  kernel::K
  tau::Float64
  degree::Int
  derivative::Int
  points::Vector{Float64}
  h::Float64
  coefficients::Matrix{Float64}   # β̂ at each point, one row per point
  selection::S
end

"""
    fit(LocalQuantileRegression, x, y; tau = 0.5, bandwidth = YuJones(), degree = 1, derivative = 0,
        kernel = EpanechnikovKernel(), points = nothing, weights = nothing)

Local polynomial estimates of the quantile of order `tau` of `y` given `x`, or of
its derivative of order `derivative`, at `points` (by default 30 points between
the quantiles of order 0.025 and 0.975 of `x`). The `bandwidth` is a positive number
or a rule, by default [`YuJones`](@ref)`()`.
"""
function fit(::Type{LocalQuantileRegression}, x::AbstractVector, y::AbstractVector; weights = nothing, kwargs...)
  return fit(LocalQuantileRegression, SmoothingData(x, y; weights); kwargs...)
end

function fit(::Type{LocalQuantileRegression}, d::SmoothingData; tau::Real = 0.5, bandwidth = YuJones(),
             degree::Integer = 1, derivative::Integer = 0, kernel::AbstractKernel = EpanechnikovKernel(),
             points = nothing)
  _noclusters(d, "LocalQuantileRegression")
  d.y === nothing && throw(ArgumentError("quantile regression needs responses; use SmoothingData(x, y)"))
  0 < tau < 1 || throw(ArgumentError("tau must be between 0 and 1, got $tau"))
  _check_degree(degree, derivative)
  τ, p = Float64(tau), Int(degree)
  pts = points === nothing ? _regressionpoints(d, false) : _checkpoints(points)
  bandwidth === nothing && (bandwidth = YuJones())
  sel = bandwidth isa YuJones ? select_bandwidth(bandwidth, d, kernel; degree = p, tau = τ) :
        bandwidth isa BandwidthSelector ? select_bandwidth(bandwidth, d, kernel; degree = p) : BandwidthSelection(bandwidth)
  h = sel.h
  h isa Float64 || throw(ArgumentError("quantile regression takes one bandwidth for all points"))
  rows = _pointwise(x0 -> _localquantile(d, kernel, x0, h, p, τ), pts)
  β = reduce(vcat, (r' for r in rows); init = zeros(0, p + 1))
  return LocalQuantileRegression(d, kernel, τ, p, Int(derivative), pts, h, β, sel)
end

"""
    YuJones(; rule = CrossValidation())

The bandwidth of Yu and Jones (1998) for local polynomial quantile regression,

    h_τ = h (τ (1 - τ) / φ(Φ⁻¹(τ))²)^(1/5),

which rescales the bandwidth `h` that `rule` chooses for local polynomial mean
regression of the same degree by the ratio of the asymptotically optimal bandwidths
for a conditional quantile and a conditional mean when the errors are normal and
homoscedastic. It widens the bandwidth away from the median, where fewer
observations inform a quantile.
"""
struct YuJones{R<:BandwidthSelector} <: BandwidthSelector
  rule::R
end

YuJones(; rule::BandwidthSelector = CrossValidation()) = YuJones(rule)

function select_bandwidth(rule::YuJones, d::SmoothingData, k::AbstractKernel; degree::Integer = 1, tau::Real = 0.5)
  0 < tau < 1 || throw(ArgumentError("tau must be between 0 and 1, got $tau"))
  base = select_bandwidth(rule.rule, d, k; degree)
  base.h isa Float64 || throw(ArgumentError("$(rule.rule) must choose one bandwidth"))
  z = quantile(Distributions.Normal(), tau)
  factor = (tau * (1 - tau) / Distributions.pdf(Distributions.Normal(), z)^2)^(1 / 5)
  return BandwidthSelection(base.h * factor, rule, (meanbandwidth = base.h, factor = factor))
end

# the coefficients of the local fit at x0, NaN where its window does not identify
# them
function _localquantile(d, k, x0, h, p, τ)
  r = _poswindow(d, k, x0, h)
  length(r) >= p + 1 && nunique(d, r) >= p + 1 || return fill(NaN, p + 1)
  Z = [((d.x[i] - x0) / h)^j for i in r, j in 0:p]
  w = [_weight(d.w, i) * k((d.x[i] - x0) / h) for i in r]
  β, _ = _rqsimplex(Z, d.y[r], w, τ)
  return β ./ h .^ (0:p)
end

function Base.show(io::IO, m::LocalQuantileRegression)
  print(io, "LocalQuantileRegression(tau = ", m.tau, ", degree = ", m.degree,
        m.derivative == 0 ? "" : ", derivative = $(m.derivative)", ", bandwidth = ", round(m.h; sigdigits = 4), ", ",
        length(m.points), " points, ", nobs(m.data), " observations)")
end

bandwidth(m::LocalQuantileRegression) = m.h
nobs(m::LocalQuantileRegression) = nobs(m.data)
coef(m::LocalQuantileRegression) = factorial(m.derivative) .* m.coefficients[:, m.derivative + 1]

"""
    stderror(m::LocalQuantileRegression)

The standard errors of the estimates, from the sandwich of a weighted quantile
regression (Koenker 2005, Section 3.4) with the conditional density at the quantile
estimated locally: at a point, the variance of `β̂` is about

    τ (1 - τ) s² (ZᵀWZ)⁻¹ (ZᵀW²Z) (ZᵀWZ)⁻¹,

where `W` holds the kernel weights times the weights of the observations and `s` is
the sparsity `1 / f(q_τ(x₀) | x₀)`, estimated by the difference quotient
`(q̂_{τ+δ}(x₀) - q̂_{τ-δ}(x₀)) / (2δ)` of local fits at the orders `τ ± δ`, with the
bandwidth `δ` of Hall and Sheather (1988) for the effective number of observations
in the window. They describe the variability of the estimates, not their bias, so
that intervals from them cover only when the bandwidth is small enough for the bias
to be negligible.
"""
stderror(m::LocalQuantileRegression) = _pointwise(j -> _rqse(m, j), eachindex(m.points))

function _rqse(m::LocalQuantileRegression, j)
  d, k, x0, h, p, τ = m.data, m.kernel, m.points[j], m.h, m.degree, m.tau
  all(isfinite, view(m.coefficients, j, :)) || return NaN
  r = _poswindow(d, k, x0, h)
  w = [_weight(d.w, i) * k((d.x[i] - x0) / h) for i in r]
  neff = sum(w)^2 / sum(abs2, w)
  z = quantile(Distributions.Normal(), 0.975)
  qz = quantile(Distributions.Normal(), τ)
  δ = neff^(-1 / 3) * z^(2 / 3) * (1.5 * Distributions.pdf(Distributions.Normal(), qz)^2 / (2 * qz^2 + 1))^(1 / 3)
  δ = min(δ, τ / 2, (1 - τ) / 2)
  hi, lo = _localquantile(d, k, x0, h, p, τ + δ)[1], _localquantile(d, k, x0, h, p, τ - δ)[1]
  s = max(hi - lo, 0.0) / (2δ)
  Z = [((d.x[i] - x0) / h)^q for i in r, q in 0:p]
  G = Z' * (w .* Z)
  V = G \ (Z' * (w .^ 2 .* Z)) / G
  ν = m.derivative
  return sqrt(max(τ * (1 - τ) * V[ν + 1, ν + 1], 0.0)) * s * factorial(ν) / h^ν
end

function confint(m::LocalQuantileRegression; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = quantile(Distributions.Normal(), (1 + level) / 2)
  b, se = coef(m), stderror(m)
  return hcat(b .- z .* se, b .+ z .* se)
end
