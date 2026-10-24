## Covariances between points and uniform confidence bands

"""
    vcov(m; method = Conventional())

The covariance matrix of the estimates of a pointwise fit `m`
([`LocalPolynomialRegression`](@ref) or [`PointwiseDensity`](@ref)) at its points,
or with `method = RobustBiasCorrected()` of the bias-corrected estimates.

Each estimate is linear in the data, and its standard error is the norm of a
vector of residual-weighted kernel weights. The covariances are the inner products
of these vectors, so that the block of identified estimates is positive
semidefinite and its diagonal holds the squared standard errors of the fit.
Regression estimates whose windows do
not overlap are uncorrelated; density estimates are then slightly negatively
correlated, since every observation counts towards the total weight. For
regression, these are the covariances of nprobust's `covgrid` with HC variances;
nprobust recomputes nearest-neighbour residuals on the union of two windows, which
changes them slightly near the edges. With [`ClusterVariance`](@ref), the vectors
hold sums over clusters, so that estimates whose windows share a cluster are
correlated; nprobust's `covgrid` ignores the clusters off the diagonal.
Unidentified estimates have `NaN` rows and columns.
"""
function StatsAPI.vcov(m::LocalPolynomialRegression; method::InferenceMethod = Conventional())
  vs = _pointwise(j -> _localfit(m.data, m.kernel, m.points[j], m.h[j], m.b[j], m.degree, m.derivative,
                                 m.vce; vectors = true, compat = m.compat), eachindex(m.points))
  bc = _usesbc(method)
  vals = [bc ? v.arb : v.aus for v in vs]
  # with clusters, the vectors hold the sums over the clusters in the windows
  C = m.vce isa ClusterVariance ? _clustergram([v.keys for v in vs], vals) : _gram([v.range for v in vs], vals)
  se = bc ? m.se_rbc : m.se
  for j in eachindex(se)
    isnan(se[j]) || continue
    for l in axes(C, 2)
      C[j, l] = C[l, j] = NaN
    end
  end
  return C
end

# C[j, l] = ∑ᵢ A[j][i] A[l][i] over the overlap of the index ranges of j and l
function _gram(ranges, vals)
  G = length(ranges)
  C = zeros(G, G)
  for j in 1:G, l in j:G
    rj, rl = ranges[j], ranges[l]
    lo, hi = max(first(rj), first(rl)), min(last(rj), last(rl))
    s = 0.0
    for i in lo:hi
      s += vals[j][i - first(rj) + 1] * vals[l][i - first(rl) + 1]
    end
    C[j, l] = C[l, j] = s
  end
  return C
end

# For densities, the estimate at point j is ∑ᵢ wᵢ gᵢⱼ / W with gᵢⱼ = k((xⱼ - xᵢ)/h)/h
# (or M for the bias-corrected one), and its plug-in variance is
# ∑ᵢ wᵢ² (gᵢⱼ - f̂ⱼ)² / W². The covariances are the matching inner products,
# (∑ wᵢ² gᵢⱼ gᵢₗ - f̂ₗ ∑ wᵢ² gᵢⱼ - f̂ⱼ ∑ wᵢ² gᵢₗ + f̂ⱼ f̂ₗ ∑ wᵢ²) / W², where the first
# sum runs over the overlap of the two windows.
function StatsAPI.vcov(m::PointwiseDensity; method::InferenceMethod = Conventional())
  d, k = m.data, m.kernel
  c0, c2 = _secondderivative(k)
  κ = m.compat ? (c0 * moment(k, 4) + c2 * moment(k, 6)) / 24 : moment(k, 2) / 2
  bc = _usesbc(_densitymethod(method))
  vs = _pointwise(eachindex(m.points)) do j
    _densityvector(d, k, m.points[j], m.h[j], m.b[j], c0, c2, κ, bc, m.compat)
  end
  W, sw2 = d.sumw, d.sumw^2 / d.neff
  C = _gram([v.range for v in vs], [v.wg for v in vs])
  est, se = bc ? (m.estimate_bc, m.se_rbc) : (m.estimate, m.se)
  for j in axes(C, 1), l in axes(C, 2)
    C[j, l] = (C[j, l] - est[l] * vs[j].s1 - est[j] * vs[l].s1 + est[j] * est[l] * sw2) / W^2
    # an estimate without an observation of nonzero weight has no variance (see fit)
    isnan(se[j]) | isnan(se[l]) && (C[j, l] = NaN)
  end
  return C
end

# wᵢ gᵢⱼ over the window of point j, and s1 = ∑ wᵢ² gᵢⱼ
function _densityvector(d, k, x0, h, b, c0, c2, κ, bc, compat = false)
  xs = d.x
  ρ = h / b
  r = _densitywindow(d, k, x0, h, bc ? ρ : 1.0)
  a, e = first(r), last(r)
  wg = Vector{Float64}(undef, length(r))
  s1 = _densityvector!(wg, d, k, x0, h, ρ, r, c0, c2, κ, bc, Val(false))
  if bc && !compat && (!isfinite(s1) || _biascare(d, k, x0, h, ρ, r, c0, c2, κ))
    s1 = _densityvector!(wg, d, k, x0, h, ρ, r, c0, c2, κ, bc, Val(true))
  end
  return (range = a:e, wg = wg, s1 = s1)
end

# wg and s1 over the window r, with the bias terms as products, or with
# careful = true from _densitybias, as in _densitysums
function _densityvector!(wg, d, k, x0, h, ρ, r, c0, c2, κ, bc, ::Val{careful}) where {careful}
  xs, a = d.x, first(r)
  s1 = 0.0
  for i in r
    u = (x0 - xs[i]) / h
    v = ρ * u
    g = if careful
      (bc ? k(u) - _densitybias(k, v, ρ, c0, c2, κ) : k(u)) / h
    else
      (bc ? k(u) - ρ^3 * κ * k(v) * (c0 + c2 * v^2) : k(u)) / h
    end
    w = _weight(d.w, i)
    wg[i - a + 1] = w * g
    s1 += w^2 * g
  end
  return s1
end

"""
    confband(m; level = 0.95, method = RobustBiasCorrected(), draws = 10_000, rng = Random.default_rng())

A uniform confidence band over the points of a pointwise fit `m`
([`LocalPolynomialRegression`](@ref) or [`PointwiseDensity`](@ref)): the intervals
`estimate ± c se` at all points, with the critical value `c` chosen so that they
cover the curve at all points at once with probability `level`. As in the sup-t
bands of Chernozhukov, Chetverikov and Kato (2014) and lpdensity, `c` is the
`level` quantile of `maxⱼ |Zⱼ|` for a Gaussian vector `Z` with the correlations of
[`vcov`](@ref)`(m; method)`, from `draws` simulated draws. The critical value lies
between the pointwise one and the Bonferroni one, and the simulated quantile is
kept between them, since it can cross them by chance; it is close to the pointwise
one when neighbouring estimates are highly correlated, and equals it at one point.

Returns a named tuple with the `lower` and `upper` ends of the band and the
critical value `critical`. By default the band is robust bias-corrected, centred at
the bias-corrected estimates; with `method = Conventional()` it ignores the bias.
"""
confband(m::_PointwiseFit; kwargs...) = _confband(m; kwargs...)

function _confband(m; level::Real = 0.95, method::InferenceMethod = RobustBiasCorrected(),
                   draws::Integer = 10_000, rng::AbstractRNG = Random.default_rng())
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  draws >= 100 || throw(ArgumentError("draws must be at least 100, got $draws"))
  method isa Union{Conventional,RobustBiasCorrected} ||
    throw(ArgumentError("uniform bands are available for Conventional() and RobustBiasCorrected()"))
  C = vcov(m; method)
  se = sqrt.(max.(LinearAlgebra.diag(C), 0.0))
  c = _supt(C, se, level, draws, rng)
  est = _usesbc(method) ? m.estimate_bc : m.estimate
  return (lower = _lowerend(m, est .- c .* se), upper = est .+ c .* se, critical = c)
end

"""
    confband(f::KDE, x; level = 0.95, method = RobustBiasCorrected(), draws = 10_000, rng)

A uniform confidence band for the density at the points `x`, with the bandwidth of
the estimate `f`; see [`confband`](@ref) for pointwise fits.
"""
function confband(f::KDE, x::AbstractVector{<:Real}; kwargs...)
  m = fit(PointwiseDensity, f.data; points = x, kernel = f.kernel, bandwidth = f.h, minobs = 0)
  return confband(m; kwargs...)
end

# The level quantile of maxⱼ |Zⱼ| for Z with the correlations of C, by simulation,
# kept between the pointwise and the Bonferroni critical values, between which the
# exact one lies and which a simulated quantile can cross with few or highly
# correlated points; clamping can only bring it closer. Points with zero variance
# are left out, and with one point the pointwise value is exact.
function _supt(C, se, level, draws, rng)
  zp = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  keep = findall(>(0), se)
  length(keep) <= 1 && return zp
  R = C[keep, keep] ./ (se[keep] * se[keep]')
  E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(R))
  L = E.vectors .* sqrt.(max.(E.values, 0.0))'   # R = L L'
  G = length(keep)
  # in blocks of draws, with the normal variates of a block drawn at once and one
  # matrix product each
  B = min(draws, max(1, 2^16 ÷ G))
  Ξ, Z = Matrix{Float64}(undef, G, B), Matrix{Float64}(undef, G, B)
  maxima = Vector{Float64}(undef, draws)
  for t0 in 1:B:draws
    nb = min(B, draws - t0 + 1)
    nb < B && ((Ξ, Z) = (Matrix{Float64}(undef, G, nb), Matrix{Float64}(undef, G, nb)))
    Random.randn!(rng, Ξ)
    LinearAlgebra.mul!(Z, L, Ξ)
    for t in 1:nb
      maxima[t0 + t - 1] = maximum(abs, view(Z, :, t))
    end
  end
  zb = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / (2G))
  return clamp(Distributions.quantile(maxima, level), zp, zb)
end
