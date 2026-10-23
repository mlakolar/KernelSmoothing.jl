## Local polynomial density estimation (Cattaneo, Jansson and Ma 2020)

"""
    LocalPolynomialDensity <: StatisticalModel

Local polynomial density estimates (Cattaneo, Jansson and Ma 2020) at a set of
points, with conventional and robust bias-corrected inference, fitted with
[`fit`](@ref)`(LocalPolynomialDensity, x)`, as R's lpdensity computes them.

At a point `x₀`, the empirical distribution function `F̂`, with tied observations
at the value at the end of their run, is regressed on a polynomial of order `p` in
`(x - x₀)/h` with kernel weights over the distinct values. The coefficient of
order `ν + 1` estimates the derivative `f⁽ᵛ⁾(x₀)` of the density. Since the fit
adapts to the data near `x₀`, the estimate needs no knowledge of a boundary and
keeps its rate there, where kernel density estimates are inconsistent. The
bias-corrected estimate is the fit of order `p + 1` with the same bandwidth. The
standard errors come from the influence function of `F̂` on the fit, so that they
account for its estimation.

`coef`, `stderror(m; method)`, `confint(m; level, method)`, `coeftable(m)`,
[`vcov`](@ref) and [`confband`](@ref) work as for [`PointwiseDensity`](@ref).
"""
struct LocalPolynomialDensity{K<:AbstractKernel,D<:SmoothingData,S<:BandwidthSelection} <: StatsAPI.StatisticalModel
  data::D
  kernel::K
  degree::Int
  derivative::Int        # of the density; 0 for the density itself
  selection::S
  minobs::Int
  points::Vector{Float64}
  h::Vector{Float64}
  b::Vector{Float64}     # the bias correction uses the same bandwidth
  estimate::Vector{Float64}
  se::Vector{Float64}
  estimate_bc::Vector{Float64}
  se_rbc::Vector{Float64}
  nwindow::Vector{Int}
  nunique::Vector{Int}
  floored::BitVector
end

"""
    fit(LocalPolynomialDensity, x; points = nothing, degree = 2, derivative = 0,
        kernel = TriangularKernel(), bandwidth = NormalReference(), minobs = degree + 21, weights = nothing)
    fit(LocalPolynomialDensity, d::SmoothingData; kwargs...)

Estimates the density of `x`, or its derivative of order `derivative`, at `points`
(by default the quantiles of order 0.05, 0.10, …, 0.95) with local polynomial fits
of order `degree` to the empirical distribution function, and the robust
bias-corrected estimates of order `degree + 1`.

- `bandwidth` is a number, a vector with one bandwidth per point, or
  `NormalReference()`, the rule of thumb of lpdensity (`bwselect = "mse-rot"`): the
  bandwidth that minimizes the asymptotic mean squared error at each point when
  the density is normal.
- `minobs` raises the bandwidth at each point, where needed, so that the window
  holds `minobs` observations and `minobs` distinct values, as lpdensity's
  `nLocalMin` and `nUniqueMin`; `minobs = 0` turns this off.

The kernel must be compactly supported; lpdensity uses the triangular,
Epanechnikov and uniform kernels. The estimates agree with lpdensity 3.0.1 with
`massPoints = TRUE`.
"""
function fit(::Type{LocalPolynomialDensity}, x::AbstractVector{<:Real}; weights = nothing, kwargs...)
  return fit(LocalPolynomialDensity, SmoothingData(x; weights); kwargs...)
end

function fit(::Type{LocalPolynomialDensity}, d::SmoothingData; points = nothing, degree::Integer = 2,
             derivative::Integer = 0, kernel::AbstractKernel = TriangularKernel(), bandwidth = NormalReference(),
             minobs::Integer = degree + 21)
  p, v = Int(degree), Int(derivative) + 1
  derivative >= 0 || throw(ArgumentError("derivative must be nonnegative, got $derivative"))
  v <= p || throw(ArgumentError("the derivative of order $derivative needs degree at least $v, got $p"))
  all(isfinite, support(kernel)) || throw(ArgumentError("$kernel must be compactly supported"))
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  pts = points === nothing ? [_quantile(d, q) for q in 0.05:0.05:0.95] : _checkpoints(points)
  sel = _lpdensity_selection(bandwidth, d, kernel, pts, p, v, minobs)
  h = _pointbandwidths(sel.h, length(pts))
  floored = _floor!(h, copy(h), d, pts, minobs)
  if minobs > 0   # also enough distinct values
    for j in eachindex(pts)
      δ = knn_distance(d, pts[j], min(minobs, nunique(d)); distinct = true)
      δ > h[j] && (h[j] = δ; floored[j] = true)
    end
  end
  aux = _lpdaux(d)
  r = _pointwise(j -> _lpdfit(d, kernel, aux, pts[j], h[j], p, v), eachindex(pts))
  return LocalPolynomialDensity(d, kernel, p, v - 1, sel, Int(minobs), pts, h, copy(h),
    [t.estimate for t in r], [t.se for t in r], [t.estimate_bc for t in r], [t.se_rbc for t in r],
    [t.nwin for t in r], [t.nuniq for t in r], floored)
end

_lpdensity_selection(h::Union{Real,AbstractVector{<:Real}}, d, k, pts, p, v, minobs) =
  _regression_selection(h, d, k, 0, 0, pts, nothing, minobs)
_lpdensity_selection(rule::NormalReference, d, k, pts, p, v, minobs) = _lpdensity_rot(rule, d, k, pts, p, v, minobs)
_lpdensity_selection(rule, d, k, pts, p, v, minobs) = throw(ArgumentError(
  "LocalPolynomialDensity takes a number, a vector or NormalReference() as bandwidth, got $(repr(rule))"))

# For each distinct value g: its total weight Pg normalized so that all weights sum
# to n, the empirical distribution function at it, and the sum of the squared
# normalized weights of its observations
function _lpdaux(d::SmoothingData)
  n, W, U = nobs(d), d.sumw, nunique(d)
  P, F, Q = zeros(U), zeros(U), zeros(U)
  for g in 1:U
    s, e = d.starts[g], d.starts[g + 1] - 1
    P[g] = (d.cumw[e + 1] - d.cumw[s]) * n / W
    F[g] = d.cumw[e + 1] / W
    Q[g] = d.w === nothing ? Float64(e - s + 1) : sum(abs2, view(d.w, s:e)) * (n / W)^2
  end
  return (P = P, F = F, Q = Q, cumQ = [0.0; cumsum(Q)])
end

# The fits of orders p and q = p + 1 at x0 over the distinct values within h, with
# S = (1/n) ∑ Pg K(tg) r(tg) r(tg)' and T = (1/n) ∑ Pg K(tg) r(tg) F̂g, tg = (ug - x0)/h,
# K including the factor 1/h. The influence of observation i, in group g(i), on
# the coefficient vector is S⁻¹ (Rg(i)/n - T) times its normalized weight, with
# Rg = ∑ over the groups g' ≥ g of Pg' K(tg') r(tg'), so that
# se² = ν!² ∑ᵢ (a'(Rg(i)/n - T))² wᵢ² / (n² h^(2ν)) with a = S⁻¹ e_ν.
function _lpdfit(d::SmoothingData, k, aux, x0, h, p, v; values::Bool = false)
  u, U, n = d.values, nunique(d), nobs(d)
  ga = _firsttrue(g -> u[g] - x0 >= -h, U)
  gb = _firsttrue(g -> u[g] - x0 > h, U) - 1
  nwin = ga <= gb ? d.starts[gb + 1] - d.starts[ga] : 0
  nuniq = max(gb - ga + 1, 0)
  q = p + 1
  s, t = zeros(2q + 1), zeros(q + 1)
  for g in ga:gb
    tg = (u[g] - x0) / h
    _powersums!(s, t, aux.P[g] * k(tg) / h / n, tg, aux.F[g])
  end
  function order(o)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric([s[i + j + 1] for i in 0:o, j in 0:o]); check = false)
    LinearAlgebra.issuccess(C) || return (est = NaN, se = NaN, vals = Float64[], aT = NaN, Atot = NaN)
    β = C \ t[1:o+1]
    a = C \ [i == v ? 1.0 : 0.0 for i in 0:o]
    aT = LinearAlgebra.dot(a, t[1:o+1])
    # a'Rg for the groups in the window, by a reverse cumulative sum
    vals = zeros(max(gb - ga + 1, 0))
    acc = 0.0
    for g in gb:-1:ga
      tg = (u[g] - x0) / h
      acc += aux.P[g] * k(tg) / h * evalpoly(tg, a)
      vals[g - ga + 1] = acc / n - aT
    end
    Atot = acc
    # the groups below the window have Rg = R over the window, those above Rg = 0
    ss = aux.cumQ[ga] * (Atot / n - aT)^2 + (aux.cumQ[U + 1] - aux.cumQ[gb + 1]) * aT^2
    for g in ga:gb
      ss += aux.Q[g] * vals[g - ga + 1]^2
    end
    scale = factorial(v) / h^v
    return (est = scale * β[v + 1], se = scale * sqrt(ss) / n, vals = vals, aT = aT, Atot = Atot)
  end
  fp, fq = order(p), order(q)
  r = (estimate = fp.est, se = fp.se, estimate_bc = fq.est, se_rbc = fq.se, nwin = nwin, nuniq = nuniq)
  values || return r
  return (r..., ga = ga, gb = gb, p = fp, q = fq)
end

nobs(m::LocalPolynomialDensity) = nobs(m.data)
bandwidth(m::LocalPolynomialDensity) = m.h
StatsAPI.coef(m::LocalPolynomialDensity) = copy(m.estimate)
stderror(m::LocalPolynomialDensity; method::InferenceMethod = Conventional()) =
  copy(_usesbc(method) ? m.se_rbc : m.se)
StatsAPI.confint(m::LocalPolynomialDensity; level::Real = 0.95, method::InferenceMethod = RobustBiasCorrected()) =
  _confint(m, level, method)

function StatsAPI.coeftable(m::LocalPolynomialDensity; level::Real = 0.95)
  c = confint(m; level, method = Conventional())
  r = confint(m; level, method = RobustBiasCorrected())
  return (point = copy(m.points), h = copy(m.h), nobs = copy(m.nwindow), nunique = copy(m.nunique),
          floored = copy(m.floored), estimate = copy(m.estimate), se = copy(m.se), lower = c[:, 1],
          upper = c[:, 2], estimate_bc = copy(m.estimate_bc), se_rbc = copy(m.se_rbc), lower_rbc = r[:, 1],
          upper_rbc = r[:, 2])
end

# Covariances from the influence functions: ν!² ∑g Qg valsⱼ(g) valsₗ(g) / (n² hⱼ^ν hₗ^ν)
function StatsAPI.vcov(m::LocalPolynomialDensity; method::InferenceMethod = Conventional())
  d, U, n, v = m.data, nunique(m.data), nobs(m.data), m.derivative + 1
  aux = _lpdaux(d)
  bc = _usesbc(method)
  dense = _pointwise(eachindex(m.points)) do j
    r = _lpdfit(d, m.kernel, aux, m.points[j], m.h[j], m.degree, v; values = true)
    f = bc ? r.q : r.p
    val = zeros(U)
    val[1:r.ga-1] .= f.Atot / n - f.aT
    val[r.ga:r.gb] .= f.vals
    val[r.gb+1:U] .= -f.aT
    val .* (factorial(v) / m.h[j]^v)
  end
  G = length(m.points)
  C = zeros(G, G)
  for j in 1:G, l in j:G
    C[j, l] = C[l, j] = sum(aux.Q .* dense[j] .* dense[l]) / n^2
  end
  return C
end

function Base.show(io::IO, m::LocalPolynomialDensity)
  print(io, "LocalPolynomialDensity(degree ", m.degree, ", ", m.kernel, ", ", length(m.points), " points, ",
        nobs(m), " observations)")
end

function Base.show(io::IO, ::MIME"text/plain", m::LocalPolynomialDensity)
  println(io, "Local polynomial density estimates of degree ", m.degree,
          m.derivative == 0 ? "" : ", derivative $(m.derivative)")
  println(io, "  kernel: ", m.kernel, ", ", nobs(m), " observations")
  m.selection.rule === nothing || println(io, "  bandwidth: ", m.selection.rule)
  any(m.floored) && println(io, "  the floor of ", m.minobs, " observations raised the bandwidth at ",
                            count(m.floored), " of ", length(m.points), " points")
  _showtable(io, m)
end


## lpdensity's rule of thumb

# The bandwidth that minimizes a^(2p+2-2ν) (B₁ + a B₂)² + σ²/a^(2ν-1) at each point
# when the data are normal with their mean and standard deviation, with the
# constants of the equivalent kernel on [-1, 1] (lpdensity's bw_ROT).
function _lpdensity_rot(rule, d::SmoothingData, k, pts, p, v, minobs)
  k isa PolynomialKernel || throw(ArgumentError("the rule of thumb needs a polynomial kernel, got $k"))
  n = nobs(d)
  x̄ = sum(d.x) / n
  sdx = sqrt(sum(abs2, d.x .- x̄) / (n - 1))
  z = (d.x .- x̄) ./ sdx
  W = d.sumw
  μ = sum(_weight(d.w, i) * z[i] for i in 1:n) / W
  σ = sqrt(sum(_weight(d.w, i) * (z[i] - μ)^2 for i in 1:n) / W)
  S = [moment(k, i + j) for i in 0:p, j in 0:p]
  C1 = [moment(k, i + p + 1) for i in 0:p]
  C2 = [moment(k, i + p + 2) for i in 0:p]
  G = _cjmG(k, p)
  Si = inv(S)
  e = v + 1
  c1, c2 = (Si * C1)[e], (Si * C2)[e]
  vfac = factorial(v)
  sdconst = sqrt(abs((Si * G * Si)[e, e]))
  # derivatives of the normal density with mean μ and standard deviation σ
  φr(x, r) = (-1)^r * _hermite(r, (x - μ) / σ) * exp(-((x - μ) / σ)^2 / 2) / (sqrt(2π) * σ^(r + 1))
  range_z = z[end] - z[1]
  h = map(pts) do x0
    z0 = (x0 - x̄) / sdx
    b1 = φr(z0, p) / factorial(p + 1) * vfac
    b2 = (φr(z0, p + 1) / factorial(p + 2) * vfac + b1 * φr(z0, 1) / φr(z0, 0)) * c2
    b1 *= c1
    sd2 = (vfac * sqrt(φr(z0, 0) / n) * sdconst)^2
    hz = _fmin(a -> a^(2p + 2 - 2v) * (b1 + a * b2)^2 + sd2 / a^(2v - 1), eps(), range_z)
    hx = hz * sdx
    if minobs > 0
      hx = max(hx, knn_distance(d, x0, min(minobs, n)), knn_distance(d, x0, min(minobs, nunique(d)); distinct = true))
    end
    min(hx, max(abs(d.values[1] - x0), abs(d.values[end] - x0)))
  end
  return BandwidthSelection(h, rule, (points = pts,))
end

# G[i, j] = ∫∫_{x<y} x^i y^(j-1) k(x) k(y) + ∫∫_{x>y} x^(i-1) y^j k(x) k(y), i, j = 1, …, p + 1,
# by Gauss–Legendre quadrature on the pieces of the kernel, exact for these polynomials
function _cjmG(k, p)
  breaks = collect(first(pieces(k)))
  xs, ws = _gausslegendre(24)
  # ∫ f over [a, b], split at the breakpoints of the kernel
  function integral(f, a, b)
    a >= b && return 0.0
    cuts = [a; [t for t in breaks if a < t < b]; b]
    s = 0.0
    for c in 1:length(cuts)-1
      m, r = (cuts[c] + cuts[c + 1]) / 2, (cuts[c + 1] - cuts[c]) / 2
      s += r * sum(ws[l] * f(m + r * xs[l]) for l in eachindex(xs))
    end
    return s
  end
  a0, b0 = first(breaks), last(breaks)
  G = zeros(p + 1, p + 1)
  for i in 1:p+1, j in 1:p+1
    G[i, j] = integral(y -> y^(j - 1) * k(y) * integral(x -> x^i * k(x), a0, y), a0, b0) +
              integral(y -> y^j * k(y) * integral(x -> x^(i - 1) * k(x), y, b0), a0, b0)
  end
  return G
end

# nodes and weights of the m-point Gauss–Legendre rule on [-1, 1] (Golub and Welsch)
function _gausslegendre(m)
  β = [l / sqrt(4l^2 - 1) for l in 1:m-1]
  E = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(m), β))
  return E.values, 2 .* E.vectors[1, :] .^ 2
end

confband(m::LocalPolynomialDensity; kwargs...) = _confband(m; kwargs...)
_confint(m::LocalPolynomialDensity, level, method::Honest) =
  throw(ArgumentError("honest intervals are implemented for local polynomial regression"))
