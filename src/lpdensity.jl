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
[`vcov`](@ref) and [`confband`](@ref) work as for [`PointwiseDensity`](@ref). The
columns `nobs` and `nunique` of `coeftable` count the observations and the distinct
values with positive weight in the window of each point, or with `compat = true`
all those within `h`, as lpdensity's `nh` and `nhu`.
"""
struct LocalPolynomialDensity{K<:AbstractKernel,D<:SmoothingData,S<:BandwidthSelection} <: StatsAPI.StatisticalModel
  data::D
  kernel::K
  degree::Int
  derivative::Int        # of the density; 0 for the density itself
  selection::S
  minobs::Int
  compat::Bool           # lpdensity's rules, points, counts and intervals
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
        kernel = TriangularKernel(), bandwidth = MSEOptimal(), minobs = degree + 21, weights = nothing,
        compat = false)
    fit(LocalPolynomialDensity, d::SmoothingData; kwargs...)

Estimates the density of `x`, or its derivative of order `derivative`, at `points`
(by default the quantiles of order 0.05, 0.10, …, 0.95, weighted for weighted
data) with local polynomial fits of order `degree` to the empirical distribution
function, and the robust bias-corrected estimates of order `degree + 1`. The
estimates are `NaN` at points outside the range of the data whose nearest
observation lies more than half a bandwidth away, where the fits extrapolate; a
natural boundary just beyond the data, as 0 for positive data, keeps its estimate.

- `bandwidth` is a number, a vector with one bandwidth per point, or a rule of
  lpdensity: `MSEOptimal()`, its plug-in estimate of the bandwidth that minimizes
  the asymptotic mean squared error at each point (`bwselect = "mse-dpi"`, the
  default, which `nothing` also selects); `IMSEOptimal()`, one plug-in bandwidth
  that minimizes the sum of these errors over the points (`"imse-dpi"`), or with
  `grid = m` over `m` equally spaced points between the smallest and the largest
  observation; or `NormalReference()`, the rule of thumb for normal data
  (`"mse-rot"`). The plug-in rules estimate the derivatives of orders `degree + 1`
  and `degree + 2` of the distribution function with local fits at rule-of-thumb
  pilot bandwidths, and the constants of the bias and the variance from the data.
- `minobs` raises selected bandwidths at each point, where needed, so that the
  window holds `minobs` observations and `minobs` distinct values, as lpdensity's
  `nLocalMin` and `nUniqueMin`; `minobs = 0` turns this off. The rules apply the
  floor before they cap the bandwidth at the distance to the farthest distinct
  value, as lpdensity does, and `IMSEOptimal` applies the largest floor over the
  evaluation points to its single bandwidth. Given bandwidths are not raised, as in
  lpdensity, and the fit warns where they fall below the floor.

The rules depart from lpdensity's in three ways. First, they standardize the data
by the median and the robust scale of [`NormalReference`](@ref) instead of the mean
and the standard deviation, so that one outlier does not inflate every bandwidth;
`NormalReference(robust = false)` keeps the standard deviation. Second, the rules
of thumb for the pilot bandwidths are integrated over the quantiles of the data and
floored at each point, while lpdensity integrates them over the evaluation points,
so that one far point changes the bandwidths at all the others. Third, the plug-in
rules bound the squared bias by `(|B₁| + a |B₂|)²`, since lpdensity's
`(B₁ + a B₂)²` vanishes at `a = -B₁/B₂` when the estimated terms have opposite
signs, and the rules of thumb add the next term of the bias, which keeps it from
vanishing at the inflection points of the normal density, where lpdensity's
bandwidth jumps to the range of the data. The lower ends of the intervals for the
density itself are clamped at zero.

With `compat = true`, the fit agrees with lpdensity 3.0.1 with `massPoints = TRUE`:
the default points are unweighted quantiles, points outside the data are
extrapolated, the rules are lpdensity's, plug-in fits whose matrix R's `solve`
rejects as ill-conditioned fall back to the floors, and intervals are not clamped.
The kernel must be compactly supported and nonnegative; lpdensity uses the
triangular, Epanechnikov and uniform kernels.
"""
function fit(::Type{LocalPolynomialDensity}, x::AbstractVector; weights = nothing, kwargs...)
  return fit(LocalPolynomialDensity, SmoothingData(x; weights); kwargs...)
end

function fit(::Type{LocalPolynomialDensity}, d::SmoothingData; points = nothing, degree::Integer = 2,
             derivative::Integer = 0, kernel::AbstractKernel = TriangularKernel(), bandwidth = MSEOptimal(),
             minobs::Integer = degree + 21, compat::Bool = false)
  d = _withouty(d)
  p, v = Int(degree), Int(derivative) + 1
  derivative >= 0 || throw(ArgumentError("derivative must be nonnegative, got $derivative"))
  v <= p || throw(ArgumentError("the derivative of order $derivative needs degree at least $v, got $p"))
  all(isfinite, support(kernel)) || throw(ArgumentError("$kernel must be compactly supported"))
  _isnonnegative(kernel) || throw(ArgumentError("$kernel takes negative values; the local fits need " *
    "nonnegative kernel weights"))
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  pts = points === nothing ? _lpdpoints(d, compat) : _checkpoints(points)
  sel = _anyselection(_lpdensity_selection(bandwidth, d, kernel, pts, p, v, minobs; compat))
  h = _pointbandwidths(sel.h, length(pts))
  # the floor raises selected bandwidths, and the flags also mark the points where it
  # decided one; given bandwidths are kept, as in lpdensity
  floored = falses(length(pts))
  if minobs > 0 && bandwidth isa Union{Real,AbstractVector{<:Real}}
    low = count(j -> _lpdfloor(d, pts[j], minobs) > h[j], eachindex(pts))
    low > 0 && @warn "LocalPolynomialDensity: the given bandwidth holds fewer than $minobs observations " *
      "or distinct values at $low of $(length(pts)) points, where the estimates vary more; the rules would " *
      "raise it there, and minobs = 0 turns this warning off"
  elseif minobs > 0
    for j in eachindex(pts)
      δ = _lpdfloor(d, pts[j], minobs)
      δ >= h[j] && (h[j] = δ; floored[j] = true)
    end
  end
  aux = _lpdaux(d)
  r = _pointwise(j -> _lpdfit(d, kernel, aux, pts[j], h[j], p, v; compat, inside = _lpdinside(d, pts[j], h[j], compat)),
                 eachindex(pts))
  return LocalPolynomialDensity(d, kernel, p, v - 1, sel, Int(minobs), compat, pts, h, copy(h),
    [t.estimate for t in r], [t.se for t in r], [t.estimate_bc for t in r], [t.se_rbc for t in r],
    [t.nwin for t in r], [t.nuniq for t in r], floored)
end

_lpdensity_selection(h::Union{Real,AbstractVector{<:Real}}, d, k, pts, p, v, minobs; compat = false) =
  _regression_selection(h, d, k, 0, 0, pts, nothing, minobs)
_lpdensity_selection(rule::NormalReference, d, k, pts, p, v, minobs; compat = false) =
  _lpdensity_rot(rule, d, k, pts, p, v, minobs; robust = rule.robust && !compat, guard = !compat)
_lpdensity_selection(rule::Union{MSEOptimal,IMSEOptimal}, d, k, pts, p, v, minobs; compat = false) =
  _lpdensity_dpi(rule, d, k, pts, p, v, minobs, compat)
_lpdensity_selection(::Nothing, d, k, pts, p, v, minobs; compat = false) =   # the default, as in the other fits
  _lpdensity_dpi(MSEOptimal(), d, k, pts, p, v, minobs, compat)
_lpdensity_selection(rule, d, k, pts, p, v, minobs; compat = false) = throw(ArgumentError(
  "LocalPolynomialDensity takes a number, a vector, MSEOptimal(), IMSEOptimal() or NormalReference() as " *
  "bandwidth, got $(repr(rule))"))

# the smallest bandwidth whose window at x0 holds minobs observations and minobs distinct values
_lpdfloor(d, x0, minobs) = max(knn_distance(d, x0, min(minobs, nobs(d))),
                               knn_distance(d, x0, min(minobs, nunique(d)); distinct = true))

# lpdensity's default points, the quantiles of order 0.05, …, 0.95: unweighted under
# compat, as in lpdensity, and otherwise weighted, as suits weighted data
_lpdpoints(d, compat) = compat ? [_unweightedquantile(d.x, q) for q in 0.05:0.05:0.95] :
  [_quantile(d, q) for q in 0.05:0.05:0.95]

# Whether the fit at x0 with bandwidth h is reported: unless compat, a point outside
# the range of the data whose nearest observation lies more than h/2 away is NaN,
# since most of its window is empty and the fit extrapolates (66 ± 106 at -1 for
# exponential data). A natural boundary just beyond the extreme observation, as 0
# for positive data, stays. At the floor of minobs observations this asks that the
# gap be at most the span of the extreme observations; a rule by that span alone
# made the boundary of Beta(2, 2), where the density vanishes, NaN in 0.4 percent
# of the samples.
_lpdinside(d, x0, h, compat) = compat || d.x[1] - h / 2 <= x0 <= d.x[end] + h / 2

# For each distinct value g: its total weight Pg normalized so that all weights sum
# to n, the empirical distribution function at it, and the sum of the squared
# normalized weights of its observations
function _lpdaux(d::SmoothingData)
  n, W, U = nobs(d), d.sumw, nunique(d)
  P, F, Q = zeros(U), zeros(U), zeros(U)
  for g in 1:U
    s, e = d.starts[g], d.starts[g + 1] - 1
    P[g] = _wsum(d, s, e) * n / W
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
# The counts are those of observations and distinct values with positive kernel
# weight, or with compat all those within h, as lpdensity's; with inside = false
# the estimates are NaN. Under compat, a fit whose matrix R's solve rejects is NaN,
# and so is the bias correction where the fit of order p is, as in lpdensity.
function _lpdfit(d::SmoothingData, k, aux, x0, h, p, v; values::Bool = false, compat::Bool = false,
                 inside::Bool = true)
  u, U, n = d.values, nunique(d), nobs(d)
  ga = _firsttrue(g -> u[g] - x0 >= -h, U)
  gb = _firsttrue(g -> u[g] - x0 > h, U) - 1
  q = p + 1
  s, t = zeros(2q + 1), zeros(q + 1)
  count = kcount = kobs = 0
  for g in ga:gb
    tg = (u[g] - x0) / h
    kt = k(tg)
    kg = aux.P[g] * kt / h / n
    count += kg > 0
    kcount += kt > 0
    kobs += ifelse(kt > 0, d.starts[g + 1] - d.starts[g], 0)
    _powersums!(s, t, kg, tg, aux.F[g])
  end
  nwin = !compat ? kobs : ga <= gb ? d.starts[gb + 1] - d.starts[ga] : 0
  nuniq = !compat ? kcount : max(gb - ga + 1, 0)
  npos = count   # distinct values with positive weight, assigned once for the closure
  function order(o)
    G = [s[i + j + 1] for i in 0:o, j in 0:o]
    C, ok = _gramfactor(G, npos)
    ok && (!compat || _rsolvable(G)) || return (est = NaN, se = NaN, vals = Float64[], aT = NaN, Atot = NaN)
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
  nan = (est = NaN, se = NaN, vals = Float64[], aT = NaN, Atot = NaN)
  fp = inside ? order(p) : nan
  fq = inside && !(compat && isnan(fp.est)) ? order(q) : nan
  r = (estimate = fp.est, se = fp.se, estimate_bc = fq.est, se_rbc = fq.se, nwin = nwin, nuniq = nuniq)
  values || return r
  return (r..., ga = ga, gb = gb, p = fp, q = fq, s = s, npos = npos)
end

nobs(m::LocalPolynomialDensity) = nobs(m.data)
bandwidth(m::LocalPolynomialDensity) = m.h
StatsAPI.coef(m::LocalPolynomialDensity) = copy(m.estimate)
stderror(m::LocalPolynomialDensity; method::InferenceMethod = Conventional()) =
  copy(_usesbc(_densitymethod(method)) ? m.se_rbc : m.se)
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

# Covariances from the influence functions: ν!² ∑g Qg valsⱼ(g) valsₗ(g) / (n² hⱼ^ν hₗ^ν).
# The influence values of a point are constant below its window and above it, so
# that each covariance is a sum over the windows of the two points and, where both
# are constant, a product of the constants with a cumulative sum of Q. With many
# distinct values this avoids a dense matrix of distinct values by points, five times
# the memory; with few, one product of that matrix is faster.
function StatsAPI.vcov(m::LocalPolynomialDensity; method::InferenceMethod = Conventional())
  d, U, n, v = m.data, nunique(m.data), nobs(m.data), m.derivative + 1
  aux = _lpdaux(d)
  bc = _usesbc(_densitymethod(method))
  infl = _pointwise(eachindex(m.points)) do j
    r = _lpdfit(d, m.kernel, aux, m.points[j], m.h[j], m.degree, v; values = true, compat = m.compat,
                inside = _lpdinside(d, m.points[j], m.h[j], m.compat))
    f = bc ? r.q : r.p
    scale = factorial(v) / m.h[j]^v
    # a fit that is not identified has no influence function
    (ok = !isnan(f.est), ga = r.ga, gb = r.gb, lo = (f.Atot / n - f.aT) * scale, hi = -f.aT * scale,
     w = f.vals .* scale)
  end
  M = length(infl)
  if U * M <= 2^18
    dense = [I.ok ? [g < I.ga ? I.lo : g > I.gb ? I.hi : I.w[g - I.ga + 1] for g in 1:U] : fill(NaN, U) for I in infl]
    S = sqrt.(aux.Q) .* reduce(hcat, dense)
    return (S' * S) ./ n^2
  end
  C = Matrix{Float64}(undef, M, M)
  cuts = Vector{Int}(undef, 6)
  for j in 1:M, l in j:M
    C[j, l] = C[l, j] = _lpdcov(infl[j], infl[l], aux, U, cuts) / n^2
  end
  return C
end

# ∑g Qg vA(g) vB(g) over the distinct values, split where either point enters or leaves
# its window
function _lpdcov(A, B, aux, U, cuts)
  A.ok && B.ok || return NaN
  Q, wA, wB = aux.Q, A.w, B.w
  oA, oB = A.ga - 1, B.ga - 1
  cuts .= (1, A.ga, A.gb + 1, B.ga, B.gb + 1, U + 1)
  sort!(cuts; alg = InsertionSort)
  s = 0.0
  for c in 1:length(cuts)-1
    g1, g2 = cuts[c], cuts[c + 1] - 1
    g1 <= g2 || continue
    # each stretch lies inside a window or outside it, below or above
    inA, inB = A.ga <= g1 && g2 <= A.gb, B.ga <= g1 && g2 <= B.gb
    cA, cB = g1 < A.ga ? A.lo : A.hi, g1 < B.ga ? B.lo : B.hi
    t = 0.0
    if inA && inB
      @inbounds for g in g1:g2
        t += Q[g] * wA[g - oA] * wB[g - oB]
      end
    elseif inA
      @inbounds for g in g1:g2
        t += Q[g] * wA[g - oA]
      end
      t *= cB
    elseif inB
      @inbounds for g in g1:g2
        t += Q[g] * wB[g - oB]
      end
      t *= cA
    else
      t = cA * cB * (aux.cumQ[g2 + 1] - aux.cumQ[g1])
    end
    s += t
  end
  return s
end

function Base.show(io::IO, m::LocalPolynomialDensity)
  print(io, "LocalPolynomialDensity(degree ", m.degree, ", ", m.kernel, ", ", length(m.points), " points, ",
        nobs(m), " observations)")
end

function Base.show(io::IO, ::MIME"text/plain", m::LocalPolynomialDensity)
  println(io, "Local polynomial density estimates of degree ", m.degree,
          m.derivative == 0 ? "" : ", derivative $(m.derivative)", m.compat ? " (lpdensity compatibility)" : "")
  println(io, "  kernel: ", m.kernel, ", ", nobs(m), " observations")
  rule = m.selection.rule
  rule === nothing || println(io, "  bandwidth: ", rule, _lpdrulenote(rule, m.compat))
  any(m.floored) && println(io, "  the floor of ", m.minobs, " observations raised the bandwidth at ",
                            count(m.floored), " of ", length(m.points), " points")
  _showtable(io, m)
end


## lpdensity's bandwidth selectors

# how the rules of a fit differ from lpdensity's, for its display
_lpdrulenote(rule, compat) = compat || !(rule isa Union{MSEOptimal,IMSEOptimal,NormalReference}) ? "" :
  rule isa NormalReference ? (rule.robust ? ", robust scale" : "") : ", robust scale, pilots over the quantiles of the data"

# The center and the scale that standardize the data for lpdensity's rules: the
# unweighted mean and standard deviation, as lpdensity, or with `robust` the
# weighted median and the robust scale of NormalReference, which one outlier among
# 2000 normal observations, at 10⁴, does not inflate from 1 to 224
function _lpdscale(d::SmoothingData, robust::Bool)
  if robust
    sd, iqr = _sdiqr(d)
    return _quantile(d, 0.5), _robustscale(sd, iqr / 1.349)
  end
  x̄ = _sortedmean(d.x)
  return x̄, sqrt(sum(abs2, d.x .- x̄) / (nobs(d) - 1))
end

# The terms of lpdensity's rules of thumb (bw_ROT and bw_IROT) at x0, for the data
# standardized by _lpdscale and a normal density with their weighted mean and
# standard deviation, or with `robust` the standard normal: B₁, B₂, B₃ and σ² of
# the asymptotic mean squared error a^(2p+2-2ν) (B₁ + a B₂ + a² B₃)² + σ²/a^(2ν-1),
# with the constants of the equivalent kernel on [-1, 1]. lpdensity has B₃ = 0; with
# `guard`, B₃ is the next term of the bias and all three enter with their absolute
# values, so that the bias does not vanish where B₁ and B₂ do, as for symmetric
# kernels at the inflection points of the normal density.
function _lpdensity_nr(d::SmoothingData, k, p, v; robust::Bool = false, guard::Bool = false)
  k isa PolynomialKernel || throw(ArgumentError("lpdensity's bandwidths need a polynomial kernel, got $k"))
  n = nobs(d)
  center, scale = _lpdscale(d, robust)
  z = (d.x .- center) ./ scale
  W = d.sumw
  μ = robust ? 0.0 : sum(_weight(d.w, i) * z[i] for i in 1:n) / W
  σ = robust ? 1.0 : sqrt(sum(_weight(d.w, i) * (z[i] - μ)^2 for i in 1:n) / W)
  S = [moment(k, i + j) for i in 0:p, j in 0:p]
  C1 = [moment(k, i + p + 1) for i in 0:p]
  C2 = [moment(k, i + p + 2) for i in 0:p]
  C3 = [moment(k, i + p + 3) for i in 0:p]
  Si = inv(S)
  e = v + 1
  c1, c2, c3 = (Si * C1)[e], (Si * C2)[e], (Si * C3)[e]
  vfac = factorial(v)
  sdconst = sqrt(abs((Si * _cjmG(k, p) * Si)[e, e]))
  # derivatives of the normal density with mean μ and standard deviation σ
  φr(x, r) = (-1)^r * _hermite(r, (x - μ) / σ) * exp(-((x - μ) / σ)^2 / 2) / (sqrt(2π) * σ^(r + 1))
  function terms(x0)
    z0 = (x0 - center) / scale
    b1 = φr(z0, p) / factorial(p + 1) * vfac
    b2 = (φr(z0, p + 1) / factorial(p + 2) * vfac + b1 * φr(z0, 1) / φr(z0, 0)) * c2
    V = (vfac * sqrt(φr(z0, 0) / n) * sdconst)^2
    guard || return (B1 = b1 * c1, B2 = b2, B3 = 0.0, V = V)
    b3 = φr(z0, p + 2) / factorial(p + 3) * vfac * c3
    return (B1 = abs(b1 * c1), B2 = abs(b2), B3 = abs(b3), V = V)
  end
  return (terms = terms, sdx = scale, range = z[end] - z[1])
end

# the asymptotic mean squared error at the standardized bandwidth a
_lpdmse(a, p, v, t) = a^(2p + 2 - 2v) * (t.B1 + a * t.B2 + a^2 * t.B3)^2 + t.V / a^(2v - 1)

# lpdensity's bw_ROT: the bandwidth that minimizes the asymptotic mean squared
# error at each point when the data are normal
function _lpdensity_rot(rule, d::SmoothingData, k, pts, p, v, minobs; robust::Bool = false, guard::Bool = false)
  _check_spread(d, rule)
  nr = _lpdensity_nr(d, k, p, v; robust, guard)
  h = map(pts) do x0
    t = nr.terms(x0)
    hx = _fmin(a -> _lpdmse(a, p, v, t), eps(), nr.range) * nr.sdx
    minobs > 0 && (hx = max(hx, _lpdfloor(d, x0, minobs)))
    min(hx, max(abs(d.values[1] - x0), abs(d.values[end] - x0)))
  end
  return BandwidthSelection(h, rule, (points = pts,))
end

# lpdensity's bw_IROT: one bandwidth that minimizes the sum of the asymptotic mean
# squared errors at the points of grid when the data are normal, with the floor and
# the cap of the points `at` where it is used
function _lpdensity_irot(d::SmoothingData, k, grid, p, v, minobs; at = grid, robust::Bool = false,
                         guard::Bool = false)
  nr = _lpdensity_nr(d, k, p, v; robust, guard)
  ts = map(nr.terms, grid)
  h = _fmin(a -> sum(t -> _lpdmse(a, p, v, t), ts), eps(), nr.range) * nr.sdx
  isfinite(h) || (h = maximum(x0 -> knn_distance(d, x0, min(nobs(d), max(minobs, p + 21))), at))
  minobs > 0 && (h = max(h, maximum(x0 -> _lpdfloor(d, x0, minobs), at)))
  return min(h, max(abs(d.values[end] - minimum(at)), abs(d.values[1] - maximum(at))))
end

# A pilot bandwidth at z0: the integrated rule of thumb g raised to the floor of
# minobs observations and distinct values at z0 and capped at the distance to the
# farther end of the data
_lpdpilot(d, g, z0, minobs) =
  min(max(g, _lpdfloor(d, z0, minobs)), max(abs(d.values[end] - z0), abs(d.values[1] - z0)))

# lpdensity's bw_MSE and bw_IMSE. On the standardized data, fits of orders p + 2 and
# p + 3 at rule-of-thumb pilot bandwidths estimate F⁽ᵖ⁺¹⁾/(p + 1)! and
# F⁽ᵖ⁺²⁾/(p + 2)!, the moments of the data at a third such bandwidth h₁ give the
# constants of the bias, and the standard error of the fit of order p at h₁ gives
# the variance. lpdensity integrates the rules of thumb for the pilots over the
# grid, so that one far evaluation point changes the pilots, and with them the
# bandwidths, at all the others. Unless compat, the data are standardized by a
# robust scale, the pilots are integrated over the quantiles of the data and floored
# at each point, and the bias is bounded by (|B₁| + a |B₂|)², which cannot vanish as
# lpdensity's (B₁ + a B₂)² does when the estimates have opposite signs. Under
# compat, a fit whose matrix R's solve rejects falls back to the floors, as in
# lpdensity.
function _lpdensity_dpi(rule, d::SmoothingData, k, pts, p, v, minobs, compat = false)
  rule.regularization == 1 && !rule.interior || throw(ArgumentError(
    "lpdensity's plug-in bandwidths have no regularization or interior option, got $rule"))
  k isa PolynomialKernel || throw(ArgumentError("lpdensity's bandwidths need a polynomial kernel, got $k"))
  _check_spread(d, rule)
  imse = rule isa IMSEOptimal
  grid = imse && rule.grid isa Integer ? collect(range(d.x[1], d.x[end]; length = rule.grid)) : pts
  n = nobs(d)
  center, scale = _lpdscale(d, !compat)
  dz = SmoothingData((d.x .- center) ./ scale; weights = d.w)
  zs = (grid .- center) ./ scale
  if compat
    # the floors and caps belong to the evaluation points, also when the integrated
    # errors run over a grid between the extremes of the data, where the floors are
    # large: the bandwidth used to be the floor at the largest observation
    zp = (pts .- center) ./ scale
    g1 = _lpdensity_irot(dz, k, zs, 2, 1, 23; at = zp)
    gp1 = _lpdensity_irot(dz, k, zs, p + 2, p + 1, p + 23; at = zp)
    gp2 = _lpdensity_irot(dz, k, zs, p + 3, p + 2, p + 24; at = zp)
    h1, hp1, hp2 = fill(g1, length(zs)), fill(gp1, length(zs)), fill(gp2, length(zs))
  else
    zq = [_quantile(dz, q) for q in 0.05:0.05:0.95]
    g1 = _lpdensity_irot(dz, k, zq, 2, 1, 0; robust = true, guard = true)
    gp1 = _lpdensity_irot(dz, k, zq, p + 2, p + 1, 0; robust = true, guard = true)
    gp2 = _lpdensity_irot(dz, k, zq, p + 3, p + 2, 0; robust = true, guard = true)
    h1 = [_lpdpilot(dz, g1, z0, 23) for z0 in zs]
    hp1 = [_lpdpilot(dz, gp1, z0, p + 23) for z0 in zs]
    hp2 = [_lpdpilot(dz, gp2, z0, p + 24) for z0 in zs]
  end
  aux = _lpdaux(dz)
  vfac = factorial(v)
  ts = _pointwise(eachindex(zs)) do j
    z0 = zs[j]
    f = _lpdfit(dz, k, aux, z0, h1[j], p, v; values = true, compat)
    G = [f.s[a + b + 1] for a in 0:p, b in 0:p]
    C, ok = _gramfactor(G, f.npos)
    ok && (!compat || _rsolvable(G)) || return (B1 = NaN, B2 = NaN, B3 = 0.0, V = NaN)
    c1 = vfac * (C \ f.s[p+2:2p+2])[v + 1]
    c2 = vfac * (C \ f.s[p+3:2p+3])[v + 1]
    B1 = _lpdcoef(dz, k, aux, z0, hp1[j], p + 2, p + 1, compat) * c1
    B2 = _lpdcoef(dz, k, aux, z0, hp2[j], p + 3, p + 2, compat) * c2
    (B1 = compat ? B1 : abs(B1), B2 = compat ? B2 : abs(B2), B3 = 0.0, V = f.p.se^2 * h1[j]^(2v - 1))
  end
  ok(t) = isfinite(t.B1) && isfinite(t.B2) && isfinite(t.V)
  fallback(x0) = knn_distance(d, x0, min(n, max(minobs, p + 21)))
  rz = dz.x[end] - dz.x[1]
  pilots = (h1 = g1 * scale, hp1 = gp1 * scale, hp2 = gp2 * scale)
  if imse
    good = filter(ok, ts)
    # where no point has its constants, lpdensity minimizes a sum without terms, which
    # leaves the bandwidth at the end of the search interval, capped below
    h = isempty(good) && !compat ? maximum(fallback, pts) :
      _fmin(a -> sum(t -> _lpdmse(a, p, v, t), good; init = 0.0), eps(), rz) * scale
    minobs > 0 && (h = max(h, maximum(x0 -> _lpdfloor(d, x0, minobs), pts)))
    h = min(h, max(abs(d.values[end] - minimum(pts)), abs(d.values[1] - maximum(pts))))
    return BandwidthSelection(h, rule, (grid = grid, pilots = pilots))
  end
  h = map(pts, ts) do x0, t
    hx = ok(t) ? _fmin(a -> _lpdmse(a, p, v, t), eps(), rz) * scale : fallback(x0)
    minobs > 0 && (hx = max(hx, _lpdfloor(d, x0, minobs)))
    min(hx, max(abs(d.values[1] - x0), abs(d.values[end] - x0)))
  end
  return BandwidthSelection(h, rule, (points = pts, pilots = pilots))
end

# β_j / hʲ of the local polynomial fit of order o to F̂ at x0 over the distinct
# values within h, or NaN if the fit is singular, or with `rtest` if R's solve
# rejects its matrix
function _lpdcoef(d::SmoothingData, k, aux, x0, h, o, j, rtest::Bool = false)
  u, U = d.values, nunique(d)
  s, t = zeros(2o + 1), zeros(o + 1)
  npos = 0
  for g in _firsttrue(g -> u[g] - x0 >= -h, U):_firsttrue(g -> u[g] - x0 > h, U)-1
    tg = (u[g] - x0) / h
    kg = aux.P[g] * k(tg)
    npos += kg > 0
    _powersums!(s, t, kg, tg, aux.F[g])
  end
  G = [s[a + b + 1] for a in 0:o, b in 0:o]
  C, ok = _gramfactor(G, npos)
  ok && (!rtest || _rsolvable(G)) || return NaN
  return (C \ t)[j + 1] / h^j
end

# Whether R's solve accepts the matrix G: it fails, and lpdensity's plug-in rules
# skip the point, where the reciprocal condition number in the 1-norm, which LAPACK
# estimates from the LU factors, is below the machine epsilon. The test depends on
# the scale of the columns, unlike that of _gramfactor: in a window much wider than
# the data around the point, the columns of high powers are tiny.
function _rsolvable(G)
  F = LinearAlgebra.lu(G; check = false)
  LinearAlgebra.issuccess(F) || return false
  return LinearAlgebra.LAPACK.gecon!('1', F.factors, LinearAlgebra.opnorm(G, 1)) >= eps()
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


confband(m::LocalPolynomialDensity; kwargs...) = _confband(m; kwargs...)
_nonnegative(m::LocalPolynomialDensity) = !m.compat && m.derivative == 0
_confint(m::LocalPolynomialDensity, level, method::Honest) =
  throw(ArgumentError("honest intervals are implemented for local polynomial regression"))
