## Plug-in bandwidths for local polynomial regression (Calonico, Cattaneo and Farrell)

"""
    MSEOptimal(; regularization = 1, interior = false)

The direct plug-in estimate of the bandwidth that minimizes the mean squared error
of a local polynomial fit at each evaluation point (Calonico, Cattaneo and Farrell
2019). At each point, the derivative of order `p + 1` that drives the bias of the
fit of order `p` is estimated by a fit of order `q = p + 1` with the pilot
bandwidth `b`, and `h` minimizes the estimated mean squared error, with the
finite-sample bias and variance constants of the local fits, a preliminary
normal-reference bandwidth for the variances, a cap at the distance to the farther
end of the data and a floor at the distance of the `minobs`-th nearest
observation. The squared bias carries the regularization term `3 B₁² V̂[β̂]` times
`regularization`, which keeps the bandwidth finite where the derivative is nearly
zero (Imbens and Kalyanaraman 2012). When `p - ν` is even and `interior = false`,
the bias includes the next term of its expansion and the mean squared error is
minimized numerically.

By default, the pilot bandwidth `b` is the [`CrossValidation`](@ref) bandwidth of
the fit of order `p` times the ratio of the [`RuleOfThumb`](@ref) bandwidths for
the derivative of order `p + 1` (fit of order `q`) and for the regression function
(fit of order `p`), so that it adapts to the roughness of the regression function;
the pilot for the derivative of order `p + 2`, which even `p - ν` needs, is formed
in the same way. Cross-validation assumes independent errors: with positively
correlated errors it undersmooths, and so does the pilot. With `compat = true`, the
rule is nprobust's `lpbwselect` with `bwselect = "mse-dpi"` instead, which agrees
with nprobust 1.0.0 to rounding. Its pilot bandwidths come from three more plug-in
stages whose bias constants are estimated by fits over the whole range of the data,
and at interior points they end at the cap, so that the derivative is estimated
over nearly the whole range. For local linear fits, its bandwidths are then about
twice the optimal ones at a sharp peak and five times for `sin(6πx)`, where the
adaptive pilots come within 10 percent at interior points.

The selection holds one bandwidth per point, and the pilot bandwidths in
`details.b`; fit with `pilot = :selected` to use them, since by default the fit
uses `b = h / rho`, as nprobust does. With the adaptive pilots, `details` also
holds the cross-validation bandwidth.

For a sample without responses, the rule gives the plug-in bandwidth for the
density at each point, `h = (R(k) f̂(x) / (4 n (B̂² + regularization R̂)))^(1/5)`
with `B̂ = μ₂(k)/2 f̂''(x)` and `R̂ = 3 (μ₂(k)/2)² V̂[f̂''(x)]`. The pilot `f̂''` uses
the bandwidth `b`, the normal-reference ratio of the bandwidths for `f''` and for
`f` applied to the [`SheatherJones`](@ref) bandwidth, so that it adapts to
multimodal densities, and `f̂` uses the Sheather–Jones bandwidth. With
`compat = true`, it gives nprobust's `kdbwselect` instead, whose rule-of-thumb
pilots assume a normal density, for the Epanechnikov and uniform kernels; for a
bimodal density they can make the bandwidth in the valley ten times too large.

For [`LocalPolynomialDensity`](@ref), the rule is the plug-in bandwidth of R's
lpdensity (`bwselect = "mse-dpi"`), which minimizes the estimated asymptotic mean
squared error `a^(2p+2-2v) (B₁ + a B₂)² + σ²/a^(2v-1)` at each point, where
`v = derivative + 1` is the order of the derivative of the distribution function.
The derivatives of orders `p + 1` and `p + 2` in `B₁` and `B₂` come from local fits
of orders `p + 2` and `p + 3` at rule-of-thumb bandwidths for the whole set of
points, and the constants of the bias and the variance from the data at a third
such bandwidth. The options `regularization` and `interior` do not apply there.
"""
struct MSEOptimal <: BandwidthSelector
  regularization::Float64
  interior::Bool
end

MSEOptimal(; regularization::Real = 1, interior::Bool = false) = MSEOptimal(regularization, interior)

"""
    IMSEOptimal(; grid = 30, regularization = 1, interior = false)

The direct plug-in estimate of the bandwidth that minimizes the integrated mean
squared error of a local polynomial fit, one bandwidth for all points: the
variance and squared bias constants of [`MSEOptimal`](@ref) are averaged over
`grid` equally spaced points between the smallest and the largest observation, as
nprobust's `imse-dpi` does, or with `grid = :points` over the evaluation points of
the fit. The pilot bandwidths are those of [`MSEOptimal`](@ref), adaptive by
default and nprobust's with `compat = true`, and `b` is in `details.b`.

For a sample without responses, the rule gives the integrated plug-in bandwidth
for densities, with the constants of [`MSEOptimal`](@ref) averaged over the
quantiles of order 0.1 to 0.9 (or the evaluation points with `grid = :points`), or
with `compat = true` nprobust's `kdbwselect`. For [`LocalPolynomialDensity`](@ref),
it is lpdensity's `imse-dpi`, which sums the terms of [`MSEOptimal`](@ref) over the
evaluation points with `grid = :points`, as lpdensity does, or over `grid` equally
spaced points between the smallest and the largest observation.
"""
struct IMSEOptimal <: BandwidthSelector
  grid::Union{Int,Symbol}
  regularization::Float64
  interior::Bool

  function IMSEOptimal(grid::Union{Integer,Symbol}, regularization::Real, interior::Bool)
    grid isa Symbol && grid != :points && throw(ArgumentError("grid must be a number of points or :points, got :$grid"))
    grid isa Integer && grid < 1 && throw(ArgumentError("grid must have at least one point, got $grid"))
    new(grid, regularization, interior)
  end
end

IMSEOptimal(; grid::Union{Integer,Symbol} = 30, regularization::Real = 1, interior::Bool = false) =
  IMSEOptimal(grid, regularization, interior)

"""
    CoverageOptimal(base = MSEOptimal(); convention = :nprobust)

Bandwidths that make the coverage error of robust bias-corrected intervals
shrink at the fastest rate, obtained by rescaling the plug-in bandwidths of
`base` ([`MSEOptimal`](@ref) or [`IMSEOptimal`](@ref)) by a power of the sample
size (Calonico, Cattaneo and Farrell 2018, 2022).

With `convention = :nprobust`, the rescaling targets the rate `n^(-1/(p+3))`,
which is coverage-error optimal at boundary points and for even `p - ν`, at every
point: `h n^(-p/((2p+3)(p+3)))` for odd `p - ν` and `h n^(-(p+2)/((2p+5)(p+3)))`
for even `p - ν`, as nprobust's `ce-rot`. With `convention = :interior`, an odd
`p - ν` targets the interior rate `n^(-1/(p+4))` instead, `h n^(-(p-1)/((2p+3)(p+4)))`,
which leaves local linear bandwidths unchanged. The pilot bandwidths are rescaled
as in nprobust in both conventions.
"""
struct CoverageOptimal{S<:Union{MSEOptimal,IMSEOptimal}} <: BandwidthSelector
  base::S
  convention::Symbol

  function CoverageOptimal(base::S, convention::Symbol) where S<:Union{MSEOptimal,IMSEOptimal}
    convention in (:nprobust, :interior) ||
      throw(ArgumentError("convention must be :nprobust or :interior, got :$convention"))
    new{S}(base, convention)
  end
end

CoverageOptimal(base::Union{MSEOptimal,IMSEOptimal} = MSEOptimal(); convention::Symbol = :nprobust) =
  CoverageOptimal(base, convention)

# the plug-in rules take the choice between the adaptive pilots and nprobust's
_regression_selection(rule::Union{MSEOptimal,IMSEOptimal,CoverageOptimal}, d, k, p, ν, pts, vce, minobs;
                      compat = false) =
  selectbandwidth(rule, d, k; degree = p, derivative = ν, points = pts, vce, minobs, compat)

function selectbandwidth(rule::MSEOptimal, d::SmoothingData, k::AbstractKernel; degree::Integer = 1,
                         derivative::Integer = 0, points = nothing, vce::VarianceEstimator = NNVariance(),
                         minobs::Integer = 21, compat::Bool = false)
  d.y === nothing && return _densityplugin(rule, d, k, points === nothing ? nothing : _checkpoints(points),
                                           minobs, compat; imse = false)
  pts = points === nothing ? collect(range(d.x[1], d.x[end]; length = 30)) : _checkpoints(points)
  p, ν = Int(degree), Int(derivative)
  pilots = compat ? nothing : _regressionpilots(d, k, p)
  r = _pointwise(c -> _msedpi(d, k, c, p, ν, rule.regularization, rule.interior, vce, minobs, pilots), pts)
  details = (b = [s.b for s in r], points = pts)
  return BandwidthSelection([s.h for s in r], rule,
                            compat ? details : (details..., crossvalidation = pilots.crossvalidation))
end

function selectbandwidth(rule::IMSEOptimal, d::SmoothingData, k::AbstractKernel; degree::Integer = 1,
                         derivative::Integer = 0, points = nothing, vce::VarianceEstimator = NNVariance(),
                         minobs::Integer = 21, compat::Bool = false)
  d.y === nothing && return _densityplugin(rule, d, k, points === nothing ? nothing : _checkpoints(points),
                                           minobs, compat; imse = true)
  p, ν = Int(degree), Int(derivative)
  q = p + 1
  grid = if rule.grid isa Integer
    collect(range(d.x[1], d.x[end]; length = rule.grid))
  else
    points === nothing && throw(ArgumentError("IMSEOptimal(grid = :points) needs the evaluation points"))
    _checkpoints(points)
  end
  pilots = compat ? nothing : _regressionpilots(d, k, p)
  r = _pointwise(c -> _msedpi(d, k, c, p, ν, rule.regularization, rule.interior, vce, minobs, pilots), grid)
  N, R = nobs(d), d.x[end] - d.x[1]
  Vh, Bh = sum(s.Vh for s in r) / length(r), sum(s.Bh for s in r) / length(r)
  if isodd(p - ν) || rule.interior
    h = (Vh / (N * Bh))^(1 / (2p + 3))
  else
    h = _fmin(H -> abs(H^(2p + 2 - 2ν) * Bh + Vh / (N * H^(1 + 2ν))), eps(), R)
  end
  compat || return BandwidthSelection(h, rule, (b = pilots.b1, grid = grid, crossvalidation = pilots.crossvalidation))
  Vb, Bb = sum(s.Vb for s in r) / length(r), sum(s.Bb for s in r) / length(r)
  if isodd(p - ν) || rule.interior
    b = (Vb / (N * Bb))^(1 / (2q + 3))
  else
    b = _fmin(H -> abs(H^(2q + 2 - 2(p + 1)) * Bb + Vb / (N * H^(1 + 2(p + 1)))), eps(), R)
  end
  return BandwidthSelection(h, rule, (b = b, grid = grid))
end

function selectbandwidth(rule::CoverageOptimal, d::SmoothingData, k::AbstractKernel; degree::Integer = 1,
                         derivative::Integer = 0, kwargs...)
  s = selectbandwidth(rule.base, d, k; degree, derivative, kwargs...)
  # for densities with second-order kernels (p = 2) the factors are one, as in nprobust
  d.y === nothing && return BandwidthSelection(s.h, rule, (s.details..., mse = s.h))
  p, ν = Int(degree), Int(derivative)
  q, N = p + 1, nobs(d)
  if iseven(p - ν)
    fh, fb = N^(-(p + 2) / ((2p + 5) * (p + 3))), N^(-q / ((2q + 3) * (q + 3)))
  else
    fh = rule.convention == :nprobust ? N^(-p / ((2p + 3) * (p + 3))) : N^(-(p - 1) / ((2p + 3) * (p + 4)))
    fb = N^(-(q + 2) / ((2q + 5) * (q + 3)))
  end
  return BandwidthSelection(s.h * fh, rule, (s.details..., b = s.details.b * fb, mse = s.h))
end

# The pilot bandwidths of the adaptive rule for a fit of order p: the
# cross-validation bandwidth times the ratios of the rule-of-thumb bandwidths for the
# derivatives of orders p + 1 (fit of order p + 1) and p + 2 (fit of order p + 2) to
# that for the regression function
function _regressionpilots(d, k, p)
  hcv = selectbandwidth(CrossValidation(), d, k; degree = p).h
  rot(o, ν) = selectbandwidth(RuleOfThumb(), d, k; degree = o, derivative = ν).h
  h0 = rot(p, 0)
  return (b1 = hcv * rot(p + 1, p + 1) / h0, b2 = hcv * rot(p + 2, p + 2) / h0, crossvalidation = hcv)
end

# The stages of nprobust's lpbwselect.mse.dpi at the point c, for a fit of order p
# of the derivative ν, or only its last stage with the adaptive pilot bandwidths;
# returns h, b and the variance and squared-bias constants that IMSEOptimal averages.
function _msedpi(d, k, c, p, ν, regul, interior, vce, minobs, pilots = nothing)
  q = p + 1
  even = iseven(p - ν)
  xs, N = d.x, nobs(d)
  R = xs[end] - xs[1]
  bwmax = max(abs(c - xs[1]), abs(c - xs[end]))
  # the preliminary bandwidth for the variances: a normal-reference rule with the
  # unweighted spread of x, as in nprobust
  sd = N > 1 ? sqrt(sum(abs2, xs .- sum(xs) / N) / (N - 1)) : 0.0
  iqr = _unweightedquantile(xs, 0.75) - _unweightedquantile(xs, 0.25)
  cbw = min(_nrconstant(k) * min(sd, iqr / 1.349) * N^(-1 / 5), bwmax)
  bwmin = minobs > 0 ? knn_distance(d, c, min(minobs, N)) : 0.0
  cbw = max(cbw, bwmin)
  clamp_bw(h) = max(min(h, bwmax), bwmin)
  # the bandwidth that minimizes the mean squared error from the constants of a stage
  function optimal(C, o, nu, scale)
    (!even || interior) && return C.bw
    return _fmin(H -> abs(H^(2o + 2 - 2nu) * (C.B1 + H * C.B2 + scale * C.R)^2 + C.V / (N * H^(1 + 2nu))),
                 eps(), R)
  end
  if pilots === nothing
    d1 = _lpbw(d, k, c, q + 1, q + 1, q + 2, cbw, R, R, 0.0, vce)
    d2 = _lpbw(d, k, c, q + 2, q + 2, q + 3, cbw, R, R, 0.0, vce)
    bw2 = clamp_bw(optimal(d1, q + 1, q + 1, 0.0))
    bw3 = clamp_bw(optimal(d2, q + 2, q + 2, 0.0))
    cb = _lpbw(d, k, c, q, p + 1, q + 1, cbw, bw2, bw3, regul, vce)
    b = clamp_bw(optimal(cb, q, p + 1, regul))
  else
    b, bw2 = clamp_bw(pilots.b1), clamp_bw(pilots.b2)
  end
  ch = _lpbw(d, k, c, p, ν, q, cbw, b, bw2, regul, vce)
  h = clamp_bw(optimal(ch, p, ν, regul))
  Vh, Bh = !even || interior ? (ch.rV * ch.V, ch.rB * ch.B1^2) : (ch.V, (ch.B1 + h * ch.B2)^2)
  pilots === nothing || return (h = h, b = b, Vh = Vh, Bh = Bh)
  Vb, Bb = !even || interior ? (cb.rV * cb.V, cb.rB * cb.B1^2) : (cb.V, (cb.B1 + b * cb.B2)^2)
  return (h = h, b = b, Vh = Vh, Bh = Bh, Vb = Vb, Bb = Bb)
end

# nprobust's normal-reference constants for its kernels, and the exact ones otherwise
function _nrconstant(k::AbstractKernel)
  k isa EpanechnikovKernel && return 2.34
  k isa UniformKernel && return 1.843
  k isa TriangularKernel && return 2.576
  k isa GaussianKernel && return 1.06
  return (8 * sqrt(π) * roughness(k) / (3 * _mu2(k)^2))^(1 / 5)
end

# R's type 7 quantile of sorted values
function _unweightedquantile(xs, p)
  n = length(xs)
  t = 1 + (n - 1) * p
  i = clamp(floor(Int, t), 1, n - 1)
  return n == 1 ? xs[1] : xs[i] + (t - i) * (xs[i + 1] - xs[i])
end

# nprobust's lprobust.bw: the variance constant V of the derivative ν of a fit of
# order o at c with bandwidth hV and its bias constants B1, B2, the leading
# derivatives estimated by fits of orders oB and oB + 1 with bandwidths hB1, hB2,
# the regularization R, and the bandwidth that minimizes the mean squared error.
# The fits use the scaled variable u = (x - c) / h; raw coefficients of (x - c)^j
# are the scaled ones divided by h^j.
function _lpbw(d, k, c, o, ν, oB, hV, hB1, hB2, scale, vce)
  N = nobs(d)
  fV = _lfit(d, k, c, hV, o, 2)
  g = fV.C \ [j == ν ? 1.0 : 0.0 for j in 0:o]
  VV = _coefvar(d, k, fV, g, vce) / hV^(2ν)
  BConst1 = sum(g[j + 1] * fV.s[o + j + 2] for j in 0:o)   # e_ν' Γ⁻¹ ∑ wk(u) u^(o+1) r(u)
  BConst2 = sum(g[j + 1] * fV.s[o + j + 3] for j in 0:o)
  fB1 = _lfit(d, k, c, hB1, oB, 0)
  Rreg = 0.0
  if scale > 0
    gB = fB1.C \ [j == o + 1 ? 1.0 : 0.0 for j in 0:oB]
    Rreg = 3 * BConst1^2 * _coefvar(d, k, fB1, gB, vce) / hB1^(2(o + 1))
  end
  fB2 = _lfit(d, k, c, hB2, oB + 1, 0)
  B1 = BConst1 * fB1.β[o + 2] / hB1^(o + 1)
  B2 = BConst2 * fB2.β[o + 3] / hB2^(o + 2)
  V = N * hV^(2ν + 1) * VV
  rB, rV = 2(o + 1 - ν), 2ν + 1
  bw = (rV * V / (N * rB * (B1^2 + scale * Rreg)))^(1 / (2o + 3))
  return (V = V, B1 = B1, B2 = B2, R = Rreg, rB = rB, rV = rV, bw = bw)
end

# A weighted fit of order o at c with bandwidth h over the observations with positive
# weight a:e; s holds the power sums ∑ wk(u) u^j up to j = 2o + extra
function _lfit(d, k, c, h, o, extra)
  xs, n = d.x, nobs(d)
  lo, hi = support(k)
  a = isfinite(lo) ? _firsttrue(i -> (xs[i] - c) / h >= lo, n) : 1
  e = isfinite(hi) ? _firsttrue(i -> (xs[i] - c) / h > hi, n) - 1 : n
  while a <= e && k((xs[a] - c) / h) == 0
    a += 1
  end
  while e >= a && k((xs[e] - c) / h) == 0
    e -= 1
  end
  s, t = zeros(2o + 1 + extra), zeros(o + 1)
  for i in a:e
    u = (xs[i] - c) / h
    _powersums!(s, t, _weight(d.w, i) * k(u), u, d.y[i])
  end
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric([s[i + j + 1] for i in 0:o, j in 0:o]))
  return (a = a, e = e, h = h, o = o, c = c, s = s, C = C, β = C \ t)
end

# ∑ᵢ (wᵢ k(uᵢ) g'r(uᵢ))² σ̂ᵢ² over the window of a fit, with σ̂ᵢ² from the variance
# estimator: nearest neighbours in the window, or the residuals of this fit
function _coefvar(d, k, f, g, vce)
  a, e, h, o, c = f.a, f.e, f.h, f.o, f.c
  σ2 = vce isa NNVariance ? _nnvariances(d, a, e, vce.matches) : nothing
  n = e - a + 1
  v = 0.0
  for i in a:e
    u = (d.x[i] - c) / h
    kw = _weight(d.w, i) * k(u)
    s2 = if vce isa NNVariance
      σ2[i - a + 1]
    else
      ε2 = (d.y[i] - evalpoly(u, f.β))^2
      if vce.type == 0
        ε2
      elseif vce.type == 1
        ε2 * n / (n - o - 1)
      else
        r = [u^j for j in 0:o]
        hii = kw * LinearAlgebra.dot(r, f.C \ r)
        vce.type == 2 ? ε2 / (1 - hii) : ε2 / (1 - hii)^2
      end
    end
    v += (kw * evalpoly(u, g))^2 * s2
  end
  return v
end

# The minimizer of f on [a, b] by Brent's method (golden sections and parabolic
# interpolation), as R's optimize, with its default tolerance
function _fmin(f, a, b; tol = eps()^0.25)
  c = (3 - sqrt(5)) / 2
  ϵ = sqrt(eps())
  v = w = x = a + c * (b - a)
  fv = fw = fx = f(x)
  d = e = 0.0
  tol3 = tol / 3
  while true
    xm = (a + b) / 2
    tol1 = ϵ * abs(x) + tol3
    t2 = 2tol1
    abs(x - xm) <= t2 - (b - a) / 2 && break
    p = q = r = 0.0
    if abs(e) > tol1   # fit a parabola
      r = (x - w) * (fx - fv)
      q = (x - v) * (fx - fw)
      p = (x - v) * q - (x - w) * r
      q = 2(q - r)
      q > 0 ? (p = -p) : (q = -q)
      r = e
      e = d
    end
    if abs(p) >= abs(q / 2 * r) || p <= q * (a - x) || p >= q * (b - x)
      e = x < xm ? b - x : a - x   # a golden-section step
      d = c * e
    else                           # a parabolic-interpolation step
      d = p / q
      u = x + d
      (u - a < t2 || b - u < t2) && (d = x < xm ? tol1 : -tol1)
    end
    u = abs(d) >= tol1 ? x + d : (d > 0 ? x + tol1 : x - tol1)
    fu = f(u)
    if fu <= fx
      u < x ? (b = x) : (a = x)
      v, fv = w, fw
      w, fw = x, fx
      x, fx = u, fu
    else
      u < x ? (a = u) : (b = u)
      if fu <= fw || w == x
        v, fv = w, fw
        w, fw = u, fu
      elseif fu <= fv || v == x || v == w
        v, fv = u, fu
      end
    end
  end
  return x
end
