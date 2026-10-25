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
observation. The squared bias `B₁²` carries the regularization term `3 B₁² V̂[β̂]`
times `regularization`, which keeps the bandwidth finite where the derivative is
nearly zero (Imbens and Kalyanaraman 2012). When `p - ν` is even and
`interior = false`, the bias includes the next term of its expansion and the mean
squared error is minimized numerically. Its squared bias is then
`(|B₁| + h |B₂|)²` plus the regularization term, which cannot vanish where the two
estimated terms have opposite signs; nprobust's `(B₁ + h B₂ + regularization R)²`,
kept with `compat = true`, can, and since it adds `R`, which has the units of `B₁²`,
to `B₁`, its bandwidth depends on the units of `y`.

By default, the pilot bandwidth `b` is the [`CrossValidation`](@ref) bandwidth of
the fit of order `p` times the ratio of the [`RuleOfThumb`](@ref) bandwidths for
the derivative of order `p + 1` and for the regression function, so that it adapts
to the roughness of the regression function; the pilot for the derivative of order
`p + 2`, which even `p - ν` needs, is formed in the same way. The derivative of
order `p + 1` is then estimated by a fit of order `q` with bandwidth `b`. Cross-validation assumes independent errors: with positively
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
Unless `compat = true`, the squared bias is bounded by `(|B₁| + a |B₂|)²`, since
lpdensity's `(B₁ + a B₂)²` vanishes at `a = -B₁/B₂` when the estimated terms have
opposite signs.
"""
struct MSEOptimal <: BandwidthSelector
  regularization::Float64
  interior::Bool

  function MSEOptimal(regularization::Real, interior::Bool)
    _check_regularization(regularization)
    new(regularization, interior)
  end
end

MSEOptimal(; regularization::Real = 1, interior::Bool = false) = MSEOptimal(regularization, interior)

_check_regularization(r) = 0 <= r < Inf ||
  throw(ArgumentError("regularization must be nonnegative and finite, got $r"))

"""
    IMSEOptimal(; grid = nothing, regularization = 1, interior = false)

The direct plug-in estimate of the bandwidth that minimizes the integrated mean
squared error of a local polynomial fit, one bandwidth for all points: the
variance and squared bias constants of [`MSEOptimal`](@ref), with its
regularization, are averaged over a grid of points, and the bandwidth is capped at
the range of the data. For regression, the grid is by default 30 quantiles of the
covariate, of order 0.05 to 0.95, so that the average estimates the error at a
typical observation; `grid = m` takes `m` such quantiles, and `grid = :points` the
evaluation points of the fit. nprobust's `imse-dpi`, which `compat = true`
reproduces, averages over equally spaced points between the smallest and the largest
observation. For a covariate with heavy tails these lie mostly where few
observations are left, and the constants estimated there, which are large and
unstable, dominate the average. The pilot bandwidths are those of [`MSEOptimal`](@ref),
adaptive by default, and `b` is in `details.b`. With `compat = true` the rule is
nprobust's, whose pilots are its own, whose averaged squared bias leaves out the
regularization and whose bandwidth has no cap, which for an exactly linear
regression function was 966 on the unit interval.

For a sample without responses, the rule gives the integrated plug-in bandwidth
for densities, with the constants of [`MSEOptimal`](@ref) averaged by default over
30 quantiles of order 0.1 to 0.9, as in nprobust's `kdbwselect`, which
`compat = true` reproduces; `grid = m` takes `m` such quantiles. For
[`LocalPolynomialDensity`](@ref), it is lpdensity's `imse-dpi`, which sums the
terms of [`MSEOptimal`](@ref) by default over the evaluation points, as lpdensity
does, and with `grid = m` over `m` equally spaced points between the smallest and
the largest observation. `grid = :points` needs evaluation points.
"""
struct IMSEOptimal <: BandwidthSelector
  grid::Union{Nothing,Int,Symbol}     # nothing: the default grid of each estimator
  regularization::Float64
  interior::Bool

  function IMSEOptimal(grid::Union{Nothing,Integer,Symbol}, regularization::Real, interior::Bool)
    grid isa Symbol && grid != :points && throw(ArgumentError("grid must be a number of points or :points, got :$grid"))
    grid isa Integer && grid < 2 && throw(ArgumentError("grid must have at least two points, got $grid"))
    _check_regularization(regularization)
    new(grid, regularization, interior)
  end
end

IMSEOptimal(; grid::Union{Nothing,Integer,Symbol} = nothing, regularization::Real = 1, interior::Bool = false) =
  IMSEOptimal(grid, regularization, interior)

# The default evaluation points of regression: 30 equally spaced points between the
# quantiles of order 0.025 and 0.975 of the covariate, or between its extremes as
# nprobust under compat (or where these quantiles coincide). Between the extremes, a
# covariate with heavy tails put every point outside the central 95 percent of the
# data: for a t₃ sample of 200 with an observation at -377, the points nearest the
# data were -7.8 and 5.4.
function _regressionpoints(d, compat)
  lo, hi = compat ? (d.x[1], d.x[end]) : (_quantile(d, 0.025), _quantile(d, 0.975))
  lo < hi || ((lo, hi) = (d.x[1], d.x[end]))
  return collect(range(lo, hi; length = 30))
end

# the grid of the integrated rule for regression: m quantiles of order 0.05 to 0.95 of
# the covariate, or m equally spaced points between its extremes as nprobust
_regressiongrid(d, m, compat) = compat ? collect(range(d.x[1], d.x[end]; length = m)) :
  [_quantile(d, q) for q in range(0.05, 0.95; length = m)]

# the evaluation points for grid = :points, which must be given
function _gridpoints(points)
  points === nothing && throw(ArgumentError("IMSEOptimal(grid = :points) needs the evaluation points"))
  return points
end

"""
    CoverageOptimal(base = MSEOptimal(); convention = nothing)

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
as in nprobust in both conventions. By default the convention is `:interior`, since
most evaluation points are interior ones, and `:nprobust` with `compat = true`.
"""
struct CoverageOptimal{S<:Union{MSEOptimal,IMSEOptimal}} <: BandwidthSelector
  base::S
  convention::Union{Nothing,Symbol}   # nothing: :interior, or :nprobust under compat

  function CoverageOptimal(base::S, convention::Union{Nothing,Symbol}) where S<:Union{MSEOptimal,IMSEOptimal}
    convention in (nothing, :nprobust, :interior) ||
      throw(ArgumentError("convention must be :nprobust or :interior, got :$convention"))
    new{S}(base, convention)
  end
end

CoverageOptimal(base::Union{MSEOptimal,IMSEOptimal} = MSEOptimal(); convention::Union{Nothing,Symbol} = nothing) =
  CoverageOptimal(base, convention)

function Base.show(io::IO, r::CoverageOptimal)
  print(io, "CoverageOptimal(")
  r.base == MSEOptimal() || show(io, r.base)
  r.convention === nothing || print(io, r.base == MSEOptimal() ? "" : "; ", "convention = ", repr(r.convention))
  print(io, ")")
end

# the plug-in rules take the choice between the adaptive pilots and nprobust's
_regression_selection(rule::Union{MSEOptimal,IMSEOptimal,CoverageOptimal}, d, k, p, ν, pts, vce, minobs;
                      compat = false) =
  select_bandwidth(rule, d, k; degree = p, derivative = ν, points = pts, vce, minobs, compat)

function select_bandwidth(rule::MSEOptimal, d::SmoothingData, k::AbstractKernel; derivative::Integer = 0,
                         degree::Integer = derivative + 1, points = nothing,
                         vce::Union{Nothing,VarianceEstimator} = nothing, minobs::Integer = 21, compat::Bool = false)
  d.y === nothing && return _densityplugin(rule, d, k, points === nothing ? nothing : _checkpoints(points),
                                           minobs, compat, Val(false))
  _check_spread(d, rule)
  _check_responses(d, rule)
  _check_degree(degree, derivative)
  vce = _variance(vce, d)
  pts = points === nothing ? _regressionpoints(d, compat) : _checkpoints(points)
  p, ν = Int(degree), Int(derivative)
  pilots = compat ? nothing : _regressionpilots(d, k, p)
  r = _msedpis(d, k, pts, p, ν, rule, vce, minobs, pilots)
  # the details have one shape for both pilots, so that the selection is inferable
  return BandwidthSelection([s.h for s in r], rule, (b = [s.b for s in r], points = pts,
                            crossvalidation = compat ? NaN : pilots.crossvalidation))
end

function select_bandwidth(rule::IMSEOptimal, d::SmoothingData, k::AbstractKernel; derivative::Integer = 0,
                         degree::Integer = derivative + 1, points = nothing,
                         vce::Union{Nothing,VarianceEstimator} = nothing, minobs::Integer = 21, compat::Bool = false)
  d.y === nothing && return _densityplugin(rule, d, k, points === nothing ? nothing : _checkpoints(points),
                                           minobs, compat, Val(true))
  _check_spread(d, rule)
  _check_responses(d, rule)
  _check_degree(degree, derivative)
  vce = _variance(vce, d)
  p, ν = Int(degree), Int(derivative)
  q = p + 1
  grid = rule.grid === :points ? _checkpoints(_gridpoints(points)) : _regressiongrid(d, something(rule.grid, 30), compat)
  pilots = compat ? nothing : _regressionpilots(d, k, p)
  r = _msedpis(d, k, grid, p, ν, rule, vce, minobs, pilots)
  N, R = nobs(d), d.x[end] - d.x[1]
  twoterms = iseven(p - ν) && !rule.interior
  # the minimizer of the integrated mean squared error of a fit of order o for the
  # derivative nu, from the averages V̄ and B̄² of its constants over the grid
  imsebandwidth(V̄, B̄², o, nu) = twoterms ?
    _fmin(H -> abs(H^(2o + 2 - 2nu) * B̄² + V̄ / (N * H^(1 + 2nu))), eps(), R) : (V̄ / (N * B̄²))^(1 / (2o + 3))
  average(f) = sum(f(s) for s in r) / length(r)
  h = imsebandwidth(average(s -> s.forh.V), average(s -> s.forh.B²), p, ν)
  # where the bias nearly vanishes, as for a linear regression function, the closed
  # form has no bound in nprobust; the adaptive rule caps it at the range, as
  # MSEOptimal does
  compat || twoterms || (h = min(h, R))
  compat || return BandwidthSelection(h, rule, (b = Float64(pilots.b1), grid = grid,
                                                crossvalidation = pilots.crossvalidation))
  b = imsebandwidth(average(s -> s.forb.V), average(s -> s.forb.B²), q, p + 1)
  return BandwidthSelection(h, rule, (b = Float64(b), grid = grid, crossvalidation = NaN))
end

function select_bandwidth(rule::CoverageOptimal, d::SmoothingData, k::AbstractKernel; derivative::Integer = 0,
                         degree::Integer = derivative + 1, compat::Bool = false, kwargs...)
  s = select_bandwidth(rule.base, d, k; degree, derivative, compat, kwargs...)
  convention = something(rule.convention, compat ? :nprobust : :interior)
  # for densities with second-order kernels (p = 2) the factors are one, as in nprobust
  d.y === nothing && return BandwidthSelection(s.h, rule, (s.details..., mse = s.h))
  p, ν = Int(degree), Int(derivative)
  q, N = p + 1, nobs(d)
  if iseven(p - ν)
    fh, fb = N^(-(p + 2) / ((2p + 5) * (p + 3))), N^(-q / ((2q + 3) * (q + 3)))
  else
    fh = convention == :nprobust ? N^(-p / ((2p + 3) * (p + 3))) : N^(-(p - 1) / ((2p + 3) * (p + 4)))
    fb = N^(-(q + 2) / ((2q + 5) * (q + 3)))
  end
  return BandwidthSelection(s.h * fh, rule, (s.details..., b = s.details.b * fb, mse = s.h))
end

# With constant responses the estimated bias and variance vanish at every bandwidth,
# so that the mean squared error has no minimizer
_check_responses(d, rule) = allequal(d.y) && throw(ArgumentError("the responses are constant, so " *
  "the estimated mean squared error vanishes at every bandwidth and $rule cannot choose one; pass a bandwidth"))

# The pilot bandwidths of the adaptive rule for a fit of order p: the
# cross-validation bandwidth times the ratios of the rule-of-thumb bandwidths for the
# derivatives of orders p + 1 (fit of order p + 1) and p + 2 (fit of order p + 2) to
# that for the regression function
function _regressionpilots(d, k, p)
  hcv = select_bandwidth(CrossValidation(), d, k; degree = p).h
  rot(o, ν) = select_bandwidth(RuleOfThumb(), d, k; degree = o, derivative = ν).h
  h0 = rot(p, 0)
  return (b1 = hcv * rot(p + 1, p + 1) / h0, b2 = hcv * rot(p + 2, p + 2) / h0, crossvalidation = hcv)
end

# _msedpi at the points cs, in tasks; a function barrier, so that the closure captures
# pilots of one type, which Julia 1.10 needs to infer the results. The preliminary
# bandwidth does not depend on the point.
function _msedpis(d, k, cs, p, ν, rule, vce, minobs, pilots)
  hσ = _variancepilot(d, k, pilots === nothing)
  return _pointwise(c -> _msedpi(d, k, c, p, ν, rule.regularization, rule.interior, vce, minobs, hσ, pilots), cs)
end

# The preliminary bandwidth at which the variance constants are estimated: a
# normal-reference rule with the unweighted spread of x, as in nprobust; the adaptive
# rule takes the standard deviation where more than half of the data are tied and
# the interquartile range vanishes, which without a floor made the bandwidth zero
function _variancepilot(d, k, compat)
  xs, N = d.x, nobs(d)
  sd = N > 1 ? sqrt(sum(abs2, xs .- _sortedmean(xs)) / (N - 1)) : 0.0
  iqr = _unweightedquantile(xs, 0.75) - _unweightedquantile(xs, 0.25)
  spread = compat ? min(sd, iqr / 1.349) : _robustscale(sd, iqr / 1.349)
  return _nrconstant(k) * spread * N^(-1 / 5)
end

# The direct plug-in bandwidth h that minimizes the estimated mean squared error of
# the estimate of the derivative ν by a fit of order p at the point c, and its pilot
# bandwidth b (Calonico, Cattaneo and Farrell 2018, 2019). The bias of the fit
# involves the derivative of order p + 1 and, for even p - ν at boundary points, the
# one of order p + 2, which fits of orders p + 1 and p + 2 estimate with the pilot
# bandwidths b and b′. These are given (the adaptive pilots), or, with
# pilots === nothing, selected by the same plug-in in turn, which reproduces
# nprobust: b′ and b″ for the derivatives of orders p + 2 and p + 3, with their bias
# estimated over the whole range of the data, then b from b′ and b″, and h from b
# and b′. Every bandwidth is capped at the distance to the farther end of the data
# and floored at the distance of the minobs-th nearest observation, and so is the
# preliminary bandwidth hσ₀ at which the variance constants are estimated.
# Returns h, b and the variance and squared-bias constants of both, which
# IMSEOptimal averages.
function _msedpi(d, k, c, p, ν, λ, interior, vce, minobs, hσ₀, pilots)
  compat = pilots === nothing
  q = p + 1
  xs, N = d.x, nobs(d)
  R = xs[end] - xs[1]
  hmax = max(abs(c - xs[1]), abs(c - xs[end]))
  hmin = minobs > 0 ? knn_distance(d, c, min(minobs, N)) : 0.0
  bounded(H) = max(min(H, hmax), hmin)
  hσ = bounded(hσ₀)
  # the second bias term enters only for even p - ν at boundary points, in every stage
  twoterms = iseven(p - ν) && !interior
  # With two terms, nprobust's squared bias (B₁ + H B₂ + λℛ)² adds ℛ, which has the
  # units of B₁², to B₁, so that the bandwidth depends on the units of y, and
  # vanishes where the estimated terms cancel; the adaptive rule bounds it by
  # (|B₁| + H |B₂|)² + λℛ instead, as for lpdensity
  sqbias(M, H, λ) = compat ? (M.B₁ + H * M.B₂ + λ * M.ℛ)^2 : (abs(M.B₁) + H * abs(M.B₂))^2 + λ * M.ℛ
  # A stage: the constants M of the fit of order o for the derivative nu, whose bias
  # terms use the pilots b₁ and b₂, and the bandwidth that minimizes the estimated
  # mean squared error H^(2(o+1-nu)) (bias)² + V / (N H^(2nu+1)), in closed form
  # with one bias term and numerically with two
  function stage(o, nu, b₁, b₂, λ)
    M = _mseconstants(d, k, c, o, nu, hσ, b₁, b₂, λ, vce, twoterms, compat)
    isfinite(M.V) || return bounded(NaN), M
    twoterms || return bounded(((2nu + 1) * M.V / (N * (2(o + 1 - nu)) * (M.B₁^2 + λ * M.ℛ)))^(1 / (2o + 3))), M
    return bounded(_fmin(H -> abs(H^(2o + 2 - 2nu) * sqbias(M, H, λ) + M.V / (N * H^(1 + 2nu))), eps(), R)), M
  end
  if compat
    b′, _ = stage(q + 1, q + 1, R, R, 0.0)
    b″, _ = stage(q + 2, q + 2, R, R, 0.0)
    b, Mb = stage(q, p + 1, b′, b″, λ)
  else
    b, b′ = bounded(pilots.b1), bounded(pilots.b2)
  end
  h, Mh = stage(p, ν, b, b′, λ)
  isfinite(h) || throw(ArgumentError("the plug-in bandwidth at x = $c is not defined, since a " *
    "local fit of its stages has fewer distinct values of x with positive weight than " *
    "coefficients" * (d.cluster === nothing ? "" : ", or fewer than two clusters") * "; pass a bandwidth"))
  # The constants of a stage at its bandwidth H that IMSEOptimal averages, with one
  # bias term weighted by the exponents of the closed form; nprobust leaves out the
  # regularization, which the adaptive rule keeps, as MSEOptimal does
  function averaged(M, o, nu, H)
    twoterms && return (V = M.V, B² = compat ? (M.B₁ + H * M.B₂)^2 : sqbias(M, H, λ))
    return (V = (2nu + 1) * M.V, B² = 2(o + 1 - nu) * (compat ? M.B₁^2 : M.B₁^2 + λ * M.ℛ))
  end
  # one shape for both pilots, so that the callers infer
  forh = averaged(Mh, p, ν, h)
  forb = compat ? averaged(Mb, q, p + 1, b) : (V = NaN, B² = NaN)
  return (h = h, b = b, forh = forh, forb = forb)
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

# The constants of the mean squared error of the estimate of the derivative ν by a
# fit of order o at c, MSE(H) ≈ H^(2(o+1-ν)) (B₁ + H B₂)² + V / (N H^(2ν+1))
# (Calonico, Cattaneo and Titiunik 2014; Calonico, Cattaneo and Farrell 2018). With
# Γ the Gram matrix of the fit with the preliminary bandwidth hσ and
# Λⱼ = ∑ wᵢ k(uᵢ) uᵢ^(o+j) r(uᵢ) its moments of order o + j, the bias
# constants are Bⱼ = e_ν'Γ⁻¹Λⱼ μ⁽ᵒ⁺ʲ⁾/(o+j)!, j = 1, 2, with the derivative from a
# fit of order o + j with the pilot bandwidth bⱼ, B₂ only with two terms; the
# variance constant is V = N hσ^(2ν+1) V̂, with V̂ the variance of the coefficient of
# order ν of the fit with hσ; and ℛ = 3 (e_ν'Γ⁻¹Λ₁)² V̂₁, with V̂₁ the variance of the
# coefficient of order o + 1 of the fit with b₁, is the term that regularizes B₁²
# (Imbens and Kalyanaraman 2012), needed only where λ > 0. The constants are NaN
# where a fit is not identified. The fits use the scaled variable u = (x - c) / h,
# whose coefficients of order j are those of (x - c)^j times h^j. In the adaptive
# rule a fit that is not identified widens its window, and neighbours at distances
# that differ by rounding are tied; under compat, as in nprobust, neither.
function _mseconstants(d, k, c, o, ν, hσ, b₁, b₂, λ, vce, twoterms::Bool, compat::Bool)
  widen = !compat
  fσ = _lfit(d, k, c, hσ, o, 2, widen)   # with the moments up to order 2o + 2
  f₁ = _lfit(d, k, c, b₁, o + 1, 0, widen)
  f₂ = twoterms ? _lfit(d, k, c, b₂, o + 2, 0, widen) : nothing
  fσ.ok && f₁.ok && (f₂ === nothing || f₂.ok) || return (V = NaN, B₁ = NaN, B₂ = NaN, ℛ = NaN)
  # the bandwidths are those of the fits, which a fit that is not identified widens
  eΓ = fσ.C \ [j == ν ? 1.0 : 0.0 for j in 0:o]   # Γ⁻¹ e_ν
  V = nobs(d) * fσ.h^(2ν + 1) * (_coefvar(d, k, fσ, eΓ, vce, compat) / fσ.h^(2ν))
  L₁ = _biasfactor(eΓ, fσ.s, o + 1)
  B₁ = _biasconstant(L₁, f₁, o + 1)
  B₂ = f₂ === nothing ? NaN : _biasconstant(_biasfactor(eΓ, fσ.s, o + 2), f₂, o + 2)
  ℛ = 0.0
  if λ > 0
    e₁ = f₁.C \ [j == o + 1 ? 1.0 : 0.0 for j in 0:o + 1]
    ℛ = 3 * L₁^2 * _coefvar(d, k, f₁, e₁, vce, compat) / f₁.h^(2(o + 1))
  end
  return (V = V, B₁ = B₁, B₂ = B₂, ℛ = ℛ)
end

# e_ν'Γ⁻¹Λ for the moments Λ of order m of a fit, from eΓ = Γ⁻¹e_ν and the power
# sums s[j + 1] = ∑ wᵢ k(uᵢ) uᵢ^j
_biasfactor(eΓ, s, m) = sum(eΓ[j + 1] * s[m + j + 1] for j in 0:length(eΓ) - 1)

# the bias constant L μ⁽ᵐ⁾/m! in the units of x, from the factor L and the
# coefficient of order m of the fit f in its scaled variable
_biasconstant(L, f, m) = L * f.β[m + 1] / f.h^m

# A weighted fit of order o at c with bandwidth h over the observations with positive
# weight a:e, and whether the fit is identified; s holds the power sums ∑ wk(u) u^j up
# to j = 2o + extra. With `widen`, a window with too few distinct values is widened
# to twice the distance of the (o + 1)-th nearest distinct value, so that o + 1 of
# them have weights well above zero also for a kernel that vanishes at the edge
# (where two values tie in distance, the next one need not be farther): the last
# stage of the adaptive rule ended in an undefined bandwidth with four distinct
# values of x, where a fit exists. Fits that are identified are not changed.
function _lfit(d, k, c, h, o, extra, widen::Bool)
  f = _lfitat(d, k, c, h, o, extra)
  (f.ok || !widen || nunique(d) < o + 1) && return f
  floor = 2 * knn_distance(d, c, o + 1; distinct = true)
  return floor > h ? _lfitat(d, k, c, floor, o, extra) : f
end

function _lfitat(d, k, c, h, o, extra)
  r = _poswindow(d, k, c, h)
  s, t = zeros(2o + 1 + extra), zeros(o + 1)
  for i in r
    u = (d.x[i] - c) / h
    _powersums!(s, t, _weight(d.w, i) * k(u), u, d.y[i])
  end
  C, ok = _gramfactor([s[i + j + 1] for i in 0:o, j in 0:o], nunique(d, r))
  return (a = first(r), e = last(r), h = h, o = o, c = c, s = s, C = C, ok = ok,
          β = ok ? C \ t : fill(NaN, o + 1))
end

# ∑ᵢ (wᵢ k(uᵢ) g'r(uᵢ))² σ̂ᵢ² over the window of a fit, with σ̂ᵢ² from the variance
# estimator: nearest neighbours in the window, or the residuals of this fit
function _coefvar(d, k, f, g, vce, compat::Bool)
  a, e, h, o, c = f.a, f.e, f.h, f.o, f.c
  σ2 = vce isa NNVariance ? _nnvariances(d, a, e, vce.matches, _nntol(compat)) : nothing
  lev = vce isa HCVariance && vce.type >= 2 ? _leveragepoly(f.C, o) : nothing
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
        ε2 * _dofscale(n, o + 1)^2
      else
        hii = kw * evalpoly(u, lev)
        # no residual to estimate the variance from at a leverage of one
        1 - hii > sqrt(eps()) ? (vce.type == 2 ? ε2 / (1 - hii) : ε2 / (1 - hii)^2) : NaN
      end
    end
    v += (kw * evalpoly(u, g))^2 * s2
  end
  return v
end

# A minimizer of f on [a, b] by Brent's method (Brent 1973, ch. 5): golden-section
# search, with steps to the vertex of the parabola through the three best points when
# that step is short enough and stays inside the bracket. The accuracy at x is
# √eps |x| + tol / 3. It evaluates f at the same points as R's optimize(f, c(a, b),
# tol = tol), which the golden comparisons with nprobust, lpdensity and RDHonest need;
# for that, every comparison below, ties included, is the one of Brent's statement.
function _fmin(f, a, b; tol = eps()^0.25)
  isfinite(a) && isfinite(b) && a < b ||
    throw(ArgumentError("_fmin needs a finite interval [a, b] with a < b, got [$a, $b]"))
  ρ = (3 - sqrt(5.0)) / 2          # golden-section fraction, 2 - golden ratio
  rtol = sqrt(eps(Float64))
  atol = tol / 3
  lo, hi = Float64(a), Float64(b)  # the bracket
  # x: best point so far; w: second best; v: the previous w
  x = lo + ρ * (hi - lo)
  fx = f(x)
  w, fw = x, fx
  v, fv = x, fx
  d = zero(x)                      # the last step
  e = zero(x)                      # the step before it
  while true
    mid = (lo + hi) / 2
    tol1 = rtol * abs(x) + atol
    tol2 = 2 * tol1
    # the bracket is within tol2 of x on both sides
    abs(x - mid) <= tol2 - (hi - lo) / 2 && return x
    # vertex of the parabola through v, w and x, as the ratio p / q with q ≥ 0;
    # p = q = r = 0 falls through to a golden-section step
    p = q = r = zero(x)
    if abs(e) > tol1
      r = (x - w) * (fx - fv)
      q = (x - v) * (fx - fw)
      p = (x - v) * q - (x - w) * r
      q = 2 * (q - r)
      if q > 0
        p = -p
      else
        q = -q
      end
      r = e                        # r now holds the step before last
      e = d
    end
    if abs(p) >= abs(q * r / 2) || p <= q * (lo - x) || p >= q * (hi - x)
      # the parabolic step is not under half the step before last, or leaves the
      # bracket: golden section into the larger part of the bracket
      e = x < mid ? hi - x : lo - x
      d = ρ * e
    else
      d = p / q
      u = x + d
      # too near an end: a step of tol1 toward the middle instead
      if u - lo < tol2 || hi - u < tol2
        d = x < mid ? tol1 : -tol1
      end
    end
    # never evaluate within tol1 of x
    u = abs(d) >= tol1 ? x + d : (d > 0 ? x + tol1 : x - tol1)
    fu = f(u)
    if fu <= fx
      if u < x
        hi = x
      else
        lo = x
      end
      v, fv = w, fw
      w, fw = x, fx
      x, fx = u, fu
    else
      if u < x
        lo = u
      else
        hi = u
      end
      if fu <= fw || w == x
        v, fv = w, fw
        w, fw = u, fu
      elseif fu <= fv || v == x || v == w
        v, fv = u, fu
      end
    end
  end
end
