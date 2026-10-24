## Variance estimators and inference methods

"""
    VarianceEstimator

Supertype of the estimators of the conditional variances `σ²(xᵢ)` of the
responses that standard errors of local polynomial fits are built from:
[`NNVariance`](@ref), [`HCVariance`](@ref) and, for clustered data,
[`ClusterVariance`](@ref).
"""
abstract type VarianceEstimator end

"""
    NNVariance(matches = 3)

Nearest-neighbour variance estimates `σ̂ᵢ² = Jᵢ / (Jᵢ + 1) (yᵢ - ȳᵢ)²`, where `ȳᵢ`
is the mean response of the `Jᵢ` nearest neighbours of `xᵢ` among the observations
in the window. The neighbours are at least `matches` observations, together with
all observations tied with `xᵢ` or with the farthest neighbour (Abadie and Imbens
2006; Calonico, Cattaneo and Titiunik 2014). This is the default of nprobust and
rdrobust. Neighbours on the left and on the right at the same distance are both
taken, also when rounding makes the two distances differ in the last digits, as in
an equally spaced design, where it changed the standard errors by up to 9 percent
under a shift of the design; with `compat = true`, the distances are compared
exactly, as in nprobust. The estimates do not depend on the fitted polynomials, so conventional
and robust bias-corrected standard errors share them.
"""
struct NNVariance <: VarianceEstimator
  matches::Int

  function NNVariance(matches::Integer = 3)
    matches >= 1 || throw(ArgumentError("matches must be at least 1, got $matches"))
    new(matches)
  end
end

"""
    HCVariance(type)

Heteroskedasticity-consistent variance estimates from the residuals `ε̂ᵢ` of the
local fits: `ε̂ᵢ²` for `type = 0`, scaled by `n / (n - k)` for `type = 1`, by
`1 / (1 - hᵢᵢ)` for `type = 2` and by `1 / (1 - hᵢᵢ)²` for `type = 3`. Here `n`
is the number of observations in the window, `k` the number of coefficients, and
`hᵢᵢ` the leverage of observation `i` in the local fit of order `p`. Conventional
standard errors use the residuals of the fit of order `p` with bandwidth `h`,
robust bias-corrected ones those of the fit of order `p + 1` with bandwidth `b`,
both with the leverages of the first fit, as in nprobust.
"""
struct HCVariance <: VarianceEstimator
  type::Int

  function HCVariance(type::Integer)
    0 <= type <= 3 || throw(ArgumentError("type must be 0, 1, 2 or 3, got $type"))
    new(type)
  end
end

"""
    ClusterVariance(type = 1)

Cluster-robust variances, for observations that are independent across clusters
but may be dependent within them. The variance of an estimate `∑ ℓᵢ yᵢ` is
estimated by `c ∑_g (∑_{i ∈ g} ℓᵢ ε̃ᵢ)²`, a sum over the clusters `g` with
observations in the window, where `ε̃ᵢ` are the residuals `ε̂ᵢ` of the local fit,
adjusted for `type = 2` and `type = 3`:

- `type = 0`: `ε̃ = ε̂` and `c = 1`;
- `type = 1` (CR1, the default): `ε̃ = ε̂` and `c = (n - 1) / (n - k) ⋅ G / (G - 1)`;
- `type = 2` (CR2, Bell and McCaffrey 2002): `ε̃_g = (I - H_gg)^(-1/2) ε̂_g` and `c = 1`;
- `type = 3` (CR3): `ε̃_g = (I - H_gg)^(-1) ε̂_g` and `c = (G - 1) / G`.

Here `n` is the number of observations and `G` the number of clusters in the
window, `k` the number of coefficients, `ε̂_g` the residuals of cluster `g`, and
`H_gg` its block of the hat matrix of the local fit. The clusters come with the
data, `fit(LocalPolynomialRegression, x, y; cluster)` or
`SmoothingData(x, y; cluster)`, and with clusters the default variance estimator
is `ClusterVariance(1)`, as in nprobust. As for [`HCVariance`](@ref), conventional
standard errors use the residuals of the fit of order `p` with bandwidth `h`, and
robust bias-corrected ones those of the fit of order `p + 1` with bandwidth `b`,
both adjusted with the hat matrix of the first fit. With one observation per
cluster, types 0, 1 and 2 give the standard errors of `HCVariance(type)`, and type
3 those of `HCVariance(3)` times `√((n - 1) / n)`.

Standard errors are `NaN` where the window holds fewer than two clusters, and for
types 2 and 3 where one cluster alone determines part of the fit, so that
`I - H_gg` is singular. For the bias-corrected estimate, nprobust adjusts the
residuals with a matrix that changes with the units of `x`; `compat = true`
reproduces it, and its treatment of singular blocks. The plug-in bandwidth rules
estimate their variance constants in the same way; with `compat = true`, they use
nprobust's: the residuals of `HCVariance(type)` with the factor `c` of type 1.
"""
struct ClusterVariance <: VarianceEstimator
  type::Int

  function ClusterVariance(type::Integer = 1)
    0 <= type <= 3 || throw(ArgumentError("type must be 0, 1, 2 or 3, got $type"))
    new(type)
  end
end

# the variance estimator of a fit: by default nprobust's, nearest neighbours or, for
# clustered data, CR1; with clusters it must be cluster-robust
_variance(::Nothing, d::SmoothingData) = d.cluster === nothing ? NNVariance() : ClusterVariance()

function _variance(vce::VarianceEstimator, d::SmoothingData)
  if d.cluster === nothing
    vce isa ClusterVariance && throw(ArgumentError(
      "ClusterVariance needs the clusters of the observations; pass them as cluster"))
  else
    vce isa ClusterVariance || throw(ArgumentError("the data have clusters, so the variances " *
      "must be cluster-robust; use ClusterVariance(type), the counterpart of HCVariance(type)"))
  end
  return vce
end

"""
    InferenceMethod

Supertype of the ways to build pointwise confidence intervals:
[`Conventional`](@ref) and [`RobustBiasCorrected`](@ref).
"""
abstract type InferenceMethod end

"""
    Conventional()

Intervals `estimate ± z se` from the estimate and its standard error. They ignore
the smoothing bias, so they cover only for bandwidths small enough that the bias
is negligible, which bandwidths that minimize the mean squared error are not.
"""
struct Conventional <: InferenceMethod end

"""
    RobustBiasCorrected()

Intervals centred at the bias-corrected estimate, with a standard error that
includes the variability of the bias correction (Calonico, Cattaneo and Farrell
2018). They remain valid at bandwidths that minimize the mean squared error.
"""
struct RobustBiasCorrected <: InferenceMethod end


## Local polynomial regression

"""
    LocalPolynomialRegression <: RegressionModel

A local polynomial fit of a regression function or its derivative at a set of
points, with conventional and robust bias-corrected inference, fitted with
[`fit`](@ref)`(LocalPolynomialRegression, x, y)`.

At a point `x₀`, the fit of order `p` minimizes

    ∑ᵢ wᵢ k((xᵢ - x₀) / h) (yᵢ - β₀ - β₁ (xᵢ - x₀) - ⋯ - βₚ (xᵢ - x₀)^p)²,

and `ν! β̂_ν` estimates the `ν`-th derivative of the regression function at `x₀`.
The bias-corrected estimate subtracts the leading bias term, estimated by a fit of
order `p + 1` with the pilot bandwidth `b`, and its standard error includes the
variability of that estimate (Calonico, Cattaneo and Titiunik 2014; Calonico,
Cattaneo and Farrell 2018), as in nprobust.

`coef(m)` and `stderror(m)` give the conventional estimates and standard errors at
the points, [`confint`](@ref)`(m; level, method)` the intervals, `predict(m, x)`
the estimates at new points, and `coeftable(m)` everything as a table, that is, a
named tuple of columns.
"""
struct LocalPolynomialRegression{K<:AbstractKernel,D<:SmoothingData,V<:VarianceEstimator,
                                 S<:BandwidthSelection} <: StatsAPI.RegressionModel
  data::D
  kernel::K
  degree::Int
  derivative::Int
  vce::V
  selection::S                # the bandwidth before the floors
  minobs::Int                 # the floor: at least this many observations in a window
  compat::Bool                # nprobust's pilots, floors and nearest-neighbour ties
  points::Vector{Float64}
  h::Vector{Float64}          # bandwidths used, after the floors
  b::Vector{Float64}          # pilot bandwidths used, after the floors
  estimate::Vector{Float64}
  se::Vector{Float64}
  estimate_bc::Vector{Float64}
  se_rbc::Vector{Float64}
  nwindow::Vector{Int}        # observations with positive weight in the larger window, nprobust's N
  nunique::Vector{Int}        # distinct values with positive weight in the window of h
  floored::BitVector          # whether the floor raised h or b
end

"""
    fit(LocalPolynomialRegression, x, y; derivative = 0, degree = derivative + 1,
        kernel = EpanechnikovKernel(), bandwidth = nothing, pilot = nothing, rho = 1,
        points = nothing, vce = nothing, minobs = 21, weights = nothing, cluster = nothing,
        compat = false)
    fit(LocalPolynomialRegression, d::SmoothingData; kwargs...)

Fits local polynomials of order `degree` to the responses `y` at `points`, by
default 30 equally spaced points between the quantiles of order 0.025 and 0.975 of
`x` (between the smallest and the largest `x` with `compat = true`, as nprobust), and
estimates the regression function or its derivative of order `derivative`. The
default order, `derivative + 1`, is nprobust's: an odd `degree - derivative` has the
same order of bias at the boundary as in the interior.

- `bandwidth` is a number, a vector with one bandwidth per point, or a
  [`BandwidthSelector`](@ref) for regression. By default it follows nprobust:
  [`MSEOptimal`](@ref) at a single point and [`IMSEOptimal`](@ref) at several, with
  adaptive pilot bandwidths unless `compat = true`.
- `pilot` is the bandwidth `b` of the fit of order `degree + 1` that estimates the
  bias; by default `b = h / rho`, so that `rho = 1` uses the same bandwidth for
  both, as nprobust does. With `pilot = :selected`, the fit uses the pilot
  bandwidths that the bandwidth selector estimated.
- `vce` estimates the conditional variances: [`NNVariance`](@ref) (the default) or
  [`HCVariance`](@ref), or for clustered data [`ClusterVariance`](@ref) (the default
  with clusters).
- `cluster` gives the cluster of each observation, as identifiers of any type, for
  cluster-robust standard errors.
- `minobs` raises `h` and `b` at each point, where needed, to the distance of the
  `minobs`-th nearest observation, so that every window holds enough data, and
  further where ties leave too few distinct values with positive weight: `p + 1`
  in the window of `h` for the fit and `p + 2` in that of `b` for its bias
  correction. `minobs = 0` turns both off, and the fit records where they were
  binding. Estimates whose window still holds fewer distinct values than
  coefficients are `NaN`.
- `compat = true` gives the plug-in rules ([`MSEOptimal`](@ref),
  [`IMSEOptimal`](@ref) and [`CoverageOptimal`](@ref)) nprobust's pilot
  bandwidths instead of the adaptive ones, so that they agree with nprobust 1.0.0,
  and the floor and the nearest-neighbour variances are nprobust's: the floor
  counts observations only, and neighbours at distances that differ by rounding
  are not tied.

The estimates are exact sums over the windows; with weights, they are weighted
least-squares fits.

# Examples
```julia
julia> x = collect(range(0, 1; length = 200)); y = sin.(2π .* x) .+ 0.1 .* cos.(37 .* x);

julia> m = fit(LocalPolynomialRegression, x, y; bandwidth = 0.1, points = [0.25, 0.5]);

julia> round.(coef(m); digits = 3)
2-element Vector{Float64}:
 0.946
 0.014
```
"""
function fit(::Type{LocalPolynomialRegression}, x::AbstractVector, y::AbstractVector;
             weights = nothing, cluster = nothing, kwargs...)
  return fit(LocalPolynomialRegression, SmoothingData(x, y; weights, cluster); kwargs...)
end

function fit(::Type{LocalPolynomialRegression}, d::SmoothingData; derivative::Integer = 0,
             degree::Integer = derivative + 1, kernel::AbstractKernel = EpanechnikovKernel(),
             bandwidth = nothing, pilot = nothing, rho::Real = 1, points = nothing,
             vce::Union{Nothing,VarianceEstimator} = nothing, minobs::Integer = 21, compat::Bool = false)
  d.y === nothing && throw(ArgumentError(
    "local polynomial regression needs responses; prepare the data as SmoothingData(x, y)"))
  vce = _variance(vce, d)
  p, ν = Int(degree), Int(derivative)
  _check_degree(p, ν)
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  pts = points === nothing ? _regressionpoints(d, compat) : _checkpoints(points)
  sel = _anyselection(_regression_selection(bandwidth, d, kernel, p, ν, pts, vce, minobs; compat))
  h = _pointbandwidths(sel.h, length(pts))
  b = if pilot === :selected
    haskey(sel.details, :b) || throw(ArgumentError("pilot = :selected needs a rule that selects b, such as MSEOptimal()"))
    _pointbandwidths(sel.details.b, length(pts))
  elseif pilot !== nothing
    _pointbandwidths(pilot, length(pts))
  else
    0 < rho < Inf || throw(ArgumentError("rho must be positive and finite, or pass the pilot bandwidth"))
    h ./ rho
  end
  floored = _floor!(h, b, d, pts, minobs, compat ? nothing : p)
  fits = _pointwise(j -> _localfit(d, kernel, pts[j], h[j], b[j], p, ν, vce; compat), eachindex(pts))
  return LocalPolynomialRegression(d, kernel, p, ν, vce, sel, Int(minobs), compat, pts, h, b,
    [f.estimate for f in fits], [f.se for f in fits], [f.estimate_bc for f in fits],
    [f.se_rbc for f in fits], [f.nwindow for f in fits], [f.nunique for f in fits], floored)
end

function _check_degree(degree, derivative)
  degree >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  0 <= derivative <= degree ||
    throw(ArgumentError("derivative must be between 0 and the degree $degree, got $derivative"))
  return nothing
end

function _checkpoints(points)
  pts = points isa Real ? [Float64(points)] : Vector{Float64}(collect(points))
  isempty(pts) && throw(ArgumentError("points must not be empty"))
  all(isfinite, pts) || throw(ArgumentError("points must be finite"))
  return pts
end

_pointbandwidths(h::Real, m) = fill(_bandwidths(h, m), m)
_pointbandwidths(h::AbstractVector{<:Real}, m) = copy(_bandwidths(h, m))
_pointbandwidths(h, m) = throw(ArgumentError("a bandwidth must be a number or a vector, got $(repr(h))"))

_regression_selection(h::Union{Real,AbstractVector{<:Real}}, d, k, p, ν, pts, vce, minobs; compat = false) =
  BandwidthSelection(h isa Real ? _bandwidths(h, 1) : _bandwidths(h, length(pts)), nothing, NamedTuple())
_regression_selection(rule::Union{NormalReference,SilvermanRule,SheatherJones,LSCV}, d, k, p, ν, pts, vce, minobs;
                      compat = false) =
  throw(ArgumentError("$rule chooses bandwidths for densities; use a regression rule such as RuleOfThumb()"))
_regression_selection(rule::BandwidthSelector, d, k, p, ν, pts, vce, minobs; compat = false) =
  selectbandwidth(rule, d, k; degree = p, derivative = ν, points = pts, vce, minobs)
_regression_selection(::Nothing, d, k, p, ν, pts, vce, minobs; compat = false) =   # nprobust's default
  _regression_selection(length(pts) == 1 ? MSEOptimal() : IMSEOptimal(), d, k, p, ν, pts, vce, minobs; compat)
_regression_selection(h, d, k, p, ν, pts, vce, minobs; compat = false) = throw(ArgumentError(
  "bandwidth must be a positive number, a vector or a BandwidthSelector, got $(repr(h))"))

# Raises h[j] and b[j] to the distance of the minobs-th nearest observation of the
# point, as nprobust's bwcheck does, and for a fit of order p (nothing for none)
# further where the windows hold fewer than p + 1 distinct values with positive
# weight for h, or p + 2 for b, the fit of order p + 1: with heavy ties, 21
# observations can hold too few, and the estimates there were NaN. Returns where
# that changed something.
function _floor!(h, b, d, points, minobs, p = nothing)
  floored = falses(length(points))
  minobs == 0 && return floored
  kmin = min(minobs, nobs(d))
  for j in eachindex(points)
    δ = knn_distance(d, points[j], kmin)
    hj, bj = max(h[j], δ), max(b[j], δ)
    if p !== nothing
      hj = _distinctfloor(d, points[j], p + 1, hj)
      bj = _distinctfloor(d, points[j], p + 2, bj)
    end
    if hj > h[j] || bj > b[j]
      floored[j] = true
      h[j], b[j] = hj, bj
    end
  end
  return floored
end

# The half-width at x0 for which its m nearest distinct values have positive weight:
# h itself where they lie strictly inside its window, and otherwise 1.5 times the
# distance of the m-th, which puts that value at two thirds of the half-width, where
# every kernel weights it well (a kernel that vanishes at the edge gives a value at
# exactly the half-width no weight); h where there are fewer than m distinct values.
function _distinctfloor(d::SmoothingData, x0, m, h)
  m <= nunique(d) || return h
  D = knn_distance(d, x0, m; distinct = true)
  return D < h ? h : 1.5D
end

# the relative tolerance below which the distances of two nearest neighbours count
# as tied, and nprobust's exact comparison under compat
_nntol(compat::Bool) = compat ? 0.0 : sqrt(eps())

# the results at one point
struct _LocalFit
  estimate::Float64
  se::Float64
  estimate_bc::Float64
  se_rbc::Float64
  nwindow::Int
  nunique::Int
end

# The fits of order p with bandwidth h and of order q = p + 1 with bandwidth b at
# x0, in the scaled variables u = (x - x0) / h and v = (x - x0) / b. With Γ the
# Gram matrices, Λ = ∑ wᵢ k(uᵢ) uᵢ^(p+1) r_p(uᵢ), g = Γ_p⁻¹ e_ν and c = Γ_q⁻¹ e_(p+1),
# the estimate is ∑ ℓᵢ yᵢ with ℓᵢ = wᵢ k(uᵢ) g'r_p(uᵢ), and the bias-corrected one
# ∑ ℓᵢᵇᶜ yᵢ with ℓᵢᵇᶜ = ℓᵢ - (h/b)^(p+1) g'Λ wᵢ k(vᵢ) c'r_q(vᵢ), the modified design of
# nprobust in these variables. With residuals ε̂ᵢ from the variance estimator, the
# variances are ∑ (ℓᵢ ε̂ᵢ)² and ∑ (ℓᵢᵇᶜ ε̂ᵢᵇ)², and these products are also the
# vectors whose inner products give the covariances between points.
function _localsetup(d::SmoothingData, k::AbstractKernel, x0, h, b, p, ν)
  xs, ys = d.x, d.y
  q = p + 1
  # the observations with positive weight for the larger bandwidth, as in nprobust
  r = _poswindow(d, k, x0, max(h, b))
  a, e = first(r), last(r)
  e >= a || return (a = a, e = e, ok = false, nuniq = 0)
  # power sums: sh[j+1] = ∑ wk(u) u^j, th[j+1] = ∑ wk(u) u^j y, and the same in v
  sh, th = zeros(2p + 2), zeros(p + 1)
  sb, tb = zeros(2q + 1), zeros(q + 1)
  ah, eh = e + 1, a - 1   # the observations with positive weight for h
  ab, eb = e + 1, a - 1   # and for b
  for i in a:e
    t = xs[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    if kh > 0
      ah, eh = min(ah, i), i
    end
    if kb > 0
      ab, eb = min(ab, i), i
    end
    _powersums!(sh, th, kh, u, ys[i])
    _powersums!(sb, tb, kb, v, ys[i])
  end
  nuniq = nunique(d, ah:eh)
  Cp, okp = _gramfactor([sh[i + j + 1] for i in 0:p, j in 0:p], nuniq)
  okp || return (a = a, e = e, ok = false, nuniq = nuniq)
  # without the fit of order q, only the bias-corrected results are missing
  Cq, okq = _gramfactor([sb[i + j + 1] for i in 0:q, j in 0:q], nunique(d, ab:eb))
  g = Cp \ [i == ν ? 1.0 : 0.0 for i in 0:p]
  return (a = a, e = e, ok = true, nuniq = nuniq, Cp = Cp, βp = Cp \ th,
          γ = okq ? Cq \ tb : fill(NaN, q + 1), g = g,
          c = okq ? Cq \ [i == p + 1 ? 1.0 : 0.0 for i in 0:q] : fill(NaN, q + 1),
          ρ = (h / b)^(p + 1), s = sum(g[i + 1] * sh[p + i + 2] for i in 0:p),
          scale = factorial(ν) / h^ν)
end

# the observations with positive weight at x0 for a kernel that is nonnegative on its
# support
function _poswindow(d::SmoothingData, k::AbstractKernel, x0, h)
  xs, n = d.x, nobs(d)
  lo, hi = support(k)
  a = isfinite(lo) ? _firsttrue(i -> (xs[i] - x0) / h >= lo, n) : 1
  e = isfinite(hi) ? _firsttrue(i -> (xs[i] - x0) / h > hi, n) - 1 : n
  while a <= e && k((xs[a] - x0) / h) == 0
    a += 1
  end
  while e >= a && k((xs[e] - x0) / h) == 0
    e -= 1
  end
  return a:e
end

# The Cholesky factorization of the Gram matrix G of a local polynomial fit, and
# whether the fit is identified. With fewer distinct values of positive weight than
# coefficients G is singular, although rounding can let its factorization succeed;
# a pivot that is a tiny fraction of its diagonal entry then marks the powers that
# are collinear. A matrix whose pivots are all above 1e-8 of their diagonal entries
# cannot come from too few distinct values, so that their number, `distinct` (a
# number, or a function that counts them), is only needed below that.
function _gramfactor(G, distinct)
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(G); check = false)
  LinearAlgebra.issuccess(C) || return C, false
  r = minimum(j -> C.factors[j, j]^2 / G[j, j], axes(G, 1))
  return C, r > 1e-8 || (r > 1e-12 && _count(distinct) >= size(G, 1))
end

_count(n::Integer) = n
_count(f) = f()

# The estimates and standard errors at x0; with `vectors = true` also the vectors
# scale ℓᵢ ε̂ᵢ and scale ℓᵢᵇᶜ ε̂ᵢᵇ over the window
function _localfit(d::SmoothingData, k::AbstractKernel, x0, h, b, p, ν, vce; vectors::Bool = false,
                   setup = nothing, compat::Bool = false)
  f = setup === nothing ? _localsetup(d, k, x0, h, b, p, ν) : setup
  nntol = _nntol(compat)
  nwin = f.e - f.a + 1   # the observations with positive weight for the larger bandwidth
  if !f.ok
    fit = _LocalFit(NaN, NaN, NaN, NaN, nwin, f.nuniq)
    return vectors ? (fit = fit, range = f.a:f.e, aus = fill(NaN, nwin), arb = fill(NaN, nwin)) : fit
  end
  xs, ys = d.x, d.y
  a, e, βp, γ, g, c, ρ, s, scale = f.a, f.e, f.βp, f.γ, f.g, f.c, f.ρ, f.s, f.scale
  r = vce isa NNVariance ? _nnresiduals(d, a, e, vce.matches, nntol) : nothing
  lev = vce isa HCVariance && vce.type >= 2 ? _leveragepoly(f.Cp, p) : nothing
  aus = vectors ? Vector{Float64}(undef, nwin) : Float64[]
  arb = vectors ? Vector{Float64}(undef, nwin) : Float64[]
  vh = vb = 0.0
  for i in a:e
    t = xs[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    ℓ = kh * evalpoly(u, g)
    ℓbc = ℓ - ρ * s * kb * evalpoly(v, c)
    rh, rb = _residuals(vce, r, i - a + 1, ys[i] - evalpoly(u, βp), ys[i] - evalpoly(v, γ),
                        kh, u, lev, nwin, p)
    vh += (ℓ * rh)^2
    vb += (ℓbc * rb)^2
    if vectors
      aus[i - a + 1] = scale * ℓ * rh
      arb[i - a + 1] = scale * ℓbc * rb
    end
  end
  fit = _LocalFit(scale * βp[ν + 1], scale * sqrt(vh), scale * (βp[ν + 1] - ρ * s * γ[p + 2]),
                  scale * sqrt(vb), nwin, f.nuniq)
  return vectors ? (fit = fit, range = a:e, aus = aus, arb = arb) : fit
end

# adds kw u^j to s[j+1] for j = 0, …, length(s) - 1 and kw u^j y to t[j+1]
function _powersums!(s, t, kw, u, y)
  kw == 0 && return
  v = kw
  for j in eachindex(s)
    s[j] += v
    j <= length(t) && (t[j] += v * y)
    v *= u
  end
  return
end

# the residuals for the conventional and the bias-corrected standard error, scaled
# so that their squares are the variance estimates
_residuals(::NNVariance, r, l, εh, εb, kh, u, lev, n, p) = (r[l], r[l])

function _residuals(vce::HCVariance, r, l, εh, εb, kh, u, lev, n, p)
  vce.type == 0 && return (εh, εb)
  vce.type == 1 && return (εh * _dofscale(n, p + 1), εb * _dofscale(n, p + 2))
  hii = kh * evalpoly(u, lev)   # the leverage in the fit of order p
  # a leverage of one, which rounding can put just above one, leaves no residual to
  # estimate the variance from
  1 - hii > sqrt(eps()) || return (NaN, NaN)
  f = vce.type == 2 ? 1 / sqrt(1 - hii) : 1 / (1 - hii)
  return (εh * f, εb * f)
end

# The leverage of an observation at u in a fit of order p, divided by its kernel
# weight, r(u)' Γ⁻¹ r(u) with r(u) = (1, u, …, u^p), as the coefficients of a
# polynomial in u of degree 2p, from the Cholesky factorization C of Γ, so that the
# leverages need no solve and no allocation per observation
function _leveragepoly(C, p)
  Γinv = C \ Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1)
  lev = zeros(2p + 1)
  for a in 0:p, b in 0:p
    lev[a + b + 1] += Γinv[a + 1, b + 1]
  end
  return lev
end

# √(n / (n - k)) for n observations and k coefficients, or NaN without residual
# degrees of freedom
_dofscale(n, k) = n > k ? sqrt(n / (n - k)) : NaN

_nnvariances(d::SmoothingData, a, e, J, tol) = _nnresiduals(d, a, e, J, tol) .^ 2

# Nearest-neighbour residuals √(J/(J+1)) (yᵢ - ȳᵢ) for the observations a:e, with the
# neighbours taken among a:e: all observations tied with xᵢ, and then whole groups
# of tied observations, nearest first and both sides at equal distances, until
# there are at least J; distances within the relative tolerance tol count as equal.
function _nnresiduals(d::SmoothingData, a, e, J, tol)
  xs, ys = d.x, d.y
  n = e - a + 1
  σ2 = Vector{Float64}(undef, n)
  groupsize(i) = (g = d.valueindex[i]; d.starts[g + 1] - d.starts[g])
  target = min(J, n - 1)
  ctarget = sqrt(target / (target + 1))
  for pos in a:e
    g = d.valueindex[pos]
    l = pos - max(d.starts[g], a)          # neighbours on the left of pos
    r = min(d.starts[g + 1] - 1, e) - pos  # and on the right
    while l + r < target
      left, right = pos - l - 1, pos + r + 1
      if left < a
        r += groupsize(right)
      elseif right > e
        l += groupsize(left)
      else
        dl, dr = xs[pos] - xs[left], xs[right] - xs[pos]
        tied = abs(dl - dr) <= tol * max(dl, dr)
        (dl >= dr || tied) && (r += groupsize(right))
        (dl <= dr || tied) && (l += groupsize(left))
      end
    end
    lo, hi = pos - l, min(e, pos + r)
    m = hi - lo
    s = 0.0
    for i in lo:pos-1
      s += ys[i]
    end
    for i in pos+1:hi
      s += ys[i]
    end
    c = m == 0 ? NaN : m == target ? ctarget : sqrt(m / (m + 1))
    σ2[pos - a + 1] = c * (ys[pos] - s / m)
  end
  return σ2
end


## Interface

nobs(m::LocalPolynomialRegression) = nobs(m.data)
bandwidth(m::LocalPolynomialRegression) = m.h
"""
    coef(m; method = Conventional())

The estimates at the points of a fit (`LocalPolynomialRegression`,
[`PointwiseDensity`](@ref) or [`LocalPolynomialDensity`](@ref)), or with
`method = RobustBiasCorrected()` the bias-corrected estimates, the centres of the
default intervals of [`confint`](@ref). By default `coef`, `stderror` and `vcov`
describe the conventional estimate, which has the smaller mean squared error, and
`confint` the robust bias-corrected interval, which covers (Calonico, Cattaneo and
Farrell 2018); with the same `method`, `coef(m; method) .± z .* stderror(m; method)`
is `confint(m; method)`.
"""
StatsAPI.coef(m::LocalPolynomialRegression; method::InferenceMethod = Conventional()) =
  copy(_usesbc(method) ? m.estimate_bc : m.estimate)

"""
    stderror(m::LocalPolynomialRegression; method = Conventional())

The standard errors at the points: of the estimates, or with
`method = RobustBiasCorrected()` of the bias-corrected estimates.
"""
stderror(m::LocalPolynomialRegression; method::InferenceMethod = Conventional()) =
  copy(_usesbc(method) ? m.se_rbc : m.se)

"""
    confint(m::LocalPolynomialRegression; level = 0.95, method = RobustBiasCorrected())

Pointwise confidence intervals at the points, as a matrix with the lower bounds in
the first column and the upper bounds in the second. By default the robust
bias-corrected intervals, centred at the bias-corrected estimates; with
`method = Conventional()`, the intervals around the estimates, which cover only
when the bias is negligible.
"""
StatsAPI.confint(m::LocalPolynomialRegression; level::Real = 0.95,
                 method::InferenceMethod = RobustBiasCorrected()) = _confint(m, level, method)

"""
    coeftable(m::LocalPolynomialRegression; level = 0.95)

The results at the points as a table, a named tuple of columns: `point`, the
bandwidths `h` and `b`, the observations with positive weight in the window of the
larger bandwidth `nobs`, nprobust's `N`, the distinct values with positive weight in
the window of `h` `nunique`, whether the floor on the window was binding `floored`, the
`estimate` with its standard error `se` and conventional interval
`lower`, `upper`, and the bias-corrected `estimate_bc` with its robust standard
error `se_rbc` and interval `lower_rbc`, `upper_rbc`.
"""
function StatsAPI.coeftable(m::LocalPolynomialRegression; level::Real = 0.95)
  c = confint(m; level, method = Conventional())
  r = confint(m; level, method = RobustBiasCorrected())
  return (point = copy(m.points), h = copy(m.h), b = copy(m.b), nobs = copy(m.nwindow),
          nunique = copy(m.nunique), floored = copy(m.floored), estimate = copy(m.estimate),
          se = copy(m.se), lower = c[:, 1], upper = c[:, 2], estimate_bc = copy(m.estimate_bc),
          se_rbc = copy(m.se_rbc), lower_rbc = r[:, 1], upper_rbc = r[:, 2])
end

"""
    predict(m::LocalPolynomialRegression, x)

The estimates of the fit at new points `x`, with the same kernel, order and
bandwidth, and the same floor on the windows. The sums are computed by
[`kernelsums`](@ref) for all points at once. Needs a fit with one bandwidth for all
points.
"""
function StatsAPI.predict(m::LocalPolynomialRegression, x::AbstractVector{<:Real})
  hs = m.selection.h
  # a pointwise rule at a single point also selects one bandwidth
  hs isa Float64 || allequal(hs) || throw(ArgumentError(
    "predict needs a fit with one bandwidth for all points; fit again at the new points"))
  h0 = first(hs)
  pts = _checkpoints(x)
  h, b = fill(h0, length(pts)), fill(h0, length(pts))
  _floor!(h, b, m.data, pts, m.minobs, m.compat ? nothing : m.degree)
  return _estimates(m.data, m.kernel, h, pts, m.degree, m.derivative)
end

# estimates of order p at the points from kernel sums
function _estimates(d, k, h, points, p, ν)
  S = kernelsums(d, k, h, points; kmax = 2p)
  T = kernelsums(d, k, h, points; kmax = p, z = d.y)
  out = Vector{Float64}(undef, length(points))
  for j in eachindex(points)
    C, ok = _gramfactor([S[i + l + 1, j] for i in 0:p, l in 0:p], () -> nunique(d, _poswindow(d, k, points[j], h[j])))
    out[j] = ok ? factorial(ν) * (C \ T[:, j])[ν + 1] / h[j]^ν : NaN
  end
  return out
end


## Local polynomial smoothing, without inference

"""
    localpolynomial(x, y, points; bandwidth, derivative = 0, degree = derivative + 1,
                    kernel = EpanechnikovKernel(), weights = nothing)
    localpolynomial(d::SmoothingData, points; bandwidth, kwargs...)

The local polynomial estimates of the regression function, or of its derivative of
order `derivative`, at `points`, with a given `bandwidth`, a number or one per
point: the estimates of [`LocalPolynomialRegression`](@ref), without floors on the
windows and without standard errors. Estimates whose window holds fewer distinct
values of `x` with positive weight than coefficients are `NaN`. The sums come from
[`kernelsums`](@ref) for all points at once, so that smoothing many curves, or
building other estimators on local fits, costs little. [`smoothermatrix`](@ref)
gives the weights of the estimates.

# Examples
```julia
julia> x = collect(range(0, 1; length = 200)); y = sin.(2π .* x);

julia> round.(localpolynomial(x, y, [0.25, 0.5]; bandwidth = 0.1); digits = 3)
2-element Vector{Float64}:
 0.961
 0.0
```
"""
function localpolynomial(x::AbstractVector, y::AbstractVector, points; weights = nothing, kwargs...)
  return localpolynomial(SmoothingData(x, y; weights), points; kwargs...)
end

function localpolynomial(d::SmoothingData, points; bandwidth, derivative::Integer = 0,
                         degree::Integer = derivative + 1, kernel::AbstractKernel = EpanechnikovKernel())
  d.y === nothing && throw(ArgumentError("local polynomial smoothing needs responses; use SmoothingData(x, y)"))
  p, ν = Int(degree), Int(derivative)
  _check_degree(p, ν)
  pts = _checkpoints(points)
  return _estimates(d, kernel, _pointbandwidths(bandwidth, length(pts)), pts, p, ν)
end

"""
    smoothermatrix(x, points; bandwidth, derivative = 0, degree = derivative + 1,
                   kernel = EpanechnikovKernel(), weights = nothing)
    smoothermatrix(d::SmoothingData, points; bandwidth, kwargs...)

The matrix `L` of the linear smoother of [`localpolynomial`](@ref): the estimate at
`points[j]` is `∑ᵢ L[j, i] yᵢ`, so that `L * y` gives the estimates. The columns
follow the observations in the input order of `x` (observations with zero weight
get zero columns), or for `d::SmoothingData` the sorted order of `d.x`. For the
regression function, every row sums to one and reproduces polynomials of order
`degree` exactly. Rows of estimates that are not identified are `NaN`.
"""
function smoothermatrix(x::AbstractVector, points; weights = nothing, kwargs...)
  d = SmoothingData(x; weights)
  Ls = smoothermatrix(d, points; kwargs...)
  L = zeros(size(Ls, 1), length(x))
  L[:, d.perm] = Ls
  return L
end

function smoothermatrix(d::SmoothingData, points; bandwidth, derivative::Integer = 0,
                        degree::Integer = derivative + 1, kernel::AbstractKernel = EpanechnikovKernel())
  p, ν = Int(degree), Int(derivative)
  _check_degree(p, ν)
  pts = _checkpoints(points)
  h = _pointbandwidths(bandwidth, length(pts))
  rows = _pointwise(j -> _smootherrow(d, kernel, pts[j], h[j], p, ν), eachindex(pts))
  L = zeros(length(pts), nobs(d))
  for j in eachindex(rows)
    r, ℓ = rows[j]
    L[j, r] = ℓ
  end
  return L
end

# the weights of the estimate at x0 over the observations r with positive weight
function _smootherrow(d, k, x0, h, p, ν)
  r = _poswindow(d, k, x0, h)
  S = zeros(2p + 1)
  for i in r
    u = (d.x[i] - x0) / h
    _powersums!(S, _weight(d.w, i) * k(u), u)
  end
  C, ok = _gramfactor([S[i + j + 1] for i in 0:p, j in 0:p], () -> nunique(d, r))
  ok || return (1:nobs(d), fill(NaN, nobs(d)))
  g = C \ [j == ν ? 1.0 : 0.0 for j in 0:p]
  scale = factorial(ν) / h^ν
  return (r, [scale * _weight(d.w, i) * k((d.x[i] - x0) / h) * evalpoly((d.x[i] - x0) / h, g) for i in r])
end

"""
    fitted(m::LocalPolynomialRegression)

The estimates of the regression function at the observations, in their input order
(observations with zero weight are left out), with the bandwidth of the fit. Needs
`derivative = 0` and one bandwidth for all points.
"""
function StatsAPI.fitted(m::LocalPolynomialRegression)
  m.derivative == 0 || throw(ArgumentError("fitted values need a fit of the regression function, derivative = 0"))
  d = m.data
  return predict(m, d.x)[sortperm(d.perm)]
end

"""
    residuals(m::LocalPolynomialRegression)

The responses minus the [`fitted`](@ref) values, in the input order of the
observations.
"""
function StatsAPI.residuals(m::LocalPolynomialRegression)
  d = m.data
  return d.y[sortperm(d.perm)] .- fitted(m)
end

function Base.show(io::IO, m::LocalPolynomialRegression)
  print(io, "LocalPolynomialRegression(degree ", m.degree, ", derivative ", m.derivative, ", ",
        m.kernel, ", ", length(m.points), " points, ", nobs(m), " observations)")
end

function Base.show(io::IO, ::MIME"text/plain", m::LocalPolynomialRegression)
  println(io, "Local polynomial regression of degree ", m.degree,
          m.derivative == 0 ? "" : ", derivative $(m.derivative)", m.compat ? " (nprobust compatibility)" : "")
  println(io, "  kernel: ", m.kernel, ", variance: ", m.vce, ", ", nobs(m), " observations",
          m.data.cluster === nothing ? "" : " in $(_nclusters(m.data)) clusters")
  rule = m.selection.rule
  rule === nothing || println(io, "  bandwidth: ", rule, !(rule isa Union{MSEOptimal,IMSEOptimal,CoverageOptimal}) ? "" :
                              m.compat ? ", nprobust's pilots" : ", pilots anchored on cross-validation")
  any(m.floored) && println(io, "  the floor of ", m.minobs, " observations raised the bandwidth at ",
                            count(m.floored), " of ", length(m.points), " points")
  _showtable(io, m)
end

_fmt(x) = string(round(x; sigdigits = 4))


## Rule of thumb

"""
    RuleOfThumb()

The rule-of-thumb bandwidth of Fan and Gijbels (1996, Section 4.2) for local
polynomial regression of order `p` and derivative `ν`,

    h = C_{ν,p}(k) (σ̃² (max x - min x) / (n D))^(1/(2p+3)),

the bandwidth that minimizes the asymptotic integrated mean squared error when the
variance `σ̃²` and the mean squared derivative `D` of order `p + 1` come from a
global polynomial fit of order `p + 3`, or of lower order when `x` has fewer than
`p + 4` distinct values. The constant `C_{ν,p}(k)` depends on the equivalent
kernel, and `n` is the effective sample size. The global fit uses the standardized
covariate, so that the rule is equivariant under changes of location and scale.
When `p - ν` is even, the rule of order `p + 1` is used, which has the same rate.
The bandwidth is capped at the range of the data, which it reaches when the
derivative of the global fit vanishes.
"""
struct RuleOfThumb <: BandwidthSelector end

_selection(rule::Union{RuleOfThumb,CrossValidation}, d, k) =
  throw(ArgumentError("$rule chooses bandwidths for regression; use a density rule such as SheatherJones()"))

# (the fits pass vce and minobs to every regression rule; this one does not use them)
function selectbandwidth(rule::RuleOfThumb, d::SmoothingData, k::AbstractKernel;
                         derivative::Integer = 0, degree::Integer = derivative + 1, points = nothing,
                         vce::Union{Nothing,VarianceEstimator} = nothing, minobs::Integer = 0)
  d.y === nothing && throw(ArgumentError("RuleOfThumb needs responses; use SmoothingData(x, y)"))
  _check_degree(degree, derivative)
  ν = Int(derivative)
  p = iseven(degree - ν) ? Int(degree) + 1 : Int(degree)   # p - ν odd
  B, V = _equivalentkernel(k, p, ν)
  C = (factorial(p + 1)^2 * (2ν + 1) * V / (2 * (p + 1 - ν) * B^2))^(1 / (2p + 3))
  σ2, D = _globalpilot(d, p + 3, p + 1)
  n = effective_nobs(d)
  R = d.x[end] - d.x[1]
  R > 0 || throw(ArgumentError("the data have no spread, so RuleOfThumb cannot choose a bandwidth"))
  h = D > 0 ? min(C * (σ2 * R / (n * D))^(1 / (2p + 3)), R) : R
  return BandwidthSelection(h, rule, (constant = C, variance = σ2, curvature = D, n = n))
end

# ∫ t^(p+1) K*_ν(t) dt and ∫ K*_ν(t)² dt for the equivalent kernel
# K*_ν(t) = e_ν' S⁻¹ (1, t, …, t^p) k(t), S = (μ_(i+j)) (Fan and Gijbels 1996, Section 3.2.2)
function _equivalentkernel(k, p, ν)
  μ = [moment(k, j) for j in 0:2p+1]
  k2 = _square(k)
  μ2 = [moment(k2, j) for j in 0:2p]
  S = [μ[i + j + 1] for i in 0:p, j in 0:p]
  g = S \ [i == ν ? 1.0 : 0.0 for i in 0:p]
  B = sum(g[i + 1] * μ[p + i + 2] for i in 0:p)
  V = sum(g[i + 1] * g[j + 1] * μ2[i + j + 1] for i in 0:p, j in 0:p)
  return B, V
end

# A weighted least-squares fit of a polynomial of order P in the standardized
# covariate, or of the highest order that the distinct values identify; returns the
# residual variance and the weighted mean of the squared derivative of order r at
# the observations.
function _globalpilot(d::SmoothingData, P, r)
  P = min(P, nunique(d) - 1)
  x̄, s = _mean(d), sqrt(_variance(d))
  z = (d.x .- x̄) ./ s
  sw = d.w === nothing ? ones(nobs(d)) : sqrt.(d.w)
  X = [z[i]^j for i in eachindex(z), j in 0:P]
  α = (LinearAlgebra.qr(sw .* X) \ (sw .* d.y))::Vector{Float64}
  res = d.y .- X * α
  n = effective_nobs(d)
  W = d.sumw
  σ2 = sum(_weight(d.w, i) * res[i]^2 for i in eachindex(res)) / W * n / max(n - P - 1, 1)
  # the derivative of order r of ∑ αⱼ z^j, divided by s^r
  dr = [sum(α[j + 1] * prod(j-r+1:j) * z[i]^(j - r) for j in r:P; init = 0.0) / s^r
        for i in eachindex(z)]
  D = sum(_weight(d.w, i) * dr[i]^2 for i in eachindex(dr)) / W
  return σ2, D
end
