## Variance estimators and inference methods

"""
    VarianceEstimator

Supertype of the estimators of the conditional variances `σ²(xᵢ)` of the
responses that standard errors of local polynomial fits are built from:
[`NNVariance`](@ref) and [`HCVariance`](@ref).
"""
abstract type VarianceEstimator end

"""
    NNVariance(matches = 3)

Nearest-neighbour variance estimates `σ̂ᵢ² = Jᵢ / (Jᵢ + 1) (yᵢ - ȳᵢ)²`, where `ȳᵢ`
is the mean response of the `Jᵢ` nearest neighbours of `xᵢ` among the observations
in the window. The neighbours are at least `matches` observations, together with
all observations tied with `xᵢ` or with the farthest neighbour (Abadie and Imbens
2006; Calonico, Cattaneo and Titiunik 2014). This is the default of nprobust and
rdrobust. The estimates do not depend on the fitted polynomials, so conventional
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
  points::Vector{Float64}
  h::Vector{Float64}          # bandwidths used, after the floors
  b::Vector{Float64}          # pilot bandwidths used, after the floors
  estimate::Vector{Float64}
  se::Vector{Float64}
  estimate_bc::Vector{Float64}
  se_rbc::Vector{Float64}
  nwindow::Vector{Int}        # observations with positive weight in the larger window
  nunique::Vector{Int}        # distinct values with positive weight in the window of h
  floored::BitVector          # whether the floor raised h or b
end

"""
    fit(LocalPolynomialRegression, x, y; degree = 1, derivative = 0, kernel = EpanechnikovKernel(),
        bandwidth = RuleOfThumb(), pilot = nothing, rho = 1, points = nothing,
        vce = NNVariance(), minobs = 21, weights = nothing)
    fit(LocalPolynomialRegression, d::SmoothingData; kwargs...)

Fits local polynomials of order `degree` to the responses `y` at `points`, by
default 30 equally spaced points between the smallest and the largest `x`, and
estimates the regression function or its derivative of order `derivative`.

- `bandwidth` is a number, a vector with one bandwidth per point, or a
  [`BandwidthSelector`](@ref) for regression, by default [`RuleOfThumb`](@ref).
- `pilot` is the bandwidth `b` of the fit of order `degree + 1` that estimates the
  bias; by default `b = h / rho`, so that `rho = 1` uses the same bandwidth for
  both, as nprobust does.
- `vce` estimates the conditional variances: [`NNVariance`](@ref) (default) or
  [`HCVariance`](@ref).
- `minobs` raises `h` and `b` at each point, where needed, to the distance of the
  `minobs`-th nearest observation, so that every window holds enough data;
  `minobs = 0` turns this off. The fit records where the floor was binding.

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
function fit(::Type{LocalPolynomialRegression}, x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
             weights = nothing, kwargs...)
  return fit(LocalPolynomialRegression, SmoothingData(x, y; weights); kwargs...)
end

function fit(::Type{LocalPolynomialRegression}, d::SmoothingData; degree::Integer = 1,
             derivative::Integer = 0, kernel::AbstractKernel = EpanechnikovKernel(),
             bandwidth = RuleOfThumb(), pilot = nothing, rho::Real = 1, points = nothing,
             vce::VarianceEstimator = NNVariance(), minobs::Integer = 21)
  d.y === nothing && throw(ArgumentError(
    "local polynomial regression needs responses; prepare the data as SmoothingData(x, y)"))
  p, ν = Int(degree), Int(derivative)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $p"))
  0 <= ν <= p || throw(ArgumentError("derivative must be between 0 and the degree $p, got $ν"))
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  pts = points === nothing ? collect(range(d.x[1], d.x[end]; length = 30)) : _checkpoints(points)
  sel = _regression_selection(bandwidth, d, kernel, p, ν, pts)
  h = _pointbandwidths(sel.h, length(pts))
  b = if pilot !== nothing
    _pointbandwidths(pilot, length(pts))
  else
    rho > 0 || throw(ArgumentError("rho must be positive, or pass the pilot bandwidth"))
    h ./ rho
  end
  floored = _floor!(h, b, d, pts, minobs)
  fits = Vector{_LocalFit}(undef, length(pts))
  _foreachchunk(eachindex(pts)) do chunk
    for j in chunk
      fits[j] = _localfit(d, kernel, pts[j], h[j], b[j], p, ν, vce)
    end
  end
  return LocalPolynomialRegression(d, kernel, p, ν, vce, sel, Int(minobs), pts, h, b,
    [f.estimate for f in fits], [f.se for f in fits], [f.estimate_bc for f in fits],
    [f.se_rbc for f in fits], [f.nwindow for f in fits], [f.nunique for f in fits], floored)
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

_regression_selection(h::Union{Real,AbstractVector{<:Real}}, d, k, p, ν, pts) =
  BandwidthSelection(h isa Real ? _bandwidths(h, 1) : _bandwidths(h, length(pts)), nothing, NamedTuple())
_regression_selection(rule::Union{NormalReference,SilvermanRule}, d, k, p, ν, pts) =
  throw(ArgumentError("$rule chooses bandwidths for densities; use a regression rule such as RuleOfThumb()"))
_regression_selection(rule::BandwidthSelector, d, k, p, ν, pts) =
  selectbandwidth(rule, d, k; degree = p, derivative = ν, points = pts)
_regression_selection(h, d, k, p, ν, pts) = throw(ArgumentError(
  "bandwidth must be a positive number, a vector or a BandwidthSelector, got $(repr(h))"))

# raises h[j] and b[j] to the distance of the minobs-th nearest observation of the
# point, as nprobust's bwcheck does; returns where that changed something
function _floor!(h, b, d, points, minobs)
  floored = falses(length(points))
  minobs == 0 && return floored
  kmin = min(minobs, nobs(d))
  for j in eachindex(points)
    δ = knn_distance(d, points[j], kmin)
    if δ > h[j] || δ > b[j]
      floored[j] = true
      h[j], b[j] = max(h[j], δ), max(b[j], δ)
    end
  end
  return floored
end

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
# nprobust in these variables. The variances are ∑ ℓᵢ² σ̂ᵢ² and ∑ (ℓᵢᵇᶜ)² σ̂ᵢ².
function _localfit(d::SmoothingData, k::AbstractKernel, x0, h, b, p, ν, vce)
  xs, ys, n = d.x, d.y, nobs(d)
  q = p + 1
  # the observations with positive weight for the larger bandwidth, as in nprobust
  hb = max(h, b)
  lo, hi = support(k)
  a = isfinite(lo) ? _firsttrue(i -> (xs[i] - x0) / hb >= lo, n) : 1
  e = isfinite(hi) ? _firsttrue(i -> (xs[i] - x0) / hb > hi, n) - 1 : n
  while a <= e && k((xs[a] - x0) / hb) == 0
    a += 1
  end
  while e >= a && k((xs[e] - x0) / hb) == 0
    e -= 1
  end
  nwin = e - a + 1
  nan = _LocalFit(NaN, NaN, NaN, NaN, nwin, 0)
  nwin >= 1 || return nan
  # power sums: sh[j+1] = ∑ wk(u) u^j, th[j+1] = ∑ wk(u) u^j y, and the same in v
  sh, th = zeros(2p + 2), zeros(p + 1)
  sb, tb = zeros(2q + 1), zeros(q + 1)
  ah, eh = e + 1, a - 1   # the observations with positive weight for h
  for i in a:e
    t = xs[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    if kh > 0
      ah, eh = min(ah, i), i
    end
    _powersums!(sh, th, kh, u, ys[i])
    _powersums!(sb, tb, kb, v, ys[i])
  end
  nuniq = nunique(d, ah:eh)
  Gp = LinearAlgebra.Symmetric([sh[i + j + 1] for i in 0:p, j in 0:p])
  Gq = LinearAlgebra.Symmetric([sb[i + j + 1] for i in 0:q, j in 0:q])
  Cp = LinearAlgebra.cholesky(Gp; check = false)
  Cq = LinearAlgebra.cholesky(Gq; check = false)
  LinearAlgebra.issuccess(Cp) && LinearAlgebra.issuccess(Cq) ||
    return _LocalFit(NaN, NaN, NaN, NaN, nwin, nuniq)
  βp, γ = Cp \ th, Cq \ tb
  g = Cp \ [i == ν ? 1.0 : 0.0 for i in 0:p]
  c = Cq \ [i == p + 1 ? 1.0 : 0.0 for i in 0:q]
  ρ = (h / b)^(p + 1)
  s = sum(g[i + 1] * sh[p + i + 2] for i in 0:p)   # g'Λ
  scale = factorial(ν) / h^ν
  estimate = scale * βp[ν + 1]
  estimate_bc = scale * (βp[ν + 1] - ρ * s * γ[p + 2])
  # the variances
  σ2 = vce isa NNVariance ? _nnvariances(d, a, e, vce.matches) : nothing
  vh = vb = 0.0
  for i in a:e
    t = xs[i] - x0
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    ℓ = kh * evalpoly(u, g)
    ℓbc = ℓ - ρ * s * kb * evalpoly(v, c)
    s2h, s2b = _variances(vce, σ2, i - a + 1, ys[i] - evalpoly(u, βp), ys[i] - evalpoly(v, γ),
                          kh, u, Cp, nwin, p)
    vh += ℓ^2 * s2h
    vb += ℓbc^2 * s2b
  end
  return _LocalFit(estimate, scale * sqrt(vh), estimate_bc, scale * sqrt(vb), nwin, nuniq)
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

# the variance estimates for the conventional and the bias-corrected standard error
_variances(::NNVariance, σ2, l, εh, εb, kh, u, Cp, n, p) = (σ2[l], σ2[l])

function _variances(vce::HCVariance, σ2, l, εh, εb, kh, u, Cp, n, p)
  vce.type == 0 && return (εh^2, εb^2)
  vce.type == 1 && return (εh^2 * n / (n - p - 1), εb^2 * n / (n - p - 2))
  r = [u^j for j in 0:p]
  hii = kh * LinearAlgebra.dot(r, Cp \ r)   # the leverage in the fit of order p
  f = vce.type == 2 ? 1 / (1 - hii) : 1 / (1 - hii)^2
  return (εh^2 * f, εb^2 * f)
end

# Nearest-neighbour variance estimates for the observations a:e, with the
# neighbours taken among a:e: all observations tied with xᵢ, and then whole groups
# of tied observations, nearest first and both sides at equal distances, until
# there are at least J.
function _nnvariances(d::SmoothingData, a, e, J)
  xs, ys = d.x, d.y
  n = e - a + 1
  σ2 = Vector{Float64}(undef, n)
  groupsize(i) = (g = d.valueindex[i]; d.starts[g + 1] - d.starts[g])
  for pos in a:e
    g = d.valueindex[pos]
    l = pos - max(d.starts[g], a)          # neighbours on the left of pos
    r = min(d.starts[g + 1] - 1, e) - pos  # and on the right
    target = min(J, n - 1)
    while l + r < target
      left, right = pos - l - 1, pos + r + 1
      if left < a
        r += groupsize(right)
      elseif right > e
        l += groupsize(left)
      else
        dl, dr = xs[pos] - xs[left], xs[right] - xs[pos]
        dl >= dr && (r += groupsize(right))
        dl <= dr && (l += groupsize(left))
      end
    end
    lo, hi = pos - l, min(e, pos + r)
    m = hi - lo
    s = 0.0
    for i in lo:hi
      i == pos || (s += ys[i])
    end
    σ2[pos - a + 1] = m == 0 ? NaN : m / (m + 1) * (ys[pos] - s / m)^2
  end
  return σ2
end


## Interface

nobs(m::LocalPolynomialRegression) = nobs(m.data)
bandwidth(m::LocalPolynomialRegression) = m.h
StatsAPI.coef(m::LocalPolynomialRegression) = copy(m.estimate)

"""
    stderror(m::LocalPolynomialRegression; method = Conventional())

The standard errors at the points: of the estimates, or with
`method = RobustBiasCorrected()` of the bias-corrected estimates.
"""
stderror(m::LocalPolynomialRegression; method::InferenceMethod = Conventional()) =
  copy(method isa Conventional ? m.se : m.se_rbc)

"""
    confint(m::LocalPolynomialRegression; level = 0.95, method = RobustBiasCorrected())

Pointwise confidence intervals at the points, as a matrix with the lower bounds in
the first column and the upper bounds in the second. By default the robust
bias-corrected intervals, centred at the bias-corrected estimates; with
`method = Conventional()`, the intervals around the estimates, which cover only
when the bias is negligible.
"""
function StatsAPI.confint(m::LocalPolynomialRegression; level::Real = 0.95,
                          method::InferenceMethod = RobustBiasCorrected())
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  est, se = method isa Conventional ? (m.estimate, m.se) : (m.estimate_bc, m.se_rbc)
  return hcat(est .- z .* se, est .+ z .* se)
end

"""
    coeftable(m::LocalPolynomialRegression; level = 0.95)

The results at the points as a table, a named tuple of columns: `point`, the
bandwidths `h` and `b`, the observations in the window `nobs` and their distinct
values `nunique`, whether the floor on the window was binding `floored`, the
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
  h0 = m.selection.h
  h0 isa Float64 || throw(ArgumentError(
    "predict needs a fit with one bandwidth for all points; fit again at the new points"))
  pts = _checkpoints(x)
  h, b = fill(h0, length(pts)), fill(h0, length(pts))
  _floor!(h, b, m.data, pts, m.minobs)
  return _estimates(m.data, m.kernel, h, pts, m.degree, m.derivative)
end

# estimates of order p at the points from kernel sums
function _estimates(d, k, h, points, p, ν)
  S = kernelsums(d, k, h, points; kmax = 2p)
  T = kernelsums(d, k, h, points; kmax = p, z = d.y)
  out = Vector{Float64}(undef, length(points))
  for j in eachindex(points)
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric([S[i + l + 1, j] for i in 0:p, l in 0:p]); check = false)
    out[j] = LinearAlgebra.issuccess(C) ? factorial(ν) * (C \ T[:, j])[ν + 1] / h[j]^ν : NaN
  end
  return out
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
          m.derivative == 0 ? "" : ", derivative $(m.derivative)")
  println(io, "  kernel: ", m.kernel, ", variance: ", m.vce, ", ", nobs(m), " observations")
  m.selection.rule === nothing || println(io, "  bandwidth: ", m.selection.rule)
  any(m.floored) && println(io, "  the floor of ", m.minobs, " observations raised the bandwidth at ",
                            count(m.floored), " of ", length(m.points), " points")
  # each estimate next to its own standard error, and the robust interval next to the
  # bias-corrected estimate it is centred at
  r = confint(m)
  println(io, rpad("  point", 12), rpad("h", 10), rpad("n", 7), rpad("estimate", 11), rpad("se", 11),
          rpad("bias-corr.", 11), rpad("robust se", 11), "95% robust interval")
  rows = length(m.points) <= 12 ? eachindex(m.points) : [1:6; length(m.points)-5:length(m.points)]
  for j in rows
    length(m.points) > 12 && j == length(m.points) - 5 && println(io, "  ⋮")
    println(io, "  ", rpad(_fmt(m.points[j]), 10), rpad(_fmt(m.h[j]), 10), rpad(m.nwindow[j], 7),
            rpad(_fmt(m.estimate[j]), 11), rpad(_fmt(m.se[j]), 11), rpad(_fmt(m.estimate_bc[j]), 11),
            rpad(_fmt(m.se_rbc[j]), 11), "[", _fmt(r[j, 1]), ", ", _fmt(r[j, 2]), "]")
  end
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
global polynomial fit of order `p + 3`. The constant `C_{ν,p}(k)` depends on the
equivalent kernel, and `n` is the effective sample size. The global fit uses the
standardized covariate, so that the rule is equivariant under changes of location
and scale. When `p - ν` is even, the rule of order `p + 1` is used, which has the
same rate. The bandwidth is capped at the range of the data.
"""
struct RuleOfThumb <: BandwidthSelector end

function selectbandwidth(rule::RuleOfThumb, d::SmoothingData, k::AbstractKernel;
                         degree::Integer = 1, derivative::Integer = 0, points = nothing)
  d.y === nothing && throw(ArgumentError("RuleOfThumb needs responses; use SmoothingData(x, y)"))
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
# covariate; returns the residual variance and the weighted mean of the squared
# derivative of order r at the observations.
function _globalpilot(d::SmoothingData, P, r)
  x̄, s = _mean(d), sqrt(_variance(d))
  z = (d.x .- x̄) ./ s
  sw = d.w === nothing ? ones(nobs(d)) : sqrt.(d.w)
  X = [z[i]^j for i in eachindex(z), j in 0:P]
  α = LinearAlgebra.qr(sw .* X) \ (sw .* d.y)
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
