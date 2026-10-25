## Robust bias-corrected inference for densities (Calonico, Cattaneo and Farrell 2018)

"""
    PointwiseDensity <: StatisticalModel

Kernel density estimates at a set of points with conventional and robust
bias-corrected inference, fitted with [`fit`](@ref)`(PointwiseDensity, x)`.

At a point `x₀`, the estimate is `f̂(x₀) = ∑ᵢ wᵢ k((x₀ - xᵢ)/h) / (h W)` with
`W = ∑ᵢ wᵢ`. The bias-corrected estimate subtracts its leading bias
`h² μ₂(k)/2 f''(x₀)`, with `f''` estimated at the pilot bandwidth `b`:

    f̂ᵇᶜ(x₀) = ∑ᵢ wᵢ M((x₀ - xᵢ)/h) / (h W),   M(u) = k(u) - ρ³ μ₂(k)/2 D(ρu),   ρ = h/b,

where `D(u) = k(u) (c₀ + c₂ u²)` is the kernel of the second derivative in a local
quadratic fit with kernel `k`. With `ρ = 1`, `M` is a fourth-order kernel. The
standard errors estimate the exact variances of the two linear estimators, so that
the robust one includes the variability of the bias estimate. The estimates are
inconsistent at boundary points, so evaluate in the interior.

`coef(m; method)`, `stderror(m; method)`, `confint(m; level, method)` and `coeftable(m)`
work as for [`LocalPolynomialRegression`](@ref); the column `nobs` of `coeftable` counts the
observations with positive weight in the larger of the windows of `h` and `b`, as
for regression, or with `compat = true` all those within `max(h, b)`, as nprobust.
"""
struct PointwiseDensity{K<:AbstractKernel,D<:SmoothingData,S<:BandwidthSelection} <: StatsAPI.StatisticalModel
  data::D
  kernel::K
  selection::S
  minobs::Int
  compat::Bool
  points::Vector{Float64}
  h::Vector{Float64}
  b::Vector{Float64}
  estimate::Vector{Float64}
  se::Vector{Float64}
  estimate_bc::Vector{Float64}
  se_rbc::Vector{Float64}
  nwindow::Vector{Int}   # observations with positive weight in the larger window; within it under compat
  floored::BitVector
end

"""
    fit(PointwiseDensity, x; points = nothing, kernel = EpanechnikovKernel(), bandwidth = nothing,
        pilot = nothing, rho = 1, minobs = 21, weights = nothing, compat = false)
    fit(PointwiseDensity, d::SmoothingData; kwargs...)

Estimates the density of the sample `x` at `points`, by default the quantiles of
order 0.1, …, 0.9 at 30 equally spaced levels, with conventional and robust
bias-corrected standard errors (Calonico, Cattaneo and Farrell 2018), as nprobust's
`kdrobust` does.

- `bandwidth` is a number, a vector with one bandwidth per point, or a
  [`BandwidthSelector`](@ref). By default it is the plug-in bandwidth,
  [`MSEOptimal`](@ref) at one point and [`IMSEOptimal`](@ref) at several, with
  pilots that adapt to the density through the Sheather–Jones bandwidth.
- `pilot` is the bandwidth `b` of the estimate of `f''`; by default `b = h / rho`.
  With `pilot = :selected`, the pilot bandwidth of the plug-in rule is used.
- `minobs` raises `h` and `b` at each point to the distance of the `minobs`-th
  nearest observation, where needed; `minobs = 0` turns this off.
- `compat = true` reproduces nprobust 1.0.0: its bias correction uses the constant
  `∫ u⁴ D(u) du / 24` instead of `μ₂(k)/2`, which removes only 5/9 of the leading
  bias for the Epanechnikov kernel, and its plug-in bandwidths use normal-reference
  pilots and, when integrated, square the average bias. nprobust's `kdrobust`
  accepts the Epanechnikov and uniform kernels only, so that `compat = true` with
  another kernel is an error.

Where no observation has positive weight in a window, the standard error of that
estimate is `NaN`: nprobust gives zero, and with it an interval of width zero
around an estimate of zero, as at points far from the data. The lower ends of the
intervals and bands are clamped at zero, since the density is nonnegative, which
can only raise their coverage. With `compat = true`, both are as in nprobust.

The kernel must be a symmetric second-order density kernel.
"""
function fit(::Type{PointwiseDensity}, x::AbstractVector; weights = nothing, kwargs...)
  return fit(PointwiseDensity, SmoothingData(x; weights); kwargs...)
end

function fit(::Type{PointwiseDensity}, d::SmoothingData; points = nothing,
             kernel::AbstractKernel = EpanechnikovKernel(), bandwidth = nothing, pilot = nothing,
             rho::Real = 1, minobs::Integer = 21, compat::Bool = false)
  _noclusters(d, "PointwiseDensity")
  d = _withouty(d)
  _check_density_kernel(kernel)
  isapprox(moment(kernel, 1), 0; atol = 1e-12) && isapprox(moment(kernel, 3), 0; atol = 1e-12) ||
    throw(ArgumentError("$kernel is not symmetric; robust bias-corrected densities need a symmetric kernel"))
  minobs >= 0 || throw(ArgumentError("minobs must be nonnegative, got $minobs"))
  compat && !(kernel isa Union{EpanechnikovKernel,UniformKernel}) && throw(ArgumentError(
    "nprobust's kdrobust accepts the Epanechnikov and uniform kernels only, so compat = true has no " *
    "counterpart for $kernel; use compat = false"))
  pts = points === nothing ? [_quantile(d, q) for q in range(0.1, 0.9; length = 30)] : _checkpoints(points)
  rule = bandwidth === nothing ? _default_density_rule(length(pts)) : bandwidth
  sel = _anyselection(_density_selection(rule, d, kernel, pts, minobs, compat))
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
  floored = _floor!(h, b, d, pts, minobs)
  c0, c2 = _secondderivative(kernel)
  κ = compat ? (c0 * moment(kernel, 4) + c2 * moment(kernel, 6)) / 24 : moment(kernel, 2) / 2
  r = _pointwise(j -> _densityfit(d, kernel, pts[j], h[j], b[j], c0, c2, κ, compat), eachindex(pts))
  return PointwiseDensity(d, kernel, sel, Int(minobs), compat, pts, h, b, [t.f for t in r],
    [t.se for t in r], [t.fbc for t in r], [t.sebc for t in r], [t.n for t in r], floored)
end

function _check_density_kernel(k)
  _isnonnegative(k) && isapprox(moment(k, 0), 1; rtol = 1e-10) ||
    throw(ArgumentError("$k is not a density kernel: it must be nonnegative and integrate to one"))
  return nothing
end

_default_density_rule(m) = m == 1 ? MSEOptimal() : IMSEOptimal()

_density_selection(h::Union{Real,AbstractVector{<:Real}}, d, k, pts, minobs, compat) =
  _regression_selection(h, d, k, 0, 0, pts, nothing, minobs)
_density_selection(rule::Union{MSEOptimal,IMSEOptimal,CoverageOptimal}, d, k, pts, minobs, compat) =
  selectbandwidth(rule, d, k; points = pts, minobs, compat)
_density_selection(rule::Union{RuleOfThumb,CrossValidation}, d, k, pts, minobs, compat) =
  throw(ArgumentError("$rule chooses bandwidths for regression; use a density rule"))
_density_selection(rule::BandwidthSelector, d, k, pts, minobs, compat) = selectbandwidth(rule, d, k)
_density_selection(h, d, k, pts, minobs, compat) =
  throw(ArgumentError("bandwidth must be a positive number, a vector or a BandwidthSelector, got $(repr(h))"))

# The observations where k(u) or k(ρu) can be nonzero, u = (x₀ - xᵢ)/h. The test
# computes the larger of the two arguments as the kernels get them, as nprobust does,
# so that rounding cannot leave out an observation that one of them weights at the
# edge of its support: a window of (xᵢ - x₀)/b put 146 observations of gridded data
# just outside, where ρu = -1 and the uniform kernel is 1/2.
function _densitywindow(d::SmoothingData, k, x0, h, ρ)
  xs, n = d.x, nobs(d)
  rad = maximum(abs, support(k))
  isfinite(rad) || return 1:n
  arg(i) = (t = (xs[i] - x0) / h; ρ < 1 ? ρ * t : t)
  return _firsttrue(i -> arg(i) >= -rad, n):_firsttrue(i -> arg(i) > rad, n)-1
end

# c₀, c₂ with ∫ k(u)(c₀ + c₂u²) u^j du = 2 δⱼ₂ for j = 0, 1, 2, 3: the kernel of the
# second derivative in a local quadratic fit with the symmetric kernel k
function _secondderivative(k)
  μ2, μ4 = moment(k, 2), moment(k, 4)
  c2 = 2 / (μ4 - μ2^2)
  return -μ2 * c2, c2
end

# The bias term ρ³ κ k(v) (c₀ + c₂ v²) of the robust density estimate is the product
# where that is finite, except that a Gaussian tail k(v) = c exp(-z²/2) that
# underflowed, which is at most max(c, 1) 2^-1075, can be a normal number when
# ρ³ κ |q| max(c, 1) ≥ 2^53, for an extreme ratio ρ = h / b or scale. The loops over
# a window sum the products, and only where a sum is not finite or _biascare finds
# such a tail possible, they sum the window again with _densitybias, which evaluates
# these terms in logarithms.
function _biascare(d, k, x0, h, ρ, r, c0, c2, κ)
  k isa GaussianFamily && !isempty(r) || return false
  # |q| is largest at the observation farthest from x0; a factor 2 covers rounding
  vmax = ρ * max(abs(x0 - d.x[first(r)]), abs(x0 - d.x[last(r)])) / h
  c = abs(_amplitude(k)) * INV_SQRT_2PI / first(_scale_order(k))
  return !(ρ^3 * κ * (abs(c0) + abs(c2) * vmax^2) * max(c, 1.0) < 0x1p52)
end

# the bias term with this care
@inline function _densitybias(k, v, ρ, c0, c2, κ)
  kv = k(v)
  q = c0 + c2 * v^2
  t = ρ^3 * κ * kv * q
  if k isa GaussianFamily && iszero(kv)
    c = abs(_amplitude(k)) * INV_SQRT_2PI / first(_scale_order(k))
    abs(ρ^3 * κ * q) * max(c, 1.0) < 0x1p53 && return zero(t)
  elseif isfinite(t)
    return t
  end
  return _densitybiaslog(k, v, ρ, c0, c2, κ, kv, q)
end

# the bias term without the product, which overflows, multiplies zero by infinity
# or loses an underflowed Gaussian tail; exactly zero contributions stay zero
@noinline function _densitybiaslog(k, v, ρ, c0, c2, κ, kv, q)
  iszero(kv) && !(k isa GaussianFamily) && return zero(kv)
  iszero(q) && return zero(q)
  logq = isfinite(q) ? log(abs(q)) : log(abs(c2)) + 2log(abs(v))
  logk = if k isa GaussianFamily
    σ = first(_scale_order(k))
    z = v / σ
    isinf(z^2) && return zero(kv)
    log(_amplitude(k) * INV_SQRT_2PI) - log(σ) - z^2 / 2
  else
    log(kv)
  end
  return copysign(exp(3log(ρ) + log(κ) + logk + logq), q)
end

# the estimates and their standard errors at one point, summed over the window of
# the larger bandwidth; with sᵣ = ∑ wᵢʳ kᵢ and s₂ = ∑ wᵢ² kᵢ², as for KDE. Unless
# compat, a standard error without an observation of nonzero weight, where the sum
# of squared weights vanishes, is NaN, and the count is that of the observations
# inside the larger window, and at its edges where the kernel is positive there.
function _densityfit(d::SmoothingData, k, x0, h, b, c0, c2, κ, compat::Bool)
  xs, n = d.x, nobs(d)
  ρ = h / b
  hb = max(h, b)
  r = _densitywindow(d, k, x0, h, ρ)
  rad = maximum(abs, support(k))
  s0 = s1 = s2 = t0 = t1 = t2 = 0.0
  for i in r
    u = (x0 - xs[i]) / h
    kv = k(u)
    v = ρ * u
    m = kv - ρ^3 * κ * k(v) * (c0 + c2 * v^2)
    w = _weight(d.w, i)
    s0 += w * kv
    s1 += w^2 * kv
    s2 += w^2 * kv^2
    t0 += w * m
    t1 += w^2 * m
    t2 += w^2 * m^2
  end
  if !compat && (!isfinite(t0 + t1 + t2) || _biascare(d, k, x0, h, ρ, r, c0, c2, κ))
    s0, s1, s2, t0, t1, t2 = _densitysumscareful(d, k, x0, h, ρ, r, c0, c2, κ)
  end
  W = d.sumw
  se(s0, s1, s2) = !compat && s2 == 0 ? NaN : _densityse(s0, s1, s2, h, W, d.neff)
  e = !compat && k(1.0) == 0   # the edges of the window get no weight
  nwin = isfinite(rad) ? _firsttrue(i -> e ? xs[i] - x0 >= hb : xs[i] - x0 > hb, n) -
                         _firsttrue(i -> e ? xs[i] - x0 > -hb : xs[i] - x0 >= -hb, n) : n
  return (f = s0 / (h * W), se = se(s0, s1, s2), fbc = t0 / (h * W), sebc = se(t0, t1, t2), n = nwin)
end

# the sums of _densityfit over the window r with the bias terms from _densitybias;
# kept out of line, so that the usual loop compiles as it would without it
@noinline function _densitysumscareful(d, k, x0, h, ρ, r, c0, c2, κ)
  xs = d.x
  s0 = s1 = s2 = t0 = t1 = t2 = 0.0
  for i in r
    u = (x0 - xs[i]) / h
    kv = k(u)
    v = ρ * u
    m = kv - _densitybias(k, v, ρ, c0, c2, κ)
    w = _weight(d.w, i)
    s0 += w * kv
    s1 += w^2 * kv
    s2 += w^2 * kv^2
    t0 += w * m
    t1 += w^2 * m
    t2 += w^2 * m^2
  end
  return s0, s1, s2, t0, t1, t2
end


## Interface, shared with local polynomial regression

const _PointwiseFit = Union{LocalPolynomialRegression,PointwiseDensity}

nobs(m::PointwiseDensity) = nobs(m.data)
bandwidth(m::PointwiseDensity) = m.h
StatsAPI.coef(m::PointwiseDensity; method::InferenceMethod = Conventional()) =
  copy(_usesbc(_densitymethod(method)) ? m.estimate_bc : m.estimate)

stderror(m::PointwiseDensity; method::InferenceMethod = Conventional()) =
  copy(_usesbc(_densitymethod(method)) ? m.se_rbc : m.se)

# the methods of density fits; honest intervals, and their standard errors, are
# implemented for regression only (see honest.jl)
_densitymethod(method::InferenceMethod) = method

function StatsAPI.confint(m::PointwiseDensity; level::Real = 0.95, method::InferenceMethod = RobustBiasCorrected())
  return _confint(m, level, method)
end

function _confint(m, level, method)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  est, se = _usesbc(method) ? (m.estimate_bc, m.se_rbc) : (m.estimate, m.se)
  return hcat(_lowerend(m, est .- z .* se), est .+ z .* se)
end

# The lower ends of intervals for a density, which is nonnegative, clamped at zero,
# which can only raise their coverage; under compat they are as in R
_lowerend(m, lower) = m isa LocalPolynomialRegression || !_nonnegative(m) ? lower : max.(lower, 0.0)
_nonnegative(m::PointwiseDensity) = !m.compat

# whether a method is built around the bias-corrected estimate
_usesbc(::Conventional) = false
_usesbc(::RobustBiasCorrected) = true

function StatsAPI.coeftable(m::PointwiseDensity; level::Real = 0.95)
  c = confint(m; level, method = Conventional())
  r = confint(m; level, method = RobustBiasCorrected())
  return (point = copy(m.points), h = copy(m.h), b = copy(m.b), nobs = copy(m.nwindow),
          floored = copy(m.floored), estimate = copy(m.estimate), se = copy(m.se), lower = c[:, 1],
          upper = c[:, 2], estimate_bc = copy(m.estimate_bc), se_rbc = copy(m.se_rbc),
          lower_rbc = r[:, 1], upper_rbc = r[:, 2])
end

function Base.show(io::IO, m::PointwiseDensity)
  print(io, "PointwiseDensity(", m.kernel, ", ", length(m.points), " points, ", nobs(m), " observations)")
end

function Base.show(io::IO, ::MIME"text/plain", m::PointwiseDensity)
  println(io, "Kernel density estimates with robust bias-corrected inference",
          m.compat ? " (nprobust compatibility)" : "")
  println(io, "  kernel: ", m.kernel, ", ", nobs(m), " observations")
  rule = m.selection.rule
  rule === nothing || println(io, "  bandwidth: ", rule, !(rule isa Union{MSEOptimal,IMSEOptimal,CoverageOptimal}) ? "" :
                              m.compat ? ", nprobust's pilots" : ", pilots anchored on Sheather–Jones")
  any(m.floored) && println(io, "  the floor of ", m.minobs, " observations raised the bandwidth at ",
                            count(m.floored), " of ", length(m.points), " points")
  _showtable(io, m)
end

function _showtable(io, m)
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

"""
    confint(f::KDE, x; level = 0.95, method = RobustBiasCorrected())

Pointwise confidence intervals for the density at the points `x`, with the
bandwidth of `f` and the same pilot bandwidth, as rows `[lower upper]` of a matrix.
The robust bias-corrected intervals (Calonico, Cattaneo and Farrell 2018) remain
valid at bandwidths that minimize the mean squared error;
`method = Conventional()` gives `f̂(x) ± z stderror(f, x)`, which covers only when
the bias is negligible. The lower ends are clamped at zero, and where no observation
has positive weight the interval is `NaN`. See [`PointwiseDensity`](@ref) for
pointwise bandwidths.
"""
function StatsAPI.confint(f::KDE, x::AbstractVector{<:Real}; level::Real = 0.95,
                          method::InferenceMethod = RobustBiasCorrected())
  m = fit(PointwiseDensity, f.data; points = x, kernel = f.kernel, bandwidth = f.h, minobs = 0)
  return confint(m; level, method)
end

# at one point, the interval as a tuple (lower, upper)
function StatsAPI.confint(f::KDE, x::Real; kwargs...)
  c = confint(f, [x]; kwargs...)
  return (c[1, 1], c[1, 2])
end


## Plug-in bandwidths for densities

# For data without responses, MSEOptimal and IMSEOptimal give
#   h(x) = (R(k) f(x) / (4 n (B(x)² + regularization · ℛ(x))))^(1/5),   B(x) = μ₂(k)/2 f''(x),
# with the pilot f'' estimated with the kernel D at the bandwidth b. By default b is
# the normal-reference ratio of the bandwidths for f'' and for f applied to the
# Sheather–Jones bandwidth, so that it adapts to the roughness of the density, and
# f is estimated at the Sheather–Jones bandwidth; ℛ = 3 (μ₂/2)² Var(f̂'') keeps h
# finite where f'' vanishes. With compat = true, the rule-of-thumb pilots of
# nprobust's kdbwselect are used instead.
# (imse, integrated or pointwise, comes as a Val, so that each selection infers)
function _densityplugin(rule, d, k, points, minobs, compat, ::Val{imse}) where imse
  rule.interior && throw(ArgumentError("the plug-in bandwidths for densities are for interior points " *
    "already, and have no interior option, got $rule"))
  compat && return _kdbwselect(rule, d, k, points, minobs, Val(imse))
  xs, N = d.x, effective_nobs(d)
  c0, c2 = _secondderivative(k)
  μ2, R = moment(k, 2), roughness(k)
  k2 = _square(k)
  RD = c0^2 * moment(k2, 0) + 2c0 * c2 * moment(k2, 2) + c2^2 * moment(k2, 4)   # ∫ D²
  κ4 = (c0 * moment(k, 4) + c2 * moment(k, 6)) / 24                            # ∫ u⁴ D / 4!
  # bandwidths for f'' (kernel D) and f (kernel k) that minimize the asymptotic mean
  # integrated squared error for the standard normal density, ψ₈ = 8!/(2⁹ 4! √π)
  ψ8 = factorial(8) / (2^9 * factorial(4) * sqrt(π))
  bnr = (5RD / (4 * κ4^2 * ψ8 * N))^(1 / 9)
  hnr = (8 * sqrt(π) * R / (3 * μ2^2 * N))^(1 / 5)
  hsj = bandwidth(SheatherJones(), d, k)
  b = hsj * bnr / hnr
  regul = rule.regularization
  # the pilot estimates of f'' and f at the points c from kernel sums over the
  # windows, exact term by term, instead of loops over all observations
  function constants(c)
    Sb, Sh = _pilotsums(d, k, b, c, 2), _pilotsums(d, k, hsj, c, 0)
    return map(eachindex(c)) do j
      f, fpp = Sh[1, j] / (d.sumw * hsj), (c0 * Sb[1, j] + c2 * Sb[3, j]) / (d.sumw * b^3)
      (V = R * f, B2 = (μ2 / 2 * fpp)^2, ℛ = 3 * (μ2 / 2)^2 * RD * max(f, 0.0) / (N * b^5))
    end
  end
  hopt(V, B2, Rr) = (V / (4N * (B2 + regul * Rr)))^(1 / 5)
  if imse
    grid = _densitygrid(rule, d, points)
    cs = constants(grid)
    h = hopt(sum(c.V for c in cs), sum(c.B2 for c in cs), sum(c.ℛ for c in cs))
    # where the pilot estimates vanish on the whole grid, the Sheather–Jones bandwidth
    h > 0 && isfinite(h) || (h = hsj)
    return BandwidthSelection(h, rule, (b = b, grid = grid, sheatherjones = hsj))
  end
  pts = points === nothing ? [_quantile(d, q) for q in range(0.1, 0.9; length = 30)] : points
  h, bs = similar(pts), similar(pts)
  all_cs = constants(pts)
  for (j, c) in enumerate(pts)
    cs = all_cs[j]
    floor = minobs > 0 ? knn_distance(d, c, min(minobs, nobs(d))) : 0.0
    cap = max(abs(c - xs[1]), abs(c - xs[end]))
    h[j], bs[j] = _clampdensity(hopt(cs.V, cs.B2, cs.ℛ), floor, cap), max(b, floor)
  end
  return BandwidthSelection(h, rule, (b = bs, points = pts, sheatherjones = hsj))
end

# the grid of the integrated density rules: quantiles of order 0.1 to 0.9, 30 of them
# as in nprobust's kdbwselect, or the evaluation points with grid = :points
_densitygrid(rule, d, points) = rule.grid === :points ? _gridpoints(points) :
  [_quantile(d, q) for q in range(0.1, 0.9; length = something(rule.grid, 30))]

# ∑ᵢ wᵢ k(uᵢ) uᵢ^r for r ≤ kmax at the points c, uᵢ = (xᵢ - c) / h, summed over the
# windows for compactly supported kernels and by the exact choice of the engines
# otherwise
_pilotsums(d, k, h, c, kmax) =
  kernelsums(d, k, h, c; kmax, engine = all(isfinite, support(k)) ? WindowSums() : nothing)

# The bandwidth at a point between the floor and the cap. Where the pilot estimates
# vanish, as between two clusters, the rule gives NaN or zero, and the floor is
# used instead, or the cap without a floor.
function _clampdensity(h, floor, cap)
  h > 0 && isfinite(h) || return floor > 0 ? floor : cap
  return max(min(h, cap), floor)
end

# nprobust's kdbwselect: rule-of-thumb pilots for f and f''
function _kdbwselect(rule, d, k, points, minobs, ::Val{imse}) where imse
  k isa Union{EpanechnikovKernel,UniformKernel} || throw(ArgumentError(
    "nprobust's plug-in density bandwidths (compat = true) are defined for the Epanechnikov and " *
    "uniform kernels; pass a bandwidth or use compat = false for $k"))
  xs, N = d.x, effective_nobs(d)
  sd = _std(d)
  # nprobust's rule-of-thumb constants for the density and its second derivative
  Ch, Cb = k isa EpanechnikovKernel ? (2.34, 3.49) : (1.06, 1.0)
  hrot, brot = sd * Ch * N^(-1 / 5), sd * Cb * N^(-1 / 13)
  c0, c2 = _secondderivative(k)
  μ2, R = moment(k, 2), roughness(k)
  # the squared-bias and variance constants at the points c, from pilot estimates of
  # f'' and f
  function constants(c)
    Sb, Sh = _pilotsums(d, k, brot, c, 2), _pilotsums(d, k, hrot, c, 0)
    return [((c0 * Sb[1, j] + c2 * Sb[3, j]) / (d.sumw * brot^3) * μ2 / 2, Sh[1, j] / (d.sumw * hrot) * R)
            for j in eachindex(c)]
  end
  hopt(V, B2) = (V / (4 * N * B2))^(1 / 5)
  if imse
    grid = _densitygrid(rule, d, points)
    cs = constants(grid)
    B = [c[1] for c in cs]
    V = sum(c[2] for c in cs) / length(cs)
    # nprobust squares the average bias, which can cancel where f'' changes sign
    B2 = (sum(B) / length(B))^2
    return BandwidthSelection(hopt(V, B2), rule, (b = brot, grid = grid, sheatherjones = NaN))
  end
  pts = points === nothing ? [_quantile(d, q) for q in range(0.1, 0.9; length = 30)] : points
  h, b = similar(pts), similar(pts)
  all_cs = constants(pts)
  for (j, c) in enumerate(pts)
    B, V = all_cs[j]
    floor = minobs > 0 ? knn_distance(d, c, min(minobs, nobs(d))) : 0.0
    # nprobust has no cap, which only replaces a bandwidth that the rule leaves undefined
    h[j] = _clampdensity(hopt(V, B^2), floor, Inf)
    isfinite(h[j]) || (h[j] = max(abs(c - xs[1]), abs(c - xs[end])))
    b[j] = max(brot, floor)
  end
  return BandwidthSelection(h, rule, (b = b, points = pts, sheatherjones = NaN))
end
