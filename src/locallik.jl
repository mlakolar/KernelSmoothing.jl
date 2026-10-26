## Local likelihood

# The families of the local likelihood fits and the links they take, the first being
# the default
const _LLLINKS = (logit = (:logit,), poisson = (:log,), gaussian = (:identity,), gamma = (:log, :inverse),
                  inverse_gaussian = (:log, :inverse_squared), negative_binomial = (:log,))
const _LLCANONICAL = (logit = :logit, poisson = :log, gaussian = :identity, gamma = :inverse,
                      inverse_gaussian = :inverse_squared)

# a family with its link, the overdispersion θ of the negative binomial, and whether
# μ'(η) = V(μ), as for the logit, Poisson and Gaussian families with their default
# links (the inverse links of the gamma and inverse Gaussian families are canonical
# only up to a constant factor)
struct _LLFamily
  family::Symbol
  link::Symbol
  theta::Float64
  canonical::Bool
end
_llcanonical(family, link) = family in (:logit, :poisson, :gaussian)
Base.broadcastable(f::_LLFamily) = Ref(f)

function _llfamily(family::Symbol, link, theta)
  haskey(_LLLINKS, family) ||
    throw(ArgumentError("family must be one of $(join(repr.(keys(_LLLINKS)), ", ")), got $(repr(family))"))
  links = _LLLINKS[family]
  l = link === nothing ? links[1] : link === :canonical && haskey(_LLCANONICAL, family) ? _LLCANONICAL[family] : link
  l in links || throw(ArgumentError("the $family family takes the link$(length(links) > 1 ? "s" : "") " *
                                    "$(join(repr.(links), " or ")), got $(repr(link))"))
  if family === :negative_binomial
    theta isa Real && isfinite(theta) && theta > 0 || throw(ArgumentError(
      "the negative binomial family needs theta, a positive number with variance μ + μ²/θ, got $(repr(theta))"))
  else
    theta === nothing || throw(ArgumentError("theta is the overdispersion of the negative binomial family only"))
  end
  return _LLFamily(family, l, theta === nothing ? NaN : Float64(theta), _llcanonical(family, l))
end

function _llcheckresponses(f::_LLFamily, y)
  fam = f.family
  if fam === :logit
    all(v -> v == 0 || v == 1, y) || throw(ArgumentError("the logit family needs responses 0 or 1"))
  elseif fam in (:poisson, :negative_binomial)
    all(v -> v >= 0 && isinteger(v), y) ||
      throw(ArgumentError("the $(fam === :poisson ? "Poisson" : "negative binomial") family needs nonnegative integer responses"))
  elseif fam in (:gamma, :inverse_gaussian)
    all(>(0), y) || throw(ArgumentError("the $fam family needs positive responses"))
  end
  return
end

# the inverse link μ(η), its derivative dμ/dη, the link η(μ) and the variance function
function _lllinkinv(f::_LLFamily, η)
  l = f.link
  l === :logit ? 1 / (1 + exp(-η)) : l === :log ? exp(η) : l === :identity ? η : l === :inverse ? 1 / η : 1 / sqrt(η)
end

function _llmueta(f::_LLFamily, η)
  l = f.link
  if l === :logit
    e = exp(-abs(η))
    return e / (1 + e)^2
  end
  l === :log ? exp(η) : l === :identity ? 1.0 : l === :inverse ? -1 / η^2 : -1 / (2 * η * sqrt(η))
end

function _lllink(f::_LLFamily, μ)
  l = f.link
  l === :logit ? log(μ / (1 - μ)) : l === :log ? log(μ) : l === :identity ? μ : l === :inverse ? 1 / μ : 1 / μ^2
end

function _llvar(f::_LLFamily, μ)
  fam = f.family
  fam === :logit ? μ * (1 - μ) : fam === :poisson ? μ : fam === :gaussian ? 1.0 : fam === :gamma ? μ^2 :
    fam === :inverse_gaussian ? μ^3 : μ + μ^2 / f.theta
end

# the mean at η, NaN where an inverse link has η ≤ 0, which gives no valid mean
_llmean(f::_LLFamily, η) = isnan(η) || (f.link in (:inverse, :inverse_squared) && η <= 0) ? NaN : _lllinkinv(f, η)

# whether η gives a mean in the range of the family, and whether a mean starts a fit
_llvalid(f::_LLFamily, η) = isfinite(η) && (f.link in (:inverse, :inverse_squared) ? η > 0 : true) &&
  (μ = _lllinkinv(f, η); isfinite(μ) && (f.family in (:logit, :gaussian) || μ > 0))
_llstart(f::_LLFamily, ȳ) = f.family === :logit ? 0 < ȳ < 1 : f.family === :gaussian ? isfinite(ȳ) : ȳ > 0

# The log likelihood of a response y at the linear predictor η, up to terms that do
# not depend on η and to the factor 1/φ of the dispersion, which change neither the
# fits nor the bandwidths: -Inf where η gives no valid mean
function _llloglik(f::_LLFamily, y, η)
  fam = f.family
  isfinite(η) || return -Inf
  if fam === :logit
    return η > 0 ? y * η - η - log1p(exp(-η)) : y * η - log1p(exp(η))
  elseif fam === :poisson
    return y * η - exp(η)
  elseif fam === :gaussian
    return -(y - η)^2 / 2
  end
  _llvalid(f, η) || return -Inf
  μ = _lllinkinv(f, η)
  fam === :gamma && return -y / μ - log(μ)
  fam === :inverse_gaussian && return -(y - μ)^2 / (2 * y * μ^2)
  return y * log(μ) - (y + f.theta) * log(μ + f.theta)   # negative binomial
end

_llsum(f, y, kw, η) = sum(kw[i] * _llloglik(f, y[i], η[i]) for i in eachindex(y) if kw[i] > 0; init = 0.0)

# The factors of the score and the information of observation i at η: the score is
# ∑ᵢ kwᵢ aᵢ (yᵢ - μᵢ) zᵢ and the information ∑ᵢ kwᵢ ωᵢ zᵢ zᵢ', with a = μ'/V and
# ω = μ'²/V, which are 1 and V when μ' = V
function _llfactors(f::_LLFamily, η)
  μ = _lllinkinv(f, η)
  v = _llvar(f, μ)
  f.canonical && return μ, 1.0, v
  me = _llmueta(f, η)
  return μ, me / v, me^2 / v
end

# The maximum of the local likelihood ∑ᵢ kwᵢ ℓ(yᵢ; r(uᵢ)'β), r(u) = (1, u, …, u^p), by
# Fisher scoring from β; nothing where it has no maximum, as where all responses of a
# logit fit are equal. Where μ' = V the steps are Newton's, taken in full as long as
# they converge, which from the local constant fit they do; otherwise, and when they
# fail, a step that would lower the likelihood is halved.
function _llirls(u, y, kw, f, β₀; halving::Bool = !f.canonical, maxiter::Integer = 200)
  P = length(β₀)
  # (the comprehension captures β₀, which is never reassigned: a captured variable
  # that is reassigned would be boxed)
  η = [evalpoly(t, β₀) for t in u]
  β = copy(β₀)
  ll = halving ? _llsum(f, y, kw, η) : 0.0
  isfinite(ll) || return nothing
  # the tolerance of the halving, relative to the scale of the likelihood and the weights
  tol = halving ? 1e-12 * (abs(ll) + sum(kw)) : 0.0
  S = zeros(2P - 1)
  g = zeros(P)
  δ = zeros(P)
  for _ in 1:maxiter
    _llscore!(S, g, u, y, kw, f, η)
    A = _hankel(S, P)
    C = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(copy(A)); check = false)
    LinearAlgebra.issuccess(C) && minimum(j -> C.factors[j, j]^2 / A[j, j], 1:P) > 1e-12 || return nothing
    LinearAlgebra.ldiv!(δ, C, g)
    if halving
      t = 1.0
      accepted = false
      for _ in 1:40
        βn = β .+ t .* δ
        ηn = [evalpoly(x, βn) for x in u]
        lln = _llsum(f, y, kw, ηn)
        if lln >= ll - tol
          β, η, ll = βn, ηn, lln
          accepted = true
          break
        end
        t /= 2
      end
      accepted || return nothing
    else
      β .+= δ
      for i in eachindex(u)
        η[i] = evalpoly(u[i], β)
      end
    end
    all(isfinite, β) || return nothing
    maximum(abs, δ) <= 1e-12 * (1 + maximum(abs, β)) && return β, η
  end
  return nothing
end

# At η, the power sums S[k+1] = ∑ᵢ kwᵢ ωᵢ uᵢ^k, k = 0, …, 2p, whose Hankel matrix is the
# information ∑ᵢ kwᵢ ωᵢ r(uᵢ) r(uᵢ)', and the score g = ∑ᵢ kwᵢ aᵢ (yᵢ - μᵢ) r(uᵢ)
function _llscore!(S, g, u, y, kw, f, η)
  fill!(S, 0.0)
  fill!(g, 0.0)
  P = length(g)
  @inbounds for i in eachindex(y)
    kw[i] > 0 || continue
    μ, a, ω = _llfactors(f, η[i])
    r, w = kw[i] * a * (y[i] - μ), kw[i] * ω
    z = 1.0
    for k in eachindex(S)
      k <= P && (g[k] += r * z)
      S[k] += w * z
      z *= u[i]
    end
  end
  return S, g
end

_hankel(S, P) = [S[i + j - 1] for i in 1:P, j in 1:P]

# the information matrix at η
function _llinfo(u, y, kw, f, η, P)
  S, _ = _llscore!(zeros(2P - 1), zeros(P), u, y, kw, f, η)
  return _hankel(S, P)
end

"""
    LocalLikelihood <: RegressionModel

Local polynomial likelihood estimates of a regression function for responses from an
exponential family (Tibshirani and Hastie 1987; Fan, Heckman and Wand 1995), fitted
with [`fit`](@ref)`(LocalLikelihood, x, y; family)`. At a point `x₀`, a polynomial of
order `degree` in the linear predictor maximizes the kernel-weighted log likelihood

    ∑ᵢ wᵢ k((xᵢ - x₀)/h) ℓ(yᵢ; β₀ + β₁ (xᵢ - x₀) + ⋯ + βₚ (xᵢ - x₀)^p),

for binary responses (`family = :logit`), counts (`:poisson`, or
`:negative_binomial` with variance `μ + μ²/θ` for overdispersed counts), positive
responses (`:gamma` and `:inverse_gaussian`, with the log link by default and their
canonical links `:inverse` and `:inverse_squared`), or `:gaussian` responses, for
which the fit is local polynomial regression. `β̂₀` estimates the linear predictor at
`x₀` and its inverse link the mean. Its standard error is the sandwich one of the
local fit, which does not rely on the variance function of the family or on its
dispersion, and intervals for the mean transform those for the linear predictor, so
that they stay within its range.

The bias-corrected estimate subtracts the leading term of the bias of `β̂₀`,
estimated from the fit of order `p + 1` with the pilot bandwidth `b`, and its robust
standard error accounts for the variability of that estimate (Calonico, Cattaneo and
Farrell 2018), as for [`LocalPolynomialRegression`](@ref): both linearize the two
local fits around their maxima, with the residuals of the fit of order `p + 1` in
the robust variance. For `family = :gaussian` they are those of
`LocalPolynomialRegression` with `vce = HCVariance(0)` and `minobs = 0` (its floor on
the windows changes the bandwidths). Where only the fit of order `p + 1` has no
maximum, the bias-corrected estimate and the default intervals are `NaN`.

`coef(m; method)` and `stderror(m; method)` describe the linear predictor at the
points, conventional by default and with `method = RobustBiasCorrected()` bias
corrected; `predict(m)` gives the means, and `confint(m; level, scale, method)` the
intervals, robust bias-corrected by default, for the mean and with `scale = :linear`
for the linear predictor. Points where the local likelihood has no maximum, as where
all responses in the window are equal for `:logit`, are `NaN`.
"""
struct LocalLikelihood{K<:AbstractKernel,S<:BandwidthSelection} <: StatsAPI.RegressionModel
  family::Symbol
  link::Symbol
  theta::Float64
  kernel::K
  degree::Int
  selection::S
  nobs::Int
  points::Vector{Float64}
  h::Vector{Float64}
  b::Vector{Float64}
  eta::Vector{Float64}
  se::Vector{Float64}
  eta_bc::Vector{Float64}
  se_rbc::Vector{Float64}
end

"""
    fit(LocalLikelihood, x, y; family = :logit, link = nothing, theta = nothing, points = nothing,
        bandwidth = CrossValidation(), pilot = nothing, rho = 1, degree = 1,
        kernel = EpanechnikovKernel(), weights = nothing)

Local likelihood estimates of the regression of the responses `y` on `x` at `points`,
by default as for [`LocalPolynomialRegression`](@ref), for the `family` and the
`link` described in [`LocalLikelihood`](@ref) (`link = :canonical` takes the canonical
link); the negative binomial family needs its overdispersion `theta`.

`bandwidth` is a number, one per point, [`CrossValidation`](@ref), which maximizes the
leave-one-out log likelihood, with the fits without each observation approximated by
one Fisher scoring step from the fit with it, over bandwidths whose windows hold at
least 21 other observations, or [`RuleOfThumb`](@ref), the plug-in bandwidth of Fan,
Heckman and Wand (1995) from a global polynomial fit. The pilot bandwidth `b` of the
bias correction is `pilot`, a number or one per point, or by default `h / rho`.
"""
function fit(::Type{LocalLikelihood}, x::AbstractVector, y::AbstractVector; family::Symbol = :logit,
             link::Union{Nothing,Symbol} = nothing, theta = nothing, points = nothing,
             bandwidth = CrossValidation(), pilot = nothing, rho::Real = 1, degree::Integer = 1,
             kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing)
  f = _llfamily(family, link, theta)
  p = Int(degree)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  _llcheckresponses(f, y)
  d = SmoothingData(x, y; weights)
  pts = points === nothing ? _regressionpoints(d, false) : _checkpoints(points)
  sel = if bandwidth isa CrossValidation
    _llselect(bandwidth, d, kernel, p, f)
  elseif bandwidth isa RuleOfThumb
    _llrot(bandwidth, d, kernel, p, f)
  elseif bandwidth isa Union{Real,AbstractVector{<:Real}}
    BandwidthSelection(bandwidth isa Real ? _bandwidths(bandwidth, 1) : _bandwidths(bandwidth, length(pts)), nothing, NamedTuple())
  else
    throw(ArgumentError("bandwidth must be positive numbers, CrossValidation() or RuleOfThumb(), got $(repr(bandwidth))"))
  end
  h = _pointbandwidths(sel.h, length(pts))
  b = if pilot !== nothing
    _pointbandwidths(pilot, length(pts))
  else
    0 < rho < Inf || throw(ArgumentError("rho must be positive and finite, or pass the pilot bandwidth"))
    h ./ rho
  end
  fits = _pointwise(j -> _llpoint(d, kernel, pts[j], h[j], b[j], p, f), eachindex(pts))
  return LocalLikelihood(f.family, f.link, f.theta, kernel, p, _anyselection(sel), nobs(d), pts, h, b,
                         [r.eta for r in fits], [r.se for r in fits], [r.eta_bc for r in fits],
                         [r.se_rbc for r in fits])
end

# The local likelihood fit of order p at x0 with bandwidth h, from the local constant
# fit, the link of the weighted mean: the window r, the kernel weights, the scaled
# covariate, the design, the coefficients in the scaled covariate and the linear
# predictor; nothing where the local likelihood has no maximum
function _llsolve(d, k, x0, h, p, f)
  r = _poswindow(d, k, x0, h)
  isempty(r) && return nothing
  u = [(d.x[i] - x0) / h for i in r]
  kw = [_weight(d.w, i) * k(u[l]) for (l, i) in enumerate(r)]
  y = view(d.y, r)
  sw = swy = 0.0
  for l in eachindex(kw)
    sw += kw[l]
    swy += kw[l] * y[l]
  end
  ȳ = swy / sw
  _llstart(f, ȳ) || return nothing
  β = zeros(p + 1)
  β[1] = _lllink(f, ȳ)
  res = _llirls(u, y, kw, f, β)
  # Newton's method without halving failed: retry with it
  res === nothing && f.canonical && (res = _llirls(u, y, kw, f, β; halving = true))
  res === nothing && return nothing
  return (r = r, u = u, kw = kw, y = y, β = res[1], η = res[2])
end

# The estimate of the linear predictor at x0 with its sandwich standard error, and
# the bias-corrected estimate with its robust standard error. Linearizing the fit of
# order p, β̂ - β ≈ A⁻¹ ∑ᵢ kwᵢ aᵢ (yᵢ - μᵢ) r_p(uᵢ) with A its information, so that
# with g = A⁻¹ e₁ the estimate is ∑ ℓᵢ (yᵢ - μᵢ) with ℓᵢ = kwᵢ aᵢ g'r_p(uᵢ), and its
# leading bias is (h/b)^(p+1) s γ_(p+1) with s = g' ∑ᵢ kwᵢ ωᵢ uᵢ^(p+1) r_p(uᵢ) and γ
# the coefficients of the fit of order q = p + 1 in v = (x - x0)/b. With c = A_q⁻¹
# e_(p+2), the bias-corrected estimate has ℓᵢᵇᶜ = ℓᵢ - (h/b)^(p+1) s kwᵢᵇ aᵢᵇ
# c'r_q(vᵢ), and the variances are ∑ (ℓᵢ ε̂ᵢ)² and ∑ (ℓᵢᵇᶜ ε̂ᵢᵇ)², with the residuals
# of the fits of order p and q, as in LocalPolynomialRegression with HC0 residuals.
function _llpoint(d, k, x0, h, b, p, f)
  s = _llsolve(d, k, x0, h, p, f)
  s === nothing && return (eta = NaN, se = NaN, eta_bc = NaN, se_rbc = NaN)
  A = _llinfo(s.u, s.y, s.kw, f, s.η, p + 1)
  F = _llfactors.(f, s.η)
  μ, a, ω = first.(F), getindex.(F, 2), last.(F)
  g = A \ [j == 1 ? 1.0 : 0.0 for j in 1:p+1]
  ℓ = s.kw .* a .* evalpoly.(s.u, Ref(g))
  se = sqrt(sum(abs2, ℓ .* (s.y .- μ)))
  q = _llsolve(d, k, x0, b, p + 1, f)
  q === nothing && return (eta = s.β[1], se = se, eta_bc = NaN, se_rbc = NaN)
  Aq = _llinfo(q.u, q.y, q.kw, f, q.η, p + 2)
  Fq = _llfactors.(f, q.η)
  μq, aq = first.(Fq), getindex.(Fq, 2)
  c = Aq \ [j == p + 2 ? 1.0 : 0.0 for j in 1:p+2]
  sb = sum(s.kw[l] * ω[l] * s.u[l]^(p + 1) * evalpoly(s.u[l], g) for l in eachindex(s.u))
  ρ = (h / b)^(p + 1)
  # the robust variance over the larger window, which contains the smaller one
  R = min(first(s.r), first(q.r)):max(last(s.r), last(q.r))
  vb = 0.0
  for i in R
    t = d.x[i] - x0
    li = i in s.r ? ℓ[i - first(s.r) + 1] : 0.0
    lq = 0.0
    if i in q.r
      l = i - first(q.r) + 1
      lq = q.kw[l] * aq[l] * evalpoly(q.u[l], c)
      ε = q.y[l] - μq[l]
    else
      η = evalpoly(t / b, q.β)
      ε = _llvalid(f, η) ? d.y[i] - _lllinkinv(f, η) : NaN
    end
    vb += ((li - ρ * sb * lq) * ε)^2
  end
  return (eta = s.β[1], se = se, eta_bc = s.β[1] - ρ * sb * q.β[p + 2], se_rbc = sqrt(vb))
end

# The leave-one-out linear predictor at observation i: one Fisher scoring step
# without it from the fit at x[i] with it, whose design vector there is e₁
function _llloo(d, k, i, h, p, f)
  s = _llsolve(d, k, d.x[i], h, p, f)
  s === nothing && return NaN
  A = _llinfo(s.u, s.y, s.kw, f, s.η, p + 1)
  l = i - first(s.r) + 1
  μ, a, ω = _llfactors(f, s.η[l])
  a11 = (A \ [j == 1 ? 1.0 : 0.0 for j in 1:p+1])[1]
  hii = s.kw[l] * ω * a11
  return s.β[1] - a11 * s.kw[l] * a * (s.y[l] - μ) / (1 - hii)
end

# The bandwidth that maximizes the leave-one-out log likelihood over a grid spaced
# evenly on the log scale, from the distance at which every observation has 21 others
# and p + 2 distinct values in its window, to the range of x. In smaller windows the
# local likelihood is unstable and the one-step leave-one-out fits are poor: for
# Poisson counts at n = 500 the criterion had a spurious minimum at windows of a few
# observations.
function _llselect(rule, d, k, p, f)
  nunique(d) > p + 1 || throw(ArgumentError("cross-validation needs more than $(p + 1) distinct values of x"))
  grid = bandwidth_grid(d; minobs = 22, mindistinct = p + 2, gridsize = rule.gridsize)
  crit(h) = _llcriterion(d, k, h, p, f)
  h, details = _gridminimize(crit, grid; largest = false, rtol = 1e-3)
  return BandwidthSelection(h, rule, (grid = details.grid, criterion = details.criterion))
end

# minus the weighted leave-one-out log likelihood per unit of weight
function _llcriterion(d, k, h, p, f)
  parts = _pointwise(i -> begin
    η = _llloo(d, k, i, h, p, f)
    wi = _weight(d.w, i)
    (-wi * (isnan(η) ? -Inf : _llloglik(f, d.y[i], η)), wi)
  end, 1:nobs(d))
  s = sum(first, parts)
  return isfinite(s) ? s / sum(last, parts) : Inf
end

# The rule-of-thumb bandwidth for local likelihood (Fan, Heckman and Wand 1995): the
# minimizer of the asymptotic integrated mean squared error of the linear predictor,
#
#     h = C_{0,p}(k) (∫ σ²(x) g'(μ(x))² dx / (n ∫ η^(p+1)(x)² f(x) dx))^(1/(2p+3)),
#
# whose variance term is that of local polynomial regression with the variance
# σ²(x) g'(μ(x))² of the linearized responses, from a global fit of a polynomial of
# order p + 3 in the linear predictor: σ² = φ V(μ), with the dispersion φ the Pearson
# statistic for the families that have one. The integral over the range of x is by
# the trapezoid rule; for the Gaussian family this is the RuleOfThumb of regression.
function _llrot(rule, d, k, p, f)
  pp = iseven(p) ? p + 1 : p
  B, V = _equivalentkernel(k, pp, 0)
  C = (factorial(pp + 1)^2 * V / (2 * (pp + 1) * B^2))^(1 / (2pp + 3))
  R = d.x[end] - d.x[1]
  R > 0 || throw(ArgumentError("the data have no spread, so RuleOfThumb cannot choose a bandwidth"))
  x̄, sd = _mean(d), sqrt(_variance(d))
  z = (d.x .- x̄) ./ sd
  w = d.w === nothing ? ones(nobs(d)) : Vector{Float64}(d.w)
  ȳ = sum(w .* d.y) / sum(w)
  n = effective_nobs(d)
  W = d.sumw
  r = pp + 1
  zg = range(z[1], z[end]; length = 401)
  # the highest order from p + 3 down to p + 1 that the distinct values identify and
  # whose fit has a maximum (which a logit fit with separated responses has not) and a
  # valid mean over the range of x; with fewer distinct values than p + 2, the
  # derivative vanishes and the bandwidth is the range
  Pmax = min(pp + 3, nunique(d) - 1)
  Pmax < r && return BandwidthSelection(R, rule, (constant = C, variance = NaN, curvature = 0.0, n = n))
  fitted = nothing
  P = Pmax
  while P >= r
    fitted = _llirls(z, d.y, w, f, [j == 0 ? _lllink(f, ȳ) : 0.0 for j in 0:P]; halving = true)
    if fitted !== nothing
      αP = fitted[1]
      all(t -> _llvalid(f, evalpoly(t, αP)), zg) && break
    end
    fitted = nothing
    P -= 1
  end
  fitted === nothing && throw(ArgumentError("RuleOfThumb: no global polynomial fit of the family of order " *
    "$r to $Pmax has a maximum and a valid mean over the data, as for separated binary responses; use CrossValidation()"))
  α = fitted[1]
  φ = if f.family in (:gaussian, :gamma, :inverse_gaussian)
    μ = _lllinkinv.(f, fitted[2])
    sum(_weight(d.w, i) * (d.y[i] - μ[i])^2 / _llvar(f, μ[i]) for i in eachindex(μ)) / W * n / max(n - P - 1, 1)
  else
    1.0
  end
  dr = [sum(α[j + 1] * prod(j-r+1:j) * z[i]^(j - r) for j in r:P; init = 0.0) / sd^r for i in eachindex(z)]
  D = sum(_weight(d.w, i) * dr[i]^2 for i in eachindex(dr)) / W
  # V/μ'² = 1/μ' where μ' = V, which stays accurate where the logit's μ rounds to 1
  vals = map(zg) do zt
    ηt = evalpoly(zt, α)
    f.canonical ? φ / _llmueta(f, ηt) : φ * _llvar(f, _lllinkinv(f, ηt)) / _llmueta(f, ηt)^2
  end
  I = sd * step(zg) * (sum(vals) - (first(vals) + last(vals)) / 2)
  h = D > 0 ? min(C * (I / (n * D))^(1 / (2pp + 3)), R) : R
  return BandwidthSelection(h, rule, (constant = C, variance = I, curvature = D, n = n))
end

nobs(m::LocalLikelihood) = m.nobs
bandwidth(m::LocalLikelihood) = m.h
StatsAPI.coef(m::LocalLikelihood; method::InferenceMethod = Conventional()) =
  copy(_usesbc(method) ? m.eta_bc : m.eta)
stderror(m::LocalLikelihood; method::InferenceMethod = Conventional()) = copy(_usesbc(method) ? m.se_rbc : m.se)
StatsAPI.predict(m::LocalLikelihood) = _llmean.(_llfamily(m), m.eta)

_llfamily(m::LocalLikelihood) = _LLFamily(m.family, m.link, m.theta, _llcanonical(m.family, m.link))

"""
    confint(m::LocalLikelihood; level = 0.95, scale = :mean, method = RobustBiasCorrected())

Pointwise intervals for the mean, the inverse link of the intervals for the linear
predictor, or with `scale = :linear` for the linear predictor, as a matrix with the
lower bounds in the first column: robust bias-corrected by default, and with
`method = Conventional()` around the estimates, which cover only when the bias is
negligible.
"""
function StatsAPI.confint(m::LocalLikelihood; level::Real = 0.95, scale::Symbol = :mean,
                          method::InferenceMethod = RobustBiasCorrected())
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  scale in (:mean, :linear) || throw(ArgumentError("scale must be :mean or :linear, got $(repr(scale))"))
  method isa Union{Conventional,RobustBiasCorrected} ||
    throw(ArgumentError("local likelihood intervals are Conventional() or RobustBiasCorrected()"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  η, se = coef(m; method), stderror(m; method)
  lo, hi = η .- z .* se, η .+ z .* se
  if scale === :mean
    f = _llfamily(m)
    if f.link in (:inverse, :inverse_squared)
      # a decreasing link swaps the ends; an end η ≤ 0 is a mean without bound, and an
      # interval of η without positive values holds no valid mean
      μlo = [b > 0 ? _lllinkinv(f, b) : NaN for b in hi]
      μhi = [isnan(a) || isnan(b) || b <= 0 ? NaN : a > 0 ? _lllinkinv(f, a) : Inf for (a, b) in zip(lo, hi)]
      lo, hi = μlo, μhi
    else
      lo, hi = _lllinkinv.(f, lo), _lllinkinv.(f, hi)
    end
  end
  return hcat(lo, hi)
end

function Base.show(io::IO, m::LocalLikelihood)
  print(io, "LocalLikelihood(", m.family, m.link === _LLLINKS[m.family][1] ? "" : " ($(m.link) link)",
        m.family === :negative_binomial ? " θ = $(m.theta)" : "", ", degree ", m.degree, ", ", m.kernel, ", ",
        length(m.points), " points, ", nobs(m), " observations)")
end


## Local likelihood density estimation

"""
    LocalLikelihoodDensity

The local likelihood density estimate of Loader (1996) and Hjort and Jones (1996),
fitted with [`fit`](@ref)`(LocalLikelihoodDensity, x)`. At a point `x₀`, the logarithm
of the density is approximated by a polynomial `P` of order `degree` in `x - x₀`,
which maximizes the local log likelihood

    ∑ᵢ wᵢ k((xᵢ - x₀)/h) P(xᵢ - x₀) - W ∫ k((u - x₀)/h) exp(P(u - x₀)) du,

with `W` the total weight, and `exp(P(0))` estimates the density. For `degree = 0` it
is the kernel density estimate; higher orders remove the bias of the kernel estimate
near the ends of the data and in its peaks and valleys, and the estimate is positive.
`coef(m)` gives the estimates at the points, `stderror(m)` their standard errors, and
`confint(m; level)` intervals from those for the logarithm of the density, which stay
positive. The estimates need not integrate to one.
"""
struct LocalLikelihoodDensity{K<:AbstractKernel}
  kernel::K
  degree::Int
  nobs::Int
  points::Vector{Float64}
  h::Vector{Float64}
  logf::Vector{Float64}
  se::Vector{Float64}   # of the logarithm of the density
end

"""
    fit(LocalLikelihoodDensity, x; points = nothing, bandwidth = nothing, span = 0.7, degree = 2,
        kernel = EpanechnikovKernel(), weights = nothing)

The local likelihood density estimate of the sample `x` at `points`, by default as
for [`PointwiseDensity`](@ref), see [`LocalLikelihoodDensity`](@ref). `bandwidth` is a
number or one per point; by default the bandwidth at each point is the distance to
its nearest observations that make up the fraction `span` of the sample, the default
of R's locfit. The kernel must have a bounded support. The standard errors are the
sandwich ones of the local fit, with the variance of the score estimated from the
observations in the window.
"""
function fit(::Type{LocalLikelihoodDensity}, x::AbstractVector; points = nothing, bandwidth = nothing,
             span::Real = 0.7, degree::Integer = 2, kernel::AbstractKernel = EpanechnikovKernel(),
             weights = nothing)
  p = Int(degree)
  p >= 0 || throw(ArgumentError("degree must be nonnegative, got $degree"))
  lo, hi = support(kernel)
  isfinite(lo) && isfinite(hi) || throw(ArgumentError("local likelihood density estimation needs a kernel with a bounded support"))
  d = SmoothingData(x; weights)
  pts = points === nothing ? [_quantile(d, q) for q in range(0.1, 0.9; length = 30)] : _checkpoints(points)
  h = if bandwidth === nothing
    0 < span <= 1 || throw(ArgumentError("span must be in (0, 1], got $span"))
    kk = max(1, floor(Int, span * nobs(d)))
    [knn_distance(d, x0, kk) for x0 in pts]
  else
    _pointbandwidths(bandwidth, length(pts))
  end
  all(>(0), h) || throw(ArgumentError("the nearest-neighbour bandwidth is zero at a point; give a bandwidth"))
  # the Gauss–Legendre rule on the pieces of the kernel, with its values at the nodes
  breaks = applicable(pieces, kernel) ? collect(first(pieces(kernel))) : [lo, hi]
  xs, ws = _GL48
  nodes = Float64[]
  wts = Float64[]
  for c in 1:length(breaks)-1
    m, r = (breaks[c] + breaks[c + 1]) / 2, (breaks[c + 1] - breaks[c]) / 2
    for l in eachindex(xs)
      t = m + r * xs[l]
      push!(nodes, t)
      push!(wts, r * ws[l] * kernel(t))
    end
  end
  fits = _pointwise(j -> _lldensity(d, kernel, pts[j], h[j], p, nodes, wts), eachindex(pts))
  return LocalLikelihoodDensity(kernel, p, nobs(d), pts, h, [r.logf for r in fits], [r.se for r in fits])
end

const _GL48 = _gausslegendre(48)

# Newton's method for the local likelihood of the density at x0, in the scaled
# variable t = (u - x0)/h, where it is β's - W h ∫ k(t) exp(β'r(t)) dt with
# s = ∑ᵢ wᵢ k(tᵢ) r(tᵢ) and r(t) = (1, t, …, t^p), concave in β; and the sandwich
# standard error of β̂₀, with the variance of the score estimated by
# (∑ wᵢ²/W) (∑ᵢ wᵢ k(tᵢ)² r r' - s s'/W) for independent observations
function _lldensity(d, k, x0, h, p, nodes, wts)
  nan = (logf = NaN, se = NaN)
  r = _poswindow(d, k, x0, h)
  isempty(r) && return nan
  P = p + 1
  t = [(d.x[i] - x0) / h for i in r]
  kw = [_weight(d.w, i) * k(t[l]) for (l, i) in enumerate(r)]
  sum(kw) > 0 || return nan
  W = d.sumw
  s = [sum(kw[l] * t[l]^j for l in eachindex(t)) for j in 0:p]
  Rn = [nodes[q]^j for q in eachindex(nodes), j in 0:p]
  # (a function of its argument only: a closure that assigned a name of this function
  # would share it)
  L(b) = LinearAlgebra.dot(b, s) - W * h * sum(wts[j] * exp(LinearAlgebra.dot(view(Rn, j, :), b)) for j in eachindex(wts))
  β = zeros(P)
  β[1] = log(sum(kw) / (W * h * sum(wts)))
  ll = L(β)
  converged = false
  A = zeros(P, P)
  for _ in 1:100
    e = exp.(Rn * β)
    g = s .- W * h .* (Rn' * (wts .* e))
    A = W * h .* (Rn' * ((wts .* e) .* Rn))
    C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
    LinearAlgebra.issuccess(C) || return nan
    δ = C \ g
    step = 1.0
    accepted = false
    for _ in 1:40
      βn = β .+ step .* δ
      lln = L(βn)
      if isfinite(lln) && lln >= ll - 1e-12 * (1 + abs(ll))
        β, ll = βn, lln
        accepted = true
        break
      end
      step /= 2
    end
    accepted || return nan
    if maximum(abs, step .* δ) <= 1e-12 * (1 + maximum(abs, β))
      converged = true
      break
    end
  end
  converged || return nan
  e = exp.(Rn * β)
  A = W * h .* (Rn' * ((wts .* e) .* Rn))
  # ∑ᵢ wᵢ² (ψᵢ - ψ̄)(ψᵢ - ψ̄)' over all observations, with ψᵢ = k(tᵢ) r(tᵢ), zero outside
  # the window, and ψ̄ = s/W, as for the standard errors of KDE
  w2 = [_weight(d.w, i)^2 for i in r]
  ψ = [k(t[l]) * t[l]^j for l in eachindex(t), j in 0:p]
  m = s ./ W
  q = ψ' * w2
  B = ψ' * (w2 .* ψ) .- m * q' .- q * m' .+ (m * m') .* (W^2 / d.neff)
  g = A \ [j == 1 ? 1.0 : 0.0 for j in 1:P]
  return (logf = β[1], se = sqrt(max(LinearAlgebra.dot(g, B * g), 0.0)))
end

nobs(m::LocalLikelihoodDensity) = m.nobs
bandwidth(m::LocalLikelihoodDensity) = m.h
StatsAPI.coef(m::LocalLikelihoodDensity) = exp.(m.logf)
stderror(m::LocalLikelihoodDensity) = exp.(m.logf) .* m.se

"""
    confint(m::LocalLikelihoodDensity; level = 0.95)

Pointwise intervals for the density, `exp(log f̂ ± z se)` from the standard errors of
the logarithm of the estimates, as a matrix with the lower bounds in the first column.
"""
function StatsAPI.confint(m::LocalLikelihoodDensity; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  return hcat(exp.(m.logf .- z .* m.se), exp.(m.logf .+ z .* m.se))
end

function Base.show(io::IO, m::LocalLikelihoodDensity)
  print(io, "LocalLikelihoodDensity(degree ", m.degree, ", ", m.kernel, ", ", length(m.points), " points, ",
        nobs(m), " observations)")
end
