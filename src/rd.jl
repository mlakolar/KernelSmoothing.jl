## Regression discontinuity

"""
    RegressionDiscontinuity <: RegressionModel

A local polynomial estimate of the effect of a treatment assigned by whether a
running variable `x` reaches a cutoff `c`, with conventional and robust
bias-corrected inference (Calonico, Cattaneo and Titiunik 2014), as rdrobust;
fitted with [`fit`](@ref)`(RegressionDiscontinuity, x, y; cutoff, bandwidth)`.

On each side of the cutoff, local polynomials of order `p` with bandwidth `h`
estimate the regression function and its derivatives at `c`, and fits of order
`p + 1` with the pilot bandwidth `b` estimate their bias, as for
[`LocalPolynomialRegression`](@ref). The estimate is the jump at `c` in the
derivative of order `derivative`:

- in a sharp design, `ν! (β̂₊ − β̂₋)` for the coefficients `β̂₊` and `β̂₋` of
  `(x − c)^ν` on the right and on the left, the jump in the regression function
  for `derivative = 0` and in its slope (a kink design) for `derivative = 1`;
- in a fuzzy design, given the treatment received as `treatment`, the ratio of the
  jumps in the outcome and in the treatment;
- with `covariates`, after adjusting the outcome (and the treatment) for the
  covariates with coefficients common to both sides (Calonico, Cattaneo, Farrell
  and Titiunik 2019).

`coef(m)` and `stderror(m)` give the estimate and its standard error, with
`method = RobustBiasCorrected()` the bias-corrected estimate and its robust
standard error, [`confint`](@ref)`(m; level, method)` the interval (robust
bias-corrected by default), and `coeftable(m)` the conventional, bias-corrected
and robust results, as rdrobust reports them. For a fuzzy design, `m.firststage`
holds the jump in the treatment.
"""
struct RegressionDiscontinuity{K<:AbstractKernel,V<:VarianceEstimator,S<:BandwidthSelection} <:
       StatsAPI.RegressionModel
  cutoff::Float64
  kernel::K
  degree::Int
  derivative::Int
  vce::V
  fuzzy::Bool
  ncovariates::Int
  selection::S
  compat::Bool
  h::NTuple{2,Float64}           # the bandwidths on the left and on the right
  b::NTuple{2,Float64}           # and the pilot bandwidths
  nobs::NTuple{2,Int}            # observations on each side
  nh::NTuple{2,Int}              # with positive weight for h
  nb::NTuple{2,Int}              # and for b
  estimate::Float64
  estimate_bc::Float64
  se::Float64
  se_rbc::Float64
  sides::NTuple{2,Float64}       # the estimates at c from the left and from the right
  bias::NTuple{2,Float64}        # the estimated biases on the two sides
  firststage::NamedTuple         # the jump in the treatment, for a fuzzy design
  gamma::Matrix{Float64}         # the coefficients of the covariates
end

"""
    fit(RegressionDiscontinuity, x, y; cutoff = 0, bandwidth, pilot = nothing, rho = 1,
        derivative = 0, degree = derivative + 1, kernel = TriangularKernel(),
        treatment = nothing, covariates = nothing, vce = nothing, weights = nothing,
        cluster = nothing, compat = false)

Estimates the jump at `cutoff` in the regression of `y` on the running variable `x`,
or in its derivative of order `derivative`, from local polynomials of order `degree`
on the two sides, as rdrobust. Observations with `x < cutoff` form the left side,
the others the right.

- `bandwidth` is a number, or a pair of numbers for the left and the right side.
- `pilot` is the bandwidth `b` of the fits of order `degree + 1` that estimate the
  bias, a number or a pair; by default `b = h / rho`, so that `b = h`, as rdrobust.
- `treatment` gives the treatment received, for a fuzzy design.
- `covariates` is a vector or a matrix with one row per observation, for the
  covariate-adjusted estimate; collinear covariates are dropped.
- `vce` estimates the conditional variances: [`NNVariance`](@ref) (the default),
  [`HCVariance`](@ref), or for clustered data [`ClusterVariance`](@ref) (the default
  with clusters). As in rdrobust, the residuals of the bias correction are adjusted
  with the leverages of the fits of order `degree + 1`.
- `compat = true` reproduces rdrobust's treatment of leverages near one, its factor
  of one for CR3, and its standard errors for clusters with singular blocks.
"""
function fit(::Type{RegressionDiscontinuity}, x::AbstractVector, y::AbstractVector; cutoff::Real = 0,
             treatment = nothing, covariates = nothing, weights = nothing, cluster = nothing,
             bandwidth = nothing, pilot = nothing, rho::Real = 1, derivative::Integer = 0,
             degree::Integer = derivative + 1, kernel::AbstractKernel = TriangularKernel(),
             vce::Union{Nothing,VarianceEstimator} = nothing, compat::Bool = false)
  c = Float64(cutoff)
  isfinite(c) || throw(ArgumentError("the cutoff must be finite, got $cutoff"))
  p, ν = Int(degree), Int(derivative)
  _check_degree(p, ν)
  n = length(x)
  length(y) == n || throw(DimensionMismatch("y has $(length(y)) values for $n observations"))
  _check_values(x, "x")
  _check_values(y, "y")
  D = reshape(Float64.(y), :, 1)
  fuzzy = treatment !== nothing
  if fuzzy
    length(treatment) == n || throw(DimensionMismatch("treatment has $(length(treatment)) values for $n observations"))
    _check_values(treatment, "treatment")
    D = hcat(D, Float64.(treatment))
  end
  dZ = 0
  if covariates !== nothing
    Z = covariates isa AbstractVector ? reshape(covariates, :, 1) : covariates
    size(Z, 1) == n || throw(DimensionMismatch("covariates has $(size(Z, 1)) rows for $n observations"))
    _check_values(vec(Z), "covariates")
    Z = _independentcolumns(Float64.(Z))
    dZ = size(Z, 2)
    D = hcat(D, Z)
  end
  left = findall(<(c), x)
  right = findall(>=(c), x)
  sides = map((left, right)) do ix
    isempty(ix) && throw(ArgumentError("the cutoff must lie inside the range of x, with observations on both sides"))
    d = SmoothingData(view(x, ix), view(y, ix); weights = weights === nothing ? nothing : view(weights, ix),
                      cluster = cluster === nothing ? nothing : view(cluster, ix))
    (data = d, D = D[ix[d.perm], :])
  end
  vce = _variance(vce, sides[1].data)
  h = _rdbandwidths(bandwidth, "bandwidth")
  b = pilot === nothing ? (0 < rho < Inf ? h ./ rho :
      throw(ArgumentError("rho must be positive and finite, or pass the pilot bandwidth"))) :
      _rdbandwidths(pilot, "pilot")
  sel = _anyselection(BandwidthSelection([h...], nothing, NamedTuple()))
  return _rdfit(sides, c, kernel, p, ν, vce, fuzzy, dZ, sel, h, b, compat)
end

function _rdbandwidths(h, name)
  h === nothing && throw(ArgumentError("pass a bandwidth; the bandwidth selectors for regression discontinuity are not implemented yet"))
  hs = h isa Real ? (h, h) : length(h) == 2 ? (h[1], h[2]) :
       throw(ArgumentError("$name must be a number or a pair of numbers for the left and the right side, got $(repr(h))"))
  all(v -> v isa Real && 0 < v < Inf, hs) || throw(ArgumentError("$name must be positive and finite, got $(repr(h))"))
  return Float64.(hs)
end

# The columns of Z without those that are collinear with earlier ones, as rdrobust's
# covs_drop, which keeps the pivots of a QR decomposition with tolerance 1e-7
function _independentcolumns(Z)
  F = LinearAlgebra.qr(Z, LinearAlgebra.ColumnNorm())
  R = F.R
  r = count(j -> abs(R[j, j]) > 1e-7 * abs(R[1, 1]), 1:min(size(R)...))
  return r == size(Z, 2) ? Z : Z[:, sort(F.p[1:r])]
end

function _rdfit(sides, c, k, p, ν, vce, fuzzy, dZ, sel, h, b, compat)
  dT = Int(fuzzy)
  hb = h == b   # rdrobust's rule for clusters at b = h on both sides
  fits = map(j -> _rdsidefit(sides[j].data, sides[j].D, k, c, h[j], b[j], p, ν), (1, 2))
  # the estimates of the columns of D at c from each side, raw: ν! β_ν / h^ν
  τp = map(f -> f.τp, fits)
  τbc = map(f -> f.τbc, fits)
  cols = 1:size(sides[1].D, 2)
  gamma = zeros(dZ, 1 + dT)
  if dZ > 0
    # the coefficients of the covariates, common to both sides, from the fits with h
    Zc = (2 + dT):(1 + dT + dZ)
    ZWZ = fits[1].ZWD[Zc, Zc] + fits[2].ZWD[Zc, Zc]
    ZWY = fits[1].ZWD[Zc, 1:1+dT] + fits[2].ZWD[Zc, 1:1+dT]
    gamma = LinearAlgebra.pinv(ZWZ; rtol = 1e-15) * ZWY
  end
  sY = vcat(1.0, -gamma[:, 1])
  YZ = [1; (2 + dT):(1 + dT + dZ)]
  jump(t, s, idx) = s' * (t[2][idx] - t[1][idx])
  side(t, s, idx, j) = s' * t[j][idx]
  if !fuzzy
    est, est_bc = jump(τp, sY, YZ), jump(τbc, sY, YZ)
    sv = zeros(length(cols))
    sv[YZ] = sY
    lr = (side(τp, sY, YZ, 1), side(τp, sY, YZ, 2))
    bias = (lr[1] - side(τbc, sY, YZ, 1), lr[2] - side(τbc, sY, YZ, 2))
    first = NamedTuple()
  else
    sT = vcat(1.0, -gamma[:, 2])
    TZ = [2; (2 + dT):(1 + dT + dZ)]
    τY, τT = jump(τp, sY, YZ), jump(τp, sT, TZ)
    τYbc, τTbc = jump(τbc, sY, YZ), jump(τbc, sT, TZ)
    est = τY / τT
    s = (1 / τT, -τY / τT^2)
    est_bc = est - (s[1] * (τY - τYbc) + s[2] * (τT - τTbc))
    sidebias(j) = s[1] * (side(τp, sY, YZ, j) - side(τbc, sY, YZ, j)) +
                  s[2] * (side(τp, sT, TZ, j) - side(τbc, sT, TZ, j))
    lr = (side(τp, sY, YZ, 1), side(τp, sY, YZ, 2))
    bias = (sidebias(1), sidebias(2))
    # the combination of the residuals of y, the treatment and the covariates whose
    # variance is that of the estimate, by the delta method
    sv = zeros(length(cols))
    sv[1], sv[2] = s
    for l in 1:dZ
      sv[2 + l] = -s[1] * gamma[l, 1] - s[2] * gamma[l, 2]
    end
    svT = zeros(length(cols))
    svT[2] = 1.0
    svT[3:end] .= -gamma[:, 2]
    vT = map(j -> _rdvariance(sides[j].data, sides[j].D, svT, k, c, h[j], b[j], p, ν, vce, compat, hb, fits[j]), (1, 2))
    first = (estimate = τT, estimate_bc = τTbc, se = sqrt(vT[1].vh + vT[2].vh), se_rbc = sqrt(vT[1].vb + vT[2].vb))
  end
  v = map(j -> _rdvariance(sides[j].data, sides[j].D, sv, k, c, h[j], b[j], p, ν, vce, compat, hb, fits[j]), (1, 2))
  return RegressionDiscontinuity(c, k, p, ν, vce, fuzzy, dZ, sel, compat, h, b,
    (nobs(sides[1].data), nobs(sides[2].data)), (fits[1].nh, fits[2].nh), (fits[1].nb, fits[2].nb),
    est, est_bc, sqrt(v[1].vh + v[2].vh), sqrt(v[1].vb + v[2].vb), lr, bias, first, gamma)
end

# The fits of order p with bandwidth h and of order q = p + 1 with bandwidth b at the
# cutoff c on one side, for each column of D, in the scaled variables u = (x - c)/h
# and v = (x - c)/b, as _localsetup: the raw estimates ν! β_ν / h^ν of the columns,
# their bias-corrected counterparts, and for the covariates the cross-products
# Z'W(D - R_p Γ_p⁻¹ U) with the kernel weights k(u)/h of rdrobust, whose 1/h matters
# when the two sides pool them with different bandwidths.
function _rdsidefit(d, D, k, c, h, b, p, ν)
  q = p + 1
  m = size(D, 2)
  r = _poswindow(d, k, c, max(h, b))
  a, e = first(r), last(r)
  sh, sb = zeros(2p + 2), zeros(2q + 1)
  th, tb = zeros(p + 1, m), zeros(q + 1, m)
  DD = zeros(m, m)
  nh = nb = 0
  for i in a:e
    t = d.x[i] - c
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    nh += kh > 0
    nb += kb > 0
    _powersums!(sh, kh, u)
    _powersums!(sb, kb, v)
    for l in 1:m
      Dl = D[i, l]
      _powersums!(view(th, :, l), kh * Dl, u)
      _powersums!(view(tb, :, l), kb * Dl, v)
      for l2 in l:m
        DD[l, l2] += kh * Dl * D[i, l2]
      end
    end
  end
  nuh = nunique(d, _poswindow(d, k, c, h))
  nub = nunique(d, _poswindow(d, k, c, b))
  (nuh >= p + 1 && nub >= q + 1) || throw(ArgumentError("a side of the cutoff has too few distinct " *
    "values of x with positive weight to fit the polynomials: $nuh for h (needs $(p + 1)) and $nub for b (needs $(q + 1))"))
  Cp, okp = _gramfactor([sh[i + j + 1] for i in 0:p, j in 0:p], nuh)
  Cq, okq = _gramfactor([sb[i + j + 1] for i in 0:q, j in 0:q], nub)
  okp && okq || throw(ArgumentError("the local polynomial fits at the cutoff are not identified"))
  βp, βq = Cp \ th, Cq \ tb
  Λ = sh[p+2:2p+2]
  ρ = (h / b)^(p + 1)
  βbc = βp .- ρ .* (Cp \ Λ) .* βq[q + 1:q + 1, :]
  scale = factorial(ν) / h^ν
  for l in 1:m, l2 in 1:l-1
    DD[l, l2] = DD[l2, l]
  end
  ZWD = (DD .- th' * βp) ./ h   # D'W D - U'Γ_p⁻¹U
  return (a = a, e = e, Cp = Cp, Cq = Cq, βp = βp, βq = βq, ρ = ρ, s = (Cp \ [j == ν ? 1.0 : 0.0 for j in 0:p])' * Λ,
          τp = scale .* βp[ν + 1, :], τbc = scale .* βbc[ν + 1, :], ZWD = ZWD, nh = nh, nb = nb)
end

# adds kw u^j to s[j+1]
function _powersums!(s, kw, u)
  kw == 0 && return
  v = kw
  for j in eachindex(s)
    s[j] += v
    v *= u
  end
  return
end

# The variances vh and vb of the raw estimate of order ν at c from one side and of
# its bias-corrected counterpart, for the response ys = D sv, whose fits are the same
# combination of the fits f of the columns of D. The residuals of the bias correction
# are adjusted with the leverages of the fit of order q, as in rdrobust; under
# compat, with its floors and its rules for clusters, which depend on whether b = h
# on both sides (hb).
function _rdvariance(d, D, sv, k, c, h, b, p, ν, vce, compat, hb, f)
  q = p + 1
  a, e, Cp, Cq, ρ, s = f.a, f.e, f.Cp, f.Cq, f.ρ, f.s
  m = e - a + 1
  ys = D * sv
  βp, βq = f.βp * sv, f.βq * sv
  g = Cp \ [j == ν ? 1.0 : 0.0 for j in 0:p]
  cq = Cq \ [j == q ? 1.0 : 0.0 for j in 0:q]
  dy = _withy(d, ys)
  ℓ, ℓbc, εh, εb = (Vector{Float64}(undef, m) for _ in 1:4)
  levp = vce isa HCVariance && vce.type >= 2 ? _leveragepoly(Cp, p) : nothing
  levq = vce isa HCVariance && vce.type >= 2 ? _leveragepoly(Cq, q) : nothing
  nn = vce isa NNVariance ? _nnresiduals(dy, a, e, vce.matches, _nntol(false)) : nothing
  adjust = vce isa ClusterVariance && vce.type >= 2
  Yp = adjust ? Matrix{Float64}(undef, p + 1, m) : nothing
  KYp = adjust ? Matrix{Float64}(undef, p + 1, m) : nothing
  Yq = adjust ? Matrix{Float64}(undef, q + 1, m) : nothing
  KYq = adjust ? Matrix{Float64}(undef, q + 1, m) : nothing
  for i in a:e
    j = i - a + 1
    t = d.x[i] - c
    u, v = t / h, t / b
    wi = _weight(d.w, i)
    kh, kb = wi * k(u), wi * k(v)
    ℓ[j] = kh * evalpoly(u, g)
    ℓbc[j] = ℓ[j] - ρ * s * kb * evalpoly(v, cq)
    if vce isa NNVariance
      εh[j] = εb[j] = nn[j]
    else
      eh, eb = ys[i] - evalpoly(u, βp), ys[i] - evalpoly(v, βq)
      if vce isa HCVariance
        hp = levp === nothing ? 0.0 : kh * evalpoly(u, levp)
        hq = levq === nothing ? 0.0 : kb * evalpoly(v, levq)
        eh, eb = _rdhc(vce.type, eh, eb, hp, hq, m, p, compat)
      end
      εh[j], εb[j] = eh, eb
    end
    if adjust
      _lsolve!(_powers!(view(Yp, :, j), u), Cp.U)
      KYp[:, j] .= kh .* view(Yp, :, j)
      _lsolve!(_powers!(view(Yq, :, j), v), Cq.U)
      KYq[:, j] .= kb .* view(Yq, :, j)
    end
  end
  scale2 = (factorial(ν) / h^ν)^2
  if vce isa ClusterVariance
    cl = view(d.cluster, a:e)
    ord = _clusterorder(cl)
    rh = compat ? :rdrobust : :default
    rb = compat ? (hb ? :rdrobust : :rdrobustqq) : :default
    keys, sh = _clustersums(cl, ord, ℓ, εh, Yp, KYp, vce.type, rh)
    sb = last(_clustersums(cl, ord, ℓbc, εb, Yq, KYq, vce.type, rb))
    G = length(keys)
    vh = _crscale(vce.type, m, G, p + 1, rh) * sum(abs2, sh)
    vb = _crscale(vce.type, m, G, q + 1, rb) * sum(abs2, sb)
    return (vh = scale2 * vh, vb = scale2 * vb)
  end
  return (vh = scale2 * sum(i -> (ℓ[i] * εh[i])^2, 1:m), vb = scale2 * sum(i -> (ℓbc[i] * εb[i])^2, 1:m))
end

# The HC residuals: scaled by √(n/(n - k)) for HC1 with k the coefficients of each
# fit, and by the leverages of the fits of order p and q for HC2 and HC3, which
# rdrobust floors at 1 - hᵢᵢ = 1e-8 and we make NaN at a leverage of one
function _rdhc(type, eh, eb, hp, hq, n, p, compat)
  type == 0 && return (eh, eb)
  type == 1 && return (eh * _dofscale(n, p + 1), eb * _dofscale(n, p + 2))
  sp, sq = 1 - hp, 1 - hq
  if compat
    sp, sq = max(sp, 1e-8), max(sq, 1e-8)
  else
    sp > sqrt(eps()) || (sp = NaN)
    sq > sqrt(eps()) || (sq = NaN)
  end
  return type == 2 ? (eh / sqrt(sp), eb / sqrt(sq)) : (eh / sp, eb / sq)
end

# the same sample with the responses y, in the sorted order of x
_withy(d::SmoothingData, y::Vector{Float64}) =
  SmoothingData(d.x, y, d.w, d.perm, d.values, d.starts, d.valueindex, d.cumw, d.cumwlo, d.sumw,
                d.neff, d.wexp, d.cluster)


## Interface

nobs(m::RegressionDiscontinuity) = sum(m.nobs)
bandwidth(m::RegressionDiscontinuity) = m.h

"""
    coef(m::RegressionDiscontinuity; method = Conventional())

The estimate of the jump as a one-element vector, or with
`method = RobustBiasCorrected()` the bias-corrected estimate, the centre of the
robust interval.
"""
StatsAPI.coef(m::RegressionDiscontinuity; method::InferenceMethod = Conventional()) =
  [_usesbc(method) ? m.estimate_bc : m.estimate]

"""
    stderror(m::RegressionDiscontinuity; method = Conventional())

The standard error of the estimate as a one-element vector, or with
`method = RobustBiasCorrected()` the robust standard error of the bias-corrected
estimate.
"""
stderror(m::RegressionDiscontinuity; method::InferenceMethod = Conventional()) =
  [_usesbc(method) ? m.se_rbc : m.se]

StatsAPI.vcov(m::RegressionDiscontinuity; method::InferenceMethod = Conventional()) =
  fill(stderror(m; method)[1]^2, 1, 1)

"""
    confint(m::RegressionDiscontinuity; level = 0.95, method = RobustBiasCorrected())

The confidence interval for the jump as a 1×2 matrix: by default the robust
bias-corrected interval, centred at the bias-corrected estimate, and with
`method = Conventional()` the interval around the estimate.
"""
function StatsAPI.confint(m::RegressionDiscontinuity; level::Real = 0.95,
                          method::InferenceMethod = RobustBiasCorrected())
  method isa Union{Conventional,RobustBiasCorrected} ||
    throw(ArgumentError("intervals for regression discontinuity are Conventional() or RobustBiasCorrected()"))
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  est, se = coef(m; method)[1], stderror(m; method)[1]
  return [est - z * se est + z * se]
end

"""
    coeftable(m::RegressionDiscontinuity; level = 0.95)

The results as rdrobust reports them, a named tuple of columns with the rows
`"Conventional"` (the estimate and its standard error), `"Bias-corrected"` (the
bias-corrected estimate with the same standard error) and `"Robust"` (the
bias-corrected estimate with its robust standard error): `method`, `estimate`,
`se`, `z`, `pvalue`, `lower` and `upper`.
"""
function StatsAPI.coeftable(m::RegressionDiscontinuity; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  est = [m.estimate, m.estimate_bc, m.estimate_bc]
  se = [m.se, m.se, m.se_rbc]
  z = est ./ se
  q = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  return (method = ["Conventional", "Bias-corrected", "Robust"], estimate = est, se = se, z = z,
          pvalue = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z)), lower = est .- q .* se,
          upper = est .+ q .* se)
end

function Base.show(io::IO, m::RegressionDiscontinuity)
  print(io, "RegressionDiscontinuity(", _rdkind(m), ", cutoff ", m.cutoff, ", ", nobs(m), " observations)")
end

_rdkind(m) = string(m.fuzzy ? "fuzzy" : "sharp", m.derivative == 1 ? " kink" : "",
                    m.derivative > 1 ? ", derivative $(m.derivative)" : "", m.ncovariates > 0 ? ", covariate-adjusted" : "")

function Base.show(io::IO, ::MIME"text/plain", m::RegressionDiscontinuity)
  println(io, "Regression discontinuity: ", _rdkind(m), ", local polynomials of degree ", m.degree,
          m.compat ? " (rdrobust compatibility)" : "")
  println(io, "  cutoff ", _fmt(m.cutoff), ", kernel: ", m.kernel, ", variance: ", m.vce)
  println(io, "  ", rpad("", 22), rpad("left", 12), "right")
  println(io, "  ", rpad("observations", 22), rpad(m.nobs[1], 12), m.nobs[2])
  println(io, "  ", rpad("in the window of h", 22), rpad(m.nh[1], 12), m.nh[2])
  println(io, "  ", rpad("bandwidth h", 22), rpad(_fmt(m.h[1]), 12), _fmt(m.h[2]))
  println(io, "  ", rpad("pilot bandwidth b", 22), rpad(_fmt(m.b[1]), 12), _fmt(m.b[2]))
  t = coeftable(m)
  println(io, "  ", rpad("", 16), rpad("estimate", 11), rpad("se", 11), rpad("z", 9), rpad("p-value", 10), "95% interval")
  for j in 1:3
    println(io, "  ", rpad(t.method[j], 16), rpad(_fmt(t.estimate[j]), 11), rpad(_fmt(t.se[j]), 11),
            rpad(_fmt(t.z[j]), 9), rpad(_fmt(t.pvalue[j]), 10), "[", _fmt(t.lower[j]), ", ", _fmt(t.upper[j]), "]")
  end
end
