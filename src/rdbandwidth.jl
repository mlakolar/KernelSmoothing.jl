## Bandwidth selection for regression discontinuity

const _RDSELECTORS = (:mserd, :msetwo, :msesum, :msecomb1, :msecomb2, :cerrd, :certwo, :cersum, :cercomb1, :cercomb2)

"""
    RDOptimal(bwselect = :mserd; regularization = 1)

The plug-in bandwidths of rdrobust's `rdbwselect` for [`RegressionDiscontinuity`](@ref)
(Calonico, Cattaneo and Titiunik 2014; Calonico, Cattaneo and Farrell 2020), named
as there:

- `:mserd`, the default: one bandwidth for both sides that minimizes the asymptotic
  mean squared error of the estimate of the jump;
- `:msetwo`: a bandwidth for each side, minimizing the mean squared errors of the
  estimates on the two sides;
- `:msesum`: one bandwidth that minimizes the mean squared error of the sum of the
  estimates on the two sides;
- `:msecomb1`: the smaller of `:mserd` and `:msesum`;
- `:msecomb2`: on each side, the median of `:mserd`, `:msesum` and `:msetwo`;
- `:cerrd`, `:certwo`, `:cersum`, `:cercomb1` and `:cercomb2`: these bandwidths times
  `n^(-p / ((p + 3)(2p + 3)))`, with `n` the number of observations or, for
  clustered data, of clusters, which shrinks them towards the bandwidths that
  minimize the coverage error of the robust intervals (Calonico, Cattaneo and
  Farrell 2020).

Each bandwidth `h` comes with a pilot bandwidth `b` for the bias correction,
selected in the same way for the derivative of order `p + 1`. The bias constants
are estimated by fits of order `p + 2` with bandwidths from a first stage, and the
variance constants by fits with a normal-reference bandwidth; `regularization`
scales the term that keeps the denominators away from zero where the estimated
bias nearly vanishes, as rdrobust's `scaleregul`.
"""
struct RDOptimal <: BandwidthSelector
  bwselect::Symbol
  regularization::Float64

  function RDOptimal(bwselect::Symbol = :mserd; regularization::Real = 1)
    bwselect in _RDSELECTORS || throw(ArgumentError("bwselect must be one of $(join(repr.(_RDSELECTORS), ", ")), got $(repr(bwselect))"))
    0 <= regularization < Inf || throw(ArgumentError("regularization must be nonnegative and finite, got $regularization"))
    new(bwselect, Float64(regularization))
  end
end

function Base.show(io::IO, r::RDOptimal)
  print(io, "RDOptimal(", repr(r.bwselect), r.regularization == 1 ? "" : "; regularization = $(r.regularization)", ")")
end

selectbandwidth(rule::RDOptimal, d::SmoothingData, k::AbstractKernel; kwargs...) =
  throw(ArgumentError("$rule chooses bandwidths for regression discontinuity; fit a RegressionDiscontinuity " *
                      "or call selectbandwidth(rule, x, y; cutoff)"))

# The bandwidths h and b on the left and on the right side by the plug-in rules of
# Calonico, Cattaneo and Titiunik (2014), Calonico, Cattaneo, Farrell and Titiunik
# (2019) and Calonico, Cattaneo and Farrell (2020), which reproduce rdrobust, from
# the sides, the cutoff c, and the fuzzy and covariate structure of the columns of
# D. Every rule has three stages: the pilot b′ for the derivative of order q + 1,
# whose bias is estimated by fits of order q + 2 over the whole side; the pilot b
# for the derivative of order p + 1, by fits of order q with b′; and h, by fits of
# order p with b. The variance constants come from fits with a preliminary
# normal-reference bandwidth hσ. The rules differ in the mean squared error that
# each stage minimizes: that of each side (msetwo), or that of the jump (mserd) or
# of the sum (msesum) of the estimates on the two sides.
function _rdselect(rule::RDOptimal, sides, c, k, p, ν, vce, masspoints, bwcheck, fuzzy, dZ, compat)
  q = p + 1
  d₋, d₊ = sides[1].data, sides[2].data
  N = nobs(d₋) + nobs(d₊)
  details = (bwselect = rule.bwselect, masspoints = false, bwcheck = bwcheck)
  if N < 20
    # too few observations for the plug-in rules: the largest distance to the cutoff
    hmax = max(c - d₋.x[1], d₊.x[end] - c)
    return (hmax, hmax), (hmax, hmax), details
  end
  # the bandwidths are selected for x and the outcome divided by their standard
  # deviations, as in rdrobust
  xs = vcat(d₋.x, d₊.x)
  σx = _sd(xs)
  σy = _sd(vcat(sides[1].D[:, 1], sides[2].D[:, 1]))
  S₋, S₊ = _rdstandardize(sides[1], σx, σy), _rdstandardize(sides[2], σx, σy)
  c̃ = c / σx
  xs ./= σx
  # with perfect compliance on a side, the bandwidths are those of the sharp design
  fuzzybw = fuzzy && !(allequal(view(sides[1].D, :, 2)) || allequal(view(sides[2].D, :, 2)))
  if fuzzy && !fuzzybw
    keep = [1; 3:size(S₋.D, 2)]
    S₋, S₊ = (data = S₋.data, D = S₋.D[:, keep]), (data = S₊.data, D = S₊.D[:, keep])
  end
  nv₋, nv₊ = nunique(S₋.data), nunique(S₊.data)   # the distinct values on each side
  if masspoints in (:check, :adjust)
    mass = 1 - nv₋ / nobs(d₋) >= 0.2 || 1 - nv₊ / nobs(d₊) >= 0.2
    details = merge(details, (masspoints = mass,))
    mass && masspoints === :adjust && bwcheck === nothing && (bwcheck = 10)
  end
  details = merge(details, (bwcheck = bwcheck,))
  # the caps, the distances to the farthest observation on each side and on either
  # side, and with bwcheck the floors of hσ and b′, the distances to the bwcheck-th
  # nearest distinct value on each side
  hmax₋, hmax₊ = c̃ - xs[1], xs[end] - c̃
  hmax = max(hmax₋, hmax₊)
  hmin₋ = bwcheck === nothing ? 0.0 : (c̃ - S₋.data.values[end - min(bwcheck, nv₋) + 1]) * (1 + sqrt(eps()))
  hmin₊ = bwcheck === nothing ? 0.0 : (S₊.data.values[min(bwcheck, nv₊)] - c̃) * (1 + sqrt(eps()))
  # the preliminary bandwidth: a normal-reference rule for the number of
  # observations, or of distinct values when mass points are adjusted for
  spread = min(_sd(xs), (_quantile2(xs, 0.75) - _quantile2(xs, 0.25)) / 1.349)
  hσ₀ = min(_nrconstant(k) * spread * (masspoints === :adjust ? nv₋ + nv₊ : N)^(-1 / 5), hmax)
  hσ = bwcheck === nothing ? hσ₀ : max(hσ₀, hmin₋, hmin₊)
  λ = rule.regularization
  constants(S, o, nu, b, λ) = _rdsideconstants(S.data, S.D, k, c̃, o, nu, hσ, b, λ, vce, fuzzybw, dZ, compat)
  # the first stage on each side, for the derivative of order q + 1, whose bias the
  # fits of order q + 2 over the whole side estimate
  M′₋ = constants(S₋, q + 1, q + 1, hmax₋ * (1 + sqrt(eps())), 0.0)
  M′₊ = constants(S₊, q + 1, q + 1, hmax₊ * (1 + sqrt(eps())), 0.0)
  base = Symbol(replace(String(rule.bwselect), "cer" => "mse"))   # the rule of the MSE bandwidths
  local jump, total, two
  if base in (:mserd, :msecomb1, :msecomb2)
    jump = _rdcommon(constants, S₋, S₊, M′₋, M′₊, -, p, ν, λ, hmax, max(hmin₋, hmin₊))
  end
  if base in (:msesum, :msecomb1, :msecomb2)
    total = _rdcommon(constants, S₋, S₊, M′₋, M′₊, +, p, ν, λ, hmax, max(hmin₋, hmin₊))
  end
  if base in (:msetwo, :msecomb2)
    two = (_rdseparate(constants, S₋, M′₋, p, ν, λ, hmax₋, hmin₋), _rdseparate(constants, S₊, M′₊, p, ν, λ, hmax₊, hmin₊))
  end
  h, b = if base === :mserd
    (jump[1], jump[1]), (jump[2], jump[2])
  elseif base === :msesum
    (total[1], total[1]), (total[2], total[2])
  elseif base === :msetwo
    (two[1][1], two[2][1]), (two[1][2], two[2][2])
  elseif base === :msecomb1
    (min(jump[1], total[1]), min(jump[1], total[1])), (min(jump[2], total[2]), min(jump[2], total[2]))
  else
    # on each side, the middle one of the three
    mid(x, y, z) = max(min(x, y), min(max(x, y), z))
    (mid(jump[1], total[1], two[1][1]), mid(jump[1], total[1], two[2][1])),
      (mid(jump[2], total[2], two[1][2]), mid(jump[2], total[2], two[2][2]))
  end
  if startswith(String(rule.bwselect), "cer")
    # the coverage-error rate, with n the observations or the clusters
    n = d₋.cluster === nothing ? N : _nclusters(d₋) + _nclusters(d₊)
    h = h .* n^(-(p / ((3 + p) * (3 + 2p))))
  end
  all(isfinite, (h..., b...)) || throw(ArgumentError("the plug-in bandwidths are not defined, since a " *
    "local fit of their stages has fewer distinct values of x with positive weight than coefficients" *
    (d₋.cluster === nothing ? "" : ", or fewer than two clusters") * "; pass a bandwidth"))
  return h .* σx, b .* σx, details
end

# The bandwidth (V / B²)^(1/(2o+3)) of a fit of order o, capped at hmax, from its
# variance and squared-bias constants, which carry the factors of the first-order
# condition of the mean squared error (see _rdsideconstants)
_rdplugin(V, B², o, hmax) = min((V / B²)^(1 / (2o + 3)), hmax)

# The bandwidths h and b of one side alone, each stage minimizing the mean squared
# error on that side, from its first stage M′; the pilot b′ is floored at hmin
function _rdseparate(constants, S, M′, p, ν, λ, hmax, hmin)
  q = p + 1
  side(M, λ) = (M.V, M.B^2 + λ * M.ℛ)
  b′ = max(_rdplugin(side(M′, 0.0)..., q + 1, hmax), hmin)
  b = _rdplugin(side(constants(S, q, p + 1, b′, λ), λ)..., q, hmax)
  h = _rdplugin(side(constants(S, p, ν, b, λ), λ)..., p, hmax)
  return h, b
end

# The common bandwidths h and b of both sides, each stage minimizing the mean squared
# error of the jump (op = -) or of the sum (op = +) of the estimates on the two
# sides, whose variance constants add and whose bias constants combine by op, from
# the first stages M′₋ and M′₊; the pilot b′ is floored at hmin
function _rdcommon(constants, S₋, S₊, M′₋, M′₊, op, p, ν, λ, hmax, hmin)
  q = p + 1
  pooled(M₋, M₊, λ) = (M₋.V + M₊.V, op(M₋.B, M₊.B)^2 + λ * (M₋.ℛ + M₊.ℛ))
  both(o, nu, b) = pooled(constants(S₋, o, nu, b, λ), constants(S₊, o, nu, b, λ), λ)
  b′ = max(_rdplugin(pooled(M′₋, M′₊, 0.0)..., q + 1, hmax), hmin)
  b = _rdplugin(both(q, p + 1, b′)..., q, hmax)
  h = _rdplugin(both(p, ν, b)..., p, hmax)
  return h, b
end

# a side with x divided by σx and the outcome, the first column of D, by σy
function _rdstandardize(side, σx, σy)
  d = side.data
  ds = SmoothingData(d.x ./ σx, d.y ./ σy, d.w, d.perm, d.values ./ σx, d.starts, d.valueindex, d.cumw,
                     d.cumwlo, d.sumw, d.neff, d.wexp, d.cluster)
  D = copy(side.D)
  D[:, 1] ./= σy
  return (data = ds, D = D)
end

_sd(v) = sqrt(sum(abs2, v .- sum(v) / length(v)) / (length(v) - 1))

# R's quantile of type 2 of sorted values: the inverse of the empirical distribution
# function, averaging at its jumps
function _quantile2(xs, p)
  n = length(xs)
  t = n * p
  j = floor(Int, t)
  g = t - j
  lo, hi = xs[clamp(j, 1, n)], xs[clamp(j + 1, 1, n)]
  return g > 0 ? hi : (lo + hi) / 2
end

# The constants of the mean squared error on one side of the estimate of the
# derivative ν by a fit of order o (Calonico, Cattaneo and Titiunik 2014;
# Calonico, Cattaneo, Farrell and Titiunik 2019): the variance constant V
# from the variance of the coefficient of order ν of the fit with the preliminary
# bandwidth hσ, the bias constant B = e_ν'Γ⁻¹Λ₁ μ⁽ᵒ⁺¹⁾/(o+1)!, with Γ and Λ₁ the Gram
# matrix and the moments of order o + 1 of that fit and the derivative from the fit
# of order o + 1 with the pilot bandwidth b, and the term ℛ = 3 (e_ν'Γ⁻¹Λ₁)² V̂₁ that
# regularizes B² (Imbens and Kalyanaraman 2012), with V̂₁ the variance of that
# derivative. For a fuzzy design or with covariates, the outcome is combined with
# the treatment and the covariates by the weights of the fit with hσ, as in
# rdrobust. The constants carry the factors of the first-order condition of the
# mean squared error, V the factor 2ν + 1, B the factor √(2(o + 1 - ν)) and ℛ the
# factor 2(o + 1 - ν) λ, so that the bandwidth that minimizes it is (V / B²)^(1/(2o+3))
# for one side and for sums of both alike; the stages add λℛ to B², so that the
# regularization enters with λ², as in rdrobust. The factor of B is applied on each
# side, before the sides are combined, which fixes the rounding of the bandwidths
# that reproduce rdrobust; applied to the combined B², it changes their last bits.
# NaN where a fit is not identified or the pilot b is not a positive number.
function _rdsideconstants(d, D, k, c, o, ν, hσ, b, λ, vce, fuzzy, dZ, compat)
  fσ = _rdpolyfit(d, D, k, c, hσ, o)
  fb = fσ.ok && isfinite(b) && b > 0 ? _rdpolyfit(d, D, k, c, b, o + 1) : (ok = false,)
  fσ.ok && fb.ok || return (V = NaN, B = NaN, ℛ = NaN)
  s = _rdbwweights(fσ, ν, fuzzy, dZ, hσ)
  L = (fσ.C \ [j == ν ? 1.0 : 0.0 for j in 0:o])' * fσ.Λ   # e_ν'Γ⁻¹Λ₁
  kB, kV = 2(o + 1 - ν), 2ν + 1
  V = kV * hσ^(2ν + 1) * _rdcoefvar(d, D, s, k, c, fσ, ν, vce, compat)
  B = sqrt(kB) * L * (s' * fb.β[o + 2, :]) / b^(o + 1)
  ℛ = λ > 0 ? λ * kB * (3 * L^2 * _rdcoefvar(d, D, s, k, c, fb, o + 1, vce, compat)) : 0.0
  return (V = V, B = B, ℛ = ℛ)
end

# The local polynomial fit of order o with bandwidth h at c for the columns of D, in
# the scaled variable u = (x - c)/h: the window, the factorization of the Gram matrix,
# the coefficients, Λ = ∑ wᵢ k(uᵢ) uᵢ^(o+1) r(uᵢ) and the cross-products for the
# covariates; whether the window holds o + 1 distinct values
function _rdpolyfit(d, D, k, c, h, o)
  r = _poswindow(d, k, c, h)
  m = size(D, 2)
  ok = !isempty(r) && nunique(d, r) >= o + 1
  ok || return (ok = false,)
  S, T, DD = zeros(2o + 2), zeros(o + 1, m), zeros(m, m)
  for i in r
    u = (d.x[i] - c) / h
    kw = _weight(d.w, i) * k(u)
    _powersums!(S, kw, u)
    for l in 1:m
      _powersums!(view(T, :, l), kw * D[i, l], u)
      for l2 in l:m
        DD[l, l2] += kw * D[i, l] * D[i, l2]
      end
    end
  end
  C, okC = _gramfactor([S[i + j + 1] for i in 0:o, j in 0:o], nunique(d, r))
  okC || return (ok = false,)
  for l in 1:m, l2 in 1:l-1
    DD[l, l2] = DD[l2, l]
  end
  β = C \ T
  return (ok = true, a = first(r), e = last(r), h = h, o = o, C = C, β = β, T = T, DD = DD, Λ = S[o+2:2o+2])
end

# The weights of the columns of D (outcome, treatment, covariates) in the response
# whose fits give the bandwidths: the coefficients of the covariates and the ratio of
# the estimates of the outcome and the treatment come from the fit f on this side
function _rdbwweights(f, ν, fuzzy, dZ, h)
  m = size(f.β, 2)
  dT = Int(fuzzy)
  s = zeros(m)
  Zc = (2 + dT):(1 + dT + dZ)
  γ = zeros(dZ, 1 + dT)
  if dZ > 0
    ZW = f.DD .- f.T' * f.β   # D'WD - U'Γ⁻¹U
    γ = LinearAlgebra.pinv(ZW[Zc, Zc]; rtol = 1e-15) * ZW[Zc, 1:1+dT]
  end
  if !fuzzy
    s[1] = 1
    s[Zc] .= -γ[:, 1]
    return s
  end
  τ(col, g) = factorial(ν) * (f.β[ν + 1, col] - (dZ > 0 ? g' * f.β[ν + 1, Zc] : 0.0)) / h^ν
  τY, τT = τ(1, γ[:, 1]), τ(2, dZ > 0 ? γ[:, 2] : zeros(0))
  s[1], s[2] = 1 / τT, -τY / τT^2
  for l in 1:dZ
    s[2 + l] = -γ[l, 1] / τT + τY * γ[l, 2] / τT^2
  end
  return s
end

# The variance of the raw coefficient of order j of the fit f for the response D s,
# with rdrobust's residuals of that fit: nearest neighbours, or its own residuals,
# leverages and, for clusters, the blocks of its own hat matrix
function _rdcoefvar(d, D, s, k, c, f, j, vce, compat)
  a, e, h, o, C = f.a, f.e, f.h, f.o, f.C
  m = e - a + 1
  ys = D * s
  β = f.β * s
  g = C \ [l == j ? 1.0 : 0.0 for l in 0:o]
  lev = vce isa HCVariance && vce.type >= 2 ? _leveragepoly(C, o) : nothing
  nn = vce isa NNVariance ? _nnresiduals(_withy(d, ys), a, e, vce.matches, _nntol(false)) : nothing
  adjust = vce isa ClusterVariance && vce.type >= 2
  Y = adjust ? Matrix{Float64}(undef, o + 1, m) : nothing
  KY = adjust ? Matrix{Float64}(undef, o + 1, m) : nothing
  ℓ, ε = Vector{Float64}(undef, m), Vector{Float64}(undef, m)
  for i in a:e
    l = i - a + 1
    u = (d.x[i] - c) / h
    kw = _weight(d.w, i) * k(u)
    ℓ[l] = kw * evalpoly(u, g)
    ε[l] = if vce isa NNVariance
      nn[l]
    else
      r = ys[i] - evalpoly(u, β)
      vce isa HCVariance ? first(_rdhc(vce.type, r, r, lev === nothing ? 0.0 : kw * evalpoly(u, lev), 0.0, m, o, compat)) : r
    end
    if adjust
      _lsolve!(_powers!(view(Y, :, l), u), C.U)
      KY[:, l] .= kw .* view(Y, :, l)
    end
  end
  v = if vce isa ClusterVariance
    cl = view(d.cluster, a:e)
    rule = compat ? :rdrobust : :default
    keys, sums = _clustersums(cl, _clusterorder(cl), ℓ, ε, Y, KY, vce.type, rule)
    _crscale(vce.type, m, length(keys), o + 1, rule) * sum(abs2, sums)
  else
    sum(i -> (ℓ[i] * ε[i])^2, 1:m)
  end
  return v / h^(2j)
end
