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

# The bandwidths h and b on the left and on the right side by rdrobust's rules, from
# the sides, the cutoff c, and the fuzzy and covariate structure of the columns of D
function _rdselect(rule::RDOptimal, sides, c, k, p, ν, vce, masspoints, bwcheck, fuzzy, dZ, compat)
  q = p + 1
  dl, dr = sides[1].data, sides[2].data
  N = nobs(dl) + nobs(dr)
  details = (bwselect = rule.bwselect, masspoints = false, bwcheck = bwcheck)
  if N < 20
    # too few observations for the plug-in rules: the largest distance to the cutoff
    hmax = max(c - dl.x[1], dr.x[end] - c)
    return (hmax, hmax), (hmax, hmax), details
  end
  # rdrobust standardizes x and y, and selects the bandwidths in these units
  xs = vcat(dl.x, dr.x)
  xsd = _sd(xs)
  ysd = _sd(vcat(sides[1].D[:, 1], sides[2].D[:, 1]))
  sl, sr = _rdstandardize(sides[1], xsd, ysd), _rdstandardize(sides[2], xsd, ysd)
  cs = c / xsd
  xs ./= xsd
  # with perfect compliance on a side, the bandwidths are those of the sharp design
  dT = fuzzy && !(allequal(view(sides[1].D, :, 2)) || allequal(view(sides[2].D, :, 2))) ? 1 : 0
  if fuzzy && dT == 0
    keep = [1; 3:size(sl.D, 2)]
    sl, sr = (data = sl.data, D = sl.D[:, keep]), (data = sr.data, D = sr.D[:, keep])
  end
  Ml, Mr = nunique(sl.data), nunique(sr.data)
  if masspoints in (:check, :adjust)
    mass = 1 - Ml / nobs(dl) >= 0.2 || 1 - Mr / nobs(dr) >= 0.2
    details = merge(details, (masspoints = mass,))
    mass && masspoints === :adjust && bwcheck === nothing && (bwcheck = 10)
  end
  details = merge(details, (bwcheck = bwcheck,))
  spread = min(_sd(xs), (_quantile2(xs, 0.75) - _quantile2(xs, 0.25)) / 1.349)
  cbw = _nrconstant(k) * spread * (masspoints === :adjust ? Ml + Mr : N)^(-1 / 5)
  maxl, maxr = cs - xs[1], xs[end] - cs
  bwmax = max(maxl, maxr)
  cbw = min(cbw, bwmax)
  minl = minr = 0.0
  if bwcheck !== nothing
    # the distances of the bwcheck-th nearest distinct values on the two sides
    minl = (cs - sl.data.values[end - min(bwcheck, Ml) + 1]) * (1 + sqrt(eps()))
    minr = (sr.data.values[min(bwcheck, Mr)] - cs) * (1 + sqrt(eps()))
    cbw = max(cbw, minl, minr)
  end
  reg = rule.regularization
  bw(s, o, nu, oB, hB, scale) = _rdbw(s.data, s.D, k, cs, o, nu, oB, cbw, hB, scale, vce, dT == 1, dZ, compat)
  # the first stage, for the derivative of order q + 1 by fits of order q + 2
  Cdl = bw(sl, q + 1, q + 1, q + 2, maxl * (1 + sqrt(eps())), 0.0)
  Cdr = bw(sr, q + 1, q + 1, q + 2, maxr * (1 + sqrt(eps())), 0.0)
  sel = rule.bwselect
  name = Symbol(replace(String(sel), "cer" => "mse"))
  rate(C) = C.rate
  local hrd, brd, hsum, bsum, htwo, btwo
  if name in (:msetwo, :msecomb2)
    dlb = min((Cdl.V / Cdl.B^2)^rate(Cdl), maxl)
    drb = min((Cdr.V / Cdr.B^2)^rate(Cdr), maxr)
    bwcheck === nothing || (dlb = max(dlb, minl); drb = max(drb, minr))
    Cbl, Cbr = bw(sl, q, p + 1, q + 1, dlb, reg), bw(sr, q, p + 1, q + 1, drb, reg)
    bl = min((Cbl.V / (Cbl.B^2 + reg * Cbl.R))^rate(Cbl), maxl)
    br = min((Cbr.V / (Cbr.B^2 + reg * Cbr.R))^rate(Cbr), maxr)
    Chl, Chr = bw(sl, p, ν, q, bl, reg), bw(sr, p, ν, q, br, reg)
    hl = min((Chl.V / (Chl.B^2 + reg * Chl.R))^rate(Chl), maxl)
    hr = min((Chr.V / (Chr.B^2 + reg * Chr.R))^rate(Chr), maxr)
    htwo, btwo = (hl, hr), (bl, br)
  end
  if name in (:msesum, :msecomb1, :msecomb2)
    hsum, bsum = _rdcommon(bw, sl, sr, Cdl, Cdr, +, p, ν, q, reg, bwmax, bwcheck === nothing ? 0.0 : max(minl, minr))
  end
  if name in (:mserd, :msecomb1, :msecomb2)
    hrd, brd = _rdcommon(bw, sl, sr, Cdl, Cdr, -, p, ν, q, reg, bwmax, bwcheck === nothing ? 0.0 : max(minl, minr))
  end
  h, b = if name === :mserd
    (hrd, hrd), (brd, brd)
  elseif name === :msesum
    (hsum, hsum), (bsum, bsum)
  elseif name === :msetwo
    htwo, btwo
  elseif name === :msecomb1
    (min(hrd, hsum), min(hrd, hsum)), (min(brd, bsum), min(brd, bsum))
  else
    med(a, b, c) = a + b + c - max(a, b, c) - min(a, b, c)
    (med(hrd, hsum, htwo[1]), med(hrd, hsum, htwo[2])), (med(brd, bsum, btwo[1]), med(brd, bsum, btwo[2]))
  end
  if String(sel)[1:3] == "cer"
    G = dl.cluster === nothing ? N : _nclusters(dl) + _nclusters(dr)
    h = h .* G^(-(p / ((3 + p) * (3 + 2p))))
  end
  all(isfinite, (h..., b...)) || throw(ArgumentError("the plug-in bandwidths are not defined, since a " *
    "local fit of their stages has fewer distinct values of x with positive weight than coefficients" *
    (dl.cluster === nothing ? "" : ", or fewer than two clusters") * "; pass a bandwidth"))
  return h .* xsd, b .* xsd, details
end

# The common bandwidths h and b of the jump (op = -) or of the sum (op = +) of the
# estimates on the two sides, from the first stage Cdl and Cdr
function _rdcommon(bw, sl, sr, Cdl, Cdr, op, p, ν, q, reg, bwmax, bwmin)
  d = min(((Cdl.V + Cdr.V) / op(Cdr.B, Cdl.B)^2)^Cdl.rate, bwmax)
  d = max(d, bwmin)
  Cbl, Cbr = bw(sl, q, p + 1, q + 1, d, reg), bw(sr, q, p + 1, q + 1, d, reg)
  b = min(((Cbl.V + Cbr.V) / (op(Cbr.B, Cbl.B)^2 + reg * (Cbr.R + Cbl.R)))^Cbl.rate, bwmax)
  Chl, Chr = bw(sl, p, ν, q, b, reg), bw(sr, p, ν, q, b, reg)
  h = min(((Chl.V + Chr.V) / (op(Chr.B, Chl.B)^2 + reg * (Chr.R + Chl.R)))^Chl.rate, bwmax)
  return h, b
end

# a side with x divided by xsd and the outcome, the first column of D, by ysd
function _rdstandardize(side, xsd, ysd)
  d = side.data
  ds = SmoothingData(d.x ./ xsd, d.y ./ ysd, d.w, d.perm, d.values ./ xsd, d.starts, d.valueindex, d.cumw,
                     d.cumwlo, d.sumw, d.neff, d.wexp, d.cluster)
  D = copy(side.D)
  D[:, 1] ./= ysd
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

# rdrobust's rdrobust_bw on one side: the variance constant V of the derivative ν of
# the fit of order o with bandwidth hV, the bias constant B from the coefficient of
# order o + 1 of the fit of order oB with bandwidth hB, the regularization R and the
# rate 1/(2o + 3). For a fuzzy design or with covariates, the outcome is combined
# with the treatment and the covariates by the weights of the fit with hV, as in
# rdrobust. NaN where a fit is not identified.
function _rdbw(d, D, k, c, o, ν, oB, hV, hB, scale, vce, fuzzy, dZ, compat)
  rate = 1 / (2o + 3)
  nan = (V = NaN, B = NaN, R = NaN, rate = rate)
  fV = _rdpolyfit(d, D, k, c, hV, o)
  fV.ok || return nan
  s = _rdbwweights(fV, ν, fuzzy, dZ, hV)
  VV = _rdcoefvar(d, D, s, k, c, fV, ν, vce, compat)
  BConst = (fV.C \ [j == ν ? 1.0 : 0.0 for j in 0:o])' * fV.Λ
  isfinite(hB) && hB > 0 || return nan
  fB = _rdpolyfit(d, D, k, c, hB, oB)
  fB.ok || return nan
  BWreg = scale > 0 ? 3 * BConst^2 * _rdcoefvar(d, D, s, k, c, fB, o + 1, vce, compat) : 0.0
  B = sqrt(2(o + 1 - ν)) * BConst * (s' * fB.β[o + 2, :]) / hB^(o + 1)
  V = (2ν + 1) * hV^(2ν + 1) * VV
  return (V = V, B = B, R = scale * 2(o + 1 - ν) * BWreg, rate = rate)
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
