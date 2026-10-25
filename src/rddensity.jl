## The manipulation test of a regression discontinuity design

"""
    RDDensityOptimal(bwselect = :comb)

The bandwidths of rddensity's `rdbwdensity` for [`DensityDiscontinuity`](@ref):
bandwidths that minimize the asymptotic mean squared error of the local polynomial
density estimates of order `p` on each side of the cutoff (`:each`), of their
difference (`:diff`) or of their sum (`:sum`), or on each side the median of these
three (`:comb`, the default). In the restricted model, `:comb` is the smaller of
`:diff` and `:sum`, and `:each` is not available. The bias constants come from fits
of order `p + 2` and the variances from fits of order `p`, with normal-reference
pilot bandwidths, as in rddensity.
"""
struct RDDensityOptimal <: BandwidthSelector
  bwselect::Symbol

  function RDDensityOptimal(bwselect::Symbol = :comb)
    bwselect in (:each, :diff, :sum, :comb) ||
      throw(ArgumentError("bwselect must be :each, :diff, :sum or :comb, got $(repr(bwselect))"))
    new(bwselect)
  end
end

Base.show(io::IO, r::RDDensityOptimal) = print(io, "RDDensityOptimal(", repr(r.bwselect), ")")

select_bandwidth(rule::RDDensityOptimal, d::SmoothingData, k::AbstractKernel; kwargs...) =
  throw(ArgumentError("$rule chooses bandwidths for the manipulation test; fit a DensityDiscontinuity"))

"""
    DensityDiscontinuity <: StatisticalModel

The manipulation test of Cattaneo, Jansson and Ma (2020) for a regression
discontinuity design, as rddensity: local polynomial estimates of the density of
the running variable on the two sides of the cutoff, fitted with
[`fit`](@ref)`(DensityDiscontinuity, x; cutoff)`, and the test that the density is
continuous at the cutoff. A density that jumps there suggests that units sort
around the cutoff.

The density on each side is the slope at the cutoff of a local polynomial fit to
the empirical distribution function, as for [`LocalPolynomialDensity`](@ref), with
separate polynomials on the two sides (or, with `restricted = true`, a distribution
function that is continuous at the cutoff and has common coefficients of order two
and higher). The test uses the fits of order `degree + 1` at bandwidths chosen for
order `degree`, which makes it robust to the bias, and its standard error comes
from the jackknife or from the plug-in formula.

`coef(m)` gives the estimated jump `f̂₊ - f̂₋`, `stderror(m)` its standard error,
`m.statistic` and `m.pvalue` the test, `coeftable(m)` the estimates on both sides,
and `m.binomial` the exact binomial tests of the numbers of observations on the two
sides in windows around the cutoff, as rddensity reports them.
"""
struct DensityDiscontinuity{K<:AbstractKernel,S<:BandwidthSelection} <: StatsAPI.StatisticalModel
  cutoff::Float64
  kernel::K
  degree::Int
  restricted::Bool
  vce::Symbol
  masspoints::Bool               # whether the ties of the running variable were corrected for
  selection::S
  h::NTuple{2,Float64}           # the bandwidths on the left and on the right
  nobs::NTuple{2,Int}            # observations on each side
  nh::NTuple{2,Int}              # in the windows
  density::NTuple{2,Float64}     # the estimates on the left and on the right
  se::NTuple{3,Float64}          # standard errors of the estimates on the two sides and of the jump
  statistic::Float64
  pvalue::Float64
  binomial::NamedTuple
end

"""
    fit(DensityDiscontinuity, x; cutoff = 0, degree = 2, kernel = TriangularKernel(),
        bandwidth = RDDensityOptimal(), restricted = false, vce = :jackknife,
        masspoints = true, regularize = true, minobs = degree + 21, minunique = degree + 21,
        binomialn = 20, binomialwindows = 10)

Tests whether the density of `x` jumps at `cutoff`, as rddensity, from local
polynomial density estimates of order `degree + 1` on each side at bandwidths
chosen for order `degree`.

- `bandwidth` is a number, a pair for the left and the right side, or an
  [`RDDensityOptimal`](@ref) rule.
- `restricted = true` fits a distribution function that is continuous at the
  cutoff, with a common bandwidth.
- `vce` is `:jackknife` (the default) or `:plugin`, the variance estimators of
  rddensity.
- `masspoints = true` corrects for ties in `x`: tied observations share the value
  of the empirical distribution function, as in rddensity.
- `regularize` keeps the selected bandwidths below the range of the data and above
  the distances of the `minobs`-th nearest observation and the `minunique`-th nearest
  distinct value on each side.
- `binomialn` and `binomialwindows` set the binomial tests: the first window holds
  at least `binomialn` observations on each side, and the windows grow up to the
  bandwidths.
"""
function fit(::Type{DensityDiscontinuity}, x::AbstractVector; cutoff::Real = 0, degree::Integer = 2,
             kernel::AbstractKernel = TriangularKernel(), bandwidth = RDDensityOptimal(),
             restricted::Bool = false, vce::Symbol = :jackknife, masspoints::Bool = true,
             regularize::Bool = true, minobs::Integer = degree + 21, minunique::Integer = degree + 21,
             binomialn::Integer = 20, binomialwindows::Integer = 10)
  _check_values(x, "x")
  c = Float64(cutoff)
  p = Int(degree)
  1 <= p <= 7 || throw(ArgumentError("degree must be between 1 and 7, got $degree"))
  vce in (:jackknife, :plugin) || throw(ArgumentError("vce must be :jackknife or :plugin, got $(repr(vce))"))
  minobs >= 0 && minunique >= 0 || throw(ArgumentError("minobs and minunique must be nonnegative"))
  binomialn >= 1 && binomialwindows >= 1 || throw(ArgumentError("binomialn and binomialwindows must be positive"))
  _rddkernel(kernel)
  X = sort!(Float64.(x)) .- c
  N = length(X)
  X[1] < 0 < X[end] || throw(ArgumentError("the cutoff must lie inside the range of x"))
  Nl = count(<(0), X)
  ties = masspoints && any(i -> X[i] == X[i - 1], 2:N)
  Y = _rddecdf(X, ties)
  sel = if bandwidth isa RDDensityOptimal
    bandwidth.bwselect === :each && restricted &&
      throw(ArgumentError("bwselect = :each is not available in the restricted model"))
    hs = _rddbandwidths(X, Y, N, Nl, p, kernel, restricted, vce, ties, regularize, minobs, minunique)
    b = bandwidth.bwselect
    med(a, b, c) = max(min(a, b), min(max(a, b), c))   # exactly one of them, as R's median
    h = if b === :each
      (hs.left, hs.right)
    elseif b === :diff
      (hs.diff, hs.diff)
    elseif b === :sum
      (hs.sum, hs.sum)
    elseif restricted
      (min(hs.diff, hs.sum), min(hs.diff, hs.sum))
    else
      (med(hs.left, hs.diff, hs.sum), med(hs.right, hs.diff, hs.sum))
    end
    BandwidthSelection([h...], bandwidth, hs)
  elseif bandwidth isa BandwidthSelector
    throw(ArgumentError("$bandwidth chooses bandwidths for other estimators; use RDDensityOptimal()"))
  else
    BandwidthSelection([_rdbandwidths(bandwidth, "bandwidth")...], nothing, NamedTuple())
  end
  hl, hr = sel.h[1], sel.h[2]
  restricted && hl != hr && throw(ArgumentError("the restricted model needs the same bandwidth on both sides"))
  win = findall(t -> -hl <= t <= hr, X)
  Xh, Yh = X[win], Y[win]
  Nlh = count(<(0), Xh)
  f = _rddfit(Xh, Yh, N, Nlh, hl, hr, p + 1, 1, kernel, restricted, vce, ties)
  se = sqrt.((f.var.left, f.var.right, f.var.diff))
  T = f.est.diff / se[3]
  pv = 2 * Distributions.ccdf(Distributions.Normal(), abs(T))
  bino = _rddbinomial(X, N, Nl, hl, hr, binomialn, binomialwindows)
  return DensityDiscontinuity(c, kernel, p, restricted, vce, ties, _anyselection(sel), (hl, hr), (Nl, N - Nl),
                              (Nlh, length(win) - Nlh), (f.est.left, f.est.right), se, T, pv, bino)
end

function _rddkernel(k)
  hasmethod(pieces, Tuple{typeof(k)}) && support(k) == (-1.0, 1.0) ||
    throw(ArgumentError("the manipulation test needs a polynomial kernel on [-1, 1], such as TriangularKernel(), got $k"))
  return nothing
end

# The empirical distribution function (i - 1)/(N - 1) of the sorted X, and with ties
# the value of the last of the tied observations, as rddensity
function _rddecdf(X, ties)
  N = length(X)
  Y = collect((0:N-1) ./ (N - 1))
  if ties
    i = N
    while i >= 1
      j = i
      while j > 1 && X[j - 1] == X[i]
        j -= 1
      end
      Y[j:i] .= Y[i]
      i = j - 1
    end
  end
  return Y
end

## The local polynomial fits on the two sides of the cutoff
#
# The density estimator of Cattaneo, Jansson and Ma (2020, JASA) regresses the
# empirical distribution function F̂ on a polynomial in u = x/h by kernel-weighted
# least squares; the coefficient of u is h times the density at the cutoff. The
# manipulation test (Cattaneo, Jansson and Ma 2018, Stata Journal, and 2020) fits
# the two sides of the cutoff at once, with u = x/hl on the left and u = x/hr on
# the right and the weights k(u)/h:
#
# - unrestricted, a polynomial of order p on each side, coefficients
#   (1, u, …, uᵖ)·1{left} followed by (1, u, …, uᵖ)·1{right};
# - restricted, a distribution function that is continuous at the cutoff, with a
#   common intercept, one slope on each side and common coefficients of order two
#   and higher, coefficients (1, u·1{left}, u·1{right}, u², …, uᵖ).
#
# A named tuple (left, right, diff, sum) holds a quantity for the two sides, their
# difference (right minus left) and their sum.

_rddsides(l, r) = (left = l, right = r, diff = r - l, sum = r + l)
const _RDDNANS = (left = NaN, right = NaN, diff = NaN, sum = NaN)

# variances below zero, possible for the plug-in estimates and from rounding, are NaN
_rddnonneg(v) = map(t -> t < 0 ? NaN : t, v)

# The index, among the restricted coefficients, of the coefficient of uʲ on one side
_rddslot(j, left) = j == 0 ? 1 : j == 1 ? (left ? 2 : 3) : j + 2

# The fit of order p to the Nh sorted observations X of the window [-hl, hr], the
# first Nl of them on the left, with values Y of the empirical distribution function
# of all N observations. Returns (est, coef, var): the densities on the two sides,
# the coefficients of order s divided by the bandwidth to the power s, and the
# variances of the densities, each as (left, right, diff, sum). All are NaN when the
# fit is not identified: a side with fewer distinct values of positive weight than
# the design needs, where the normal matrix is singular and its solution arbitrary,
# or a normal matrix that is singular to working precision. With variance = false
# the variances are NaN, for the pilot fits that do not need them.
function _rddfit(X, Y, N, Nl, hl, hr, p, s, k, restricted, vce, ties; variance::Bool = true)
  Nh = length(X)
  nan = (est = _RDDNANS, coef = _RDDNANS, var = _RDDNANS)
  # kernel-weighted power sums of each side: P[m + 1] = ∑ w uᵐ, m = 0, …, 2p, and
  # Q[j + 1] = ∑ w uʲ F̂, j = 0, …, p, and the numbers of distinct values with
  # positive weight, in all and away from the cutoff
  PL, QL, nL, _ = _rddpowersums(X, Y, 1:Nl, hl, k, p)
  PR, QR, nR, nR0 = _rddpowersums(X, Y, Nl+1:Nh, hr, k, p)
  _rddidentified(nL, nR, nR0, p, restricted) || return nan
  S, b = _rddnormal(PL, PR, QL, QR, p, restricted)
  _rddrcond(S) >= eps() || return nan
  F = LinearAlgebra.lu!(S; check = false)
  LinearAlgebra.issuccess(F) || return nan
  β = LinearAlgebra.ldiv!(F, b)
  # the positions of the two slopes, and the bandwidths that undo the scaling of
  # the coefficients of order one; the restricted model has hl = hr and uses hl
  iL, iR = 2, restricted ? 3 : p + 3
  gL, gR = hl, restricted ? hl : hr
  est = _rddsides(β[iL] / gL, β[iR] / gR)
  coef = if restricted
    c = β[s == 0 ? 1 : s == 1 ? 3 : s + 2] / hl^s
    _rddsides(c, c)
  else
    _rddsides(β[s + 1] / hl^s, β[p + 2 + s] / hr^s)
  end
  variance || return (est = est, coef = coef, var = _RDDNANS)
  var = if vce === :jackknife
    _rddjackknife(F, X, N, Nl, hl, hr, p, k, restricted, ties, iL, iR, gL, gR)
  else
    _rddplugin(est.left, est.right, N, hl, hr, p, k, restricted)
  end
  return (est = est, coef = coef, var = _rddnonneg(var))
end

# The normal equations S β = b of the fit, with S = ∑ w z zᵀ and b = ∑ w z F̂ for
# the regressors z of each observation, from the power sums of the two sides
function _rddnormal(PL, PR, QL, QR, p, restricted)
  d = restricted ? p + 2 : 2p + 2
  S, b = zeros(d, d), zeros(d)
  if restricted
    for (P, Q, left) in ((PL, QL, true), (PR, QR, false)), j in 0:p
      a = _rddslot(j, left)
      b[a] += Q[j + 1]
      for l in 0:p
        S[a, _rddslot(l, left)] += P[j + l + 1]
      end
    end
  else
    for j in 0:p
      b[j + 1], b[p + 2 + j] = QL[j + 1], QR[j + 1]
      for l in 0:p
        S[j + 1, l + 1] = PL[j + l + 1]
        S[p + 2 + j, p + 2 + l] = PR[j + l + 1]
      end
    end
  end
  return S, b
end

# Whether the design can identify the coefficients, from the numbers of distinct
# values with positive weight on the left (nL) and on the right (nR, and nR0 of them
# away from the cutoff), since the rank of S is at most the number of distinct
# regressor vectors. Unrestricted, each side's polynomial of order p needs p + 1
# distinct values. Restricted, the p + 2 coefficients need p + 2 distinct values
# in all, and each slope a value on its side away from the cutoff.
_rddidentified(nL, nR, nR0, p, restricted) =
  restricted ? nL >= 1 && nR0 >= 1 && nL + nR >= p + 2 : nL >= p + 1 && nR >= p + 1

# The reciprocal condition number, in the 1-norm, of the normal matrix scaled to a
# unit diagonal. A fit needs it at least eps, the tolerance of R's solve. Unscaled,
# the test would depend on the units of the coefficients: the columns of high powers
# are small where the data fill only part of the window, and the unscaled number
# falls below 1e-17 for ordinary pilot fits of order p + 2 = 4, whose solutions are
# accurate. Scaled, it stayed above 1e-13 for ordinary fits up to order 8, and
# above 7e-16 for the pilot fits of order 9 (p = 7).
function _rddrcond(S)
  d = size(S, 1)
  D = [S[i, i] > 0 ? 1 / sqrt(S[i, i]) : 0.0 for i in 1:d]
  all(>(0), D) || return 0.0
  A = D .* S .* D'
  anorm = LinearAlgebra.opnorm(A, 1)
  F = LinearAlgebra.lu!(A; check = false)
  LinearAlgebra.issuccess(F) || return 0.0
  return LinearAlgebra.LAPACK.gecon!('1', F.factors, anorm)
end

# The power sums ∑ w uᵐ, m = 0, …, 2p, and ∑ w uʲ F̂, j = 0, …, p, over the
# observations r of one side, with u = x/h and w = k(u)/h. The sums are compensated
# (Knuth's TwoSum), so that they are nearly exact: the normal matrix of a fit of
# order p + 2 is ill-conditioned, and its solution amplifies the rounding of its
# entries.
function _rddpowersums(X, Y, r, h, k, p)
  P, Pc = zeros(2p + 1), zeros(2p + 1)
  Q, Qc = zeros(p + 1), zeros(p + 1)
  n = n0 = 0                    # distinct values with positive weight, and those not at 0
  last = 0.0
  @inbounds for i in r
    u = X[i] / h
    t = k(u) / h
    if t > 0 && (n == 0 || X[i] != last)
      n += 1
      n0 += X[i] != 0
      last = X[i]
    end
    y = Y[i]
    for m in 1:2p+1
      P[m], e = _twosum(P[m], t)
      Pc[m] += e
      if m <= p + 1
        Q[m], e = _twosum(Q[m], t * y)
        Qc[m] += e
      end
      t *= u
    end
  end
  return P .+ Pc, Q .+ Qc, n, n0
end

@inline function _twosum(a, b)
  s = a + b
  z = s - a
  return s, (a - (s - z)) + (b - z)
end

# The jackknife variance of the two slopes (Cattaneo, Jansson and Ma 2020): with Lᵢ = ∑_{l > i} w_l z_l / (N - 1), the sum over the later observations
# of the window, tied observations taking the Lᵢ of the first of their group, the
# covariance matrix of the coefficients is S⁻¹ (∑ᵢ Lᵢ Lᵢᵀ) S⁻¹. Its entries for the
# slopes are ∑ᵢ (aᵀLᵢ)(cᵀLᵢ) with a and c the columns of S⁻¹ of the two slopes, and
# aᵀLᵢ is a reverse cumulative sum of the scalars w_l aᵀz_l / (N - 1).
function _rddjackknife(F, X, N, Nl, hl, hr, p, k, restricted, ties, iL, iR, gL, gR)
  d = size(F, 1)
  aL = zeros(d); aL[iL] = 1.0
  aR = zeros(d); aR[iR] = 1.0
  LinearAlgebra.ldiv!(F, aL)
  LinearAlgebra.ldiv!(F, aR)
  sL = sR = 0.0                 # ∑_{l > i} w_l aᵀz_l for the two slopes
  vLL = vRR = vLR = 0.0
  pending = 0                   # observations waiting for the first of their group
  @inbounds for i in length(X):-1:1
    left = i <= Nl
    h = left ? hl : hr
    u = X[i] / h
    w = k(u) / h
    zL, zR = _rddrow(aL, aR, u, p, restricted, left)
    pending += 1
    if ties && i > 1 && X[i - 1] == X[i]
      sL += w * zL
      sR += w * zR
    else
      λL, λR = sL / (N - 1), sR / (N - 1)
      vLL += pending * λL^2
      vRR += pending * λR^2
      vLR += pending * λL * λR
      pending = 0
      sL += w * zL
      sR += w * zR
    end
  end
  vLL /= gL^2
  vRR /= gR^2
  vLR /= gL * gR
  return (left = vLL, right = vRR, diff = vLL + vRR - 2vLR, sum = vLL + vRR + 2vLR)
end

# aᵀz and cᵀz for the regressors z at u of an observation on the given side
@inline function _rddrow(a, c, u, p, restricted, left)
  if restricted
    j1 = left ? 2 : 3
    za, zc = a[1] + a[j1] * u, c[1] + c[j1] * u
    t = u
    for j in 2:p
      t *= u
      za += a[j + 2] * t
      zc += c[j + 2] * t
    end
  else
    o = left ? 0 : p + 1
    za, zc = a[o + 1], c[o + 1]
    t = 1.0
    for j in 1:p
      t *= u
      za += a[o + j + 1] * t
      zc += c[o + j + 1] * t
    end
  end
  return za, zc
end

# The moment matrix S₊ (or vector) of the right side, in the coordinates (1, u, …, uᵖ),
# placed in the restricted coordinates (1, u·1{left}, u·1{right}, u², …, uᵖ): the
# entries of order j go to _rddslot(j, false), and the left slope stays zero
function _rddright(M::AbstractMatrix)
  q = size(M, 1)
  R = zeros(q + 1, q + 1)
  for j in 1:q, l in 1:q
    R[_rddslot(j - 1, false), _rddslot(l - 1, false)] = M[j, l]
  end
  return R
end
function _rddright(v::AbstractVector)
  r = zeros(length(v) + 1)
  for j in eachindex(v)
    r[_rddslot(j - 1, false)] = v[j]
  end
  return r
end

# The reflection R of the restricted coordinates, which maps the regressors of the
# right side at t to those of the left side at -t: it keeps the intercept, sends
# the right slope to minus the left slope and the left slope to minus the right
# slope, and multiplies the coefficient of uʲ by (-1)ʲ. For a symmetric kernel the
# moments of the left side are R M R from those M of the right side. R is a signed
# permutation, so that R M R and R v only move entries and change their signs.
_rddflipindex(a) = a == 2 ? 3 : a == 3 ? 2 : a
_rddflipsign(a) = a == 1 ? 1.0 : a <= 3 ? -1.0 : iseven(a) ? 1.0 : -1.0   # a = j + 2 for uʲ
_rddreflect(M::AbstractMatrix) =
  [_rddflipsign(a) * _rddflipsign(c) * M[_rddflipindex(a), _rddflipindex(c)] for a in axes(M, 1), c in axes(M, 2)]
_rddreflect(v::AbstractVector) = [_rddflipsign(a) * v[_rddflipindex(a)] for a in eachindex(v)]

# The plug-in variances of the densities fL and fR (Cattaneo, Jansson and Ma 2018,
# 2020): the variance of the slope of a fit of order p is
# f e₂ᵀ S⁻¹ G S⁻¹ e₂ / (N h) with S and G the matrices of the kernel on the side,
# [0, 1] for the right. In the restricted model the two sides share coefficients,
# and the sandwich has the bread fL S₋ + fR S₊ and the meat fL³ G₋ + fR³ G₊, the
# left matrices reflected from the right ones.
function _rddplugin(fL, fR, N, hl, hr, p, k, restricted)
  Sp, Gp = _rdd_s(k, p, 0, 1), _cjm_g(k, p, 0, 1)
  if !restricted
    Si = inv(Sp)
    v = (Si * Gp * Si)[2, 2]
    l, r = fL * v / (N * hl), fR * v / (N * hr)
    return (left = l, right = r, diff = l + r, sum = l + r)
  end
  S₊, G₊ = _rddright(Sp), _rddright(Gp)
  A = fL * _rddreflect(S₊) + fR * S₊
  Ai = inv(A)
  V = Ai * (fL^3 * _rddreflect(G₊) + fR^3 * G₊) * Ai
  nh = N * hl
  return (left = V[2, 2] / nh, right = V[3, 3] / nh, diff = (V[2, 2] + V[3, 3] - 2V[2, 3]) / nh,
          sum = (V[2, 2] + V[3, 3] + 2V[2, 3]) / nh)
end

# S[i, j] = ∫_a^b x^(i+j-2) k(x) dx and C[i] = ∫_a^b x^(i+m-1) k(x) dx, i, j = 1, …, p + 1
_rdd_s(k, p, a, b) = [_kintegral(k, x -> x^(i + j - 2) * k(x), a, b) for i in 1:p+1, j in 1:p+1]
_rdd_c(k, m, p, a, b) = [_kintegral(k, x -> x^(i + m - 1) * k(x), a, b) for i in 1:p+1]

# probabilists' Hermite polynomials
_hermite(x, p) = p == 0 ? 1.0 : p == 1 ? x : x * _hermite(x, p - 1) - (p - 1) * _hermite(x, p - 2)

# rdbwdensity's constants of the normal-reference pilot bandwidths, for the uniform kernel
const _RDD_CB = (25884.444444494150957, 3430865.4551236177795, 845007948.04262602329, 330631733667.03808594,
                 187774809656037.3125, 145729502641999264, 146013502974449876992)
const _RDD_CC = (4.8000000000000246914, 548.57142857155463389, 100800.00000020420703, 29558225.458100609481,
                 12896196859.612621307, 7890871468221.609375, 6467911284037581)

## The bandwidths that minimize the asymptotic mean squared errors
#
# Cattaneo, Jansson and Ma (2018, 2020): the density estimate of order p on each
# side has the bias hᵖ B and the variance V/(N h), so that
# h = (V / (2p B² N))^(1/(2p+1)) minimizes its mean squared error, and likewise
# for the difference and the sum of the two sides. The derivative of order p + 1 in
# B comes from fits of order p + 2 and V from fits of order p, at normal-reference
# pilot bandwidths.

# The distance from the cutoff to the m-th nearest observation on the left and on
# the right of the sorted X, the first Nl of them negative, or to the m-th nearest
# distinct value; on a side with fewer, to the farthest one. Values are distinct by
# ==, so that -0.0 and 0.0 are one value, as for R's unique() and the ties of the
# empirical distribution function
function _rddreach(X, Nl, m, distinct::Bool)
  N = length(X)
  if !distinct
    return -X[Nl - min(m, Nl) + 1], X[Nl + min(m, N - Nl)]
  end
  i, c = Nl, 1                  # the c-th distinct value from the cutoff on the left is X[i]
  while i > 1 && c < m
    i -= 1
    c += X[i] != X[i + 1]
  end
  j, c = Nl + 1, 1
  while j < N && c < m
    j += 1
    c += X[j] != X[j - 1]
  end
  return -X[i], X[j]
end

function _rddbandwidths(X, Y, N, Nl, p, k, restricted, vce, ties, regularize, minobs, minunique)
  # normal-reference pilot bandwidths, for the derivative of order p + 1 (bn) and for
  # the density (cn), with the constants of the uniform kernel
  μ = sum(X) / N
  σ = _sd(X)
  z = μ / σ
  φ = exp(-z^2 / 2) / sqrt(2π)
  bn = ((2p + 1) / 4 * _RDD_CB[p] / (_hermite(z, p + 2)^2 * φ) / N)^(1 / (2p + 5)) * σ
  cn = (1 / (2p) * _RDD_CC[p] / (_hermite(z, p)^2 * φ) / N)^(1 / (2p + 1)) * σ
  far = max(-X[1], X[N])
  if regularize
    bn, cn = min(bn, far), min(cn, far)
    for (on, distinct) in ((minobs > 0, false), (minunique > 0, true))
      on || continue
      bn = max(bn, _rddreach(X, Nl, p + 23, distinct)...)
      cn = max(cn, _rddreach(X, Nl, p + 21, distinct)...)
    end
  end
  # the pilot fits on the observations within bn and within cn of the cutoff
  function pilot(h, q, s, withvar)
    lo, hi = searchsortedfirst(X, -h), searchsortedlast(X, h)
    nl = max(min(hi, Nl) - lo + 1, 0)
    return _rddfit(view(X, lo:hi), view(Y, lo:hi), N, nl, h, h, q, s, k, restricted, vce, ties; variance = withvar)
  end
  fb = pilot(bn, p + 2, p + 1, false)
  fc = pilot(cn, p, 1, true)
  variance = map(v -> N * cn * v, fc.var)
  # the bias constants: the leading bias of the slope of a fit of order p is hᵖ times
  # the coefficient β of order p + 1 of the distribution function, estimated by the
  # first pilot fit, times the entry of S⁻¹C for the slope; reflecting the left side
  # onto [0, 1] gives its sign (-1)ᵖ, and in the restricted model the two sides are
  # solved together
  βL, βR = fb.coef.left, fb.coef.right
  fL, fR = fc.est.left, fc.est.right
  Sp, Cp = _rdd_s(k, p, 0, 1), _rdd_c(k, p + 1, p, 0, 1)
  BL, BR = if restricted
    S₊, C₊ = _rddright(Sp), _rddright(Cp)
    t = (fR * S₊ + fL * _rddreflect(S₊)) \ (βL * (fR * C₊ + (-1)^(p + 1) * fL * _rddreflect(C₊)))
    t[2], t[3]
  else
    a = (Sp \ Cp)[2]
    βL * a * (-1)^p, βR * a
  end
  biassq = (left = BL^2, right = BR^2, diff = (BR - BL)^2, sum = (BR + BL)^2)
  hopt(V, B2) = isnan(V) || V < 0 ? 0.0 : (h = (1 / (2p) * V / B2 / N)^(1 / (2p + 1)); isnan(h) ? 0.0 : h)
  hl, hr = hopt(variance.left, biassq.left), hopt(variance.right, biassq.right)
  hd, hs = hopt(variance.diff, biassq.diff), hopt(variance.sum, biassq.sum)
  if regularize
    hl, hr, hd, hs = min(hl, -X[1]), min(hr, X[N]), min(hd, far), min(hs, far)
    for (m, distinct) in ((minobs, false), (minunique, true))
      m > 0 || continue
      dl, dr = _rddreach(X, Nl, m, distinct)
      hl, hr, hd, hs = max(hl, dl), max(hr, dr), max(hd, dl, dr), max(hs, dl, dr)
    end
  end
  return (left = hl, right = hr, diff = hd, sum = hs, variance = variance, biassq = biassq, pilots = (bn, cn))
end

# rddensity's binomial tests: the numbers of observations on the two sides in windows
# around the cutoff that hold at least n observations on each side and grow in nw
# steps up to the bandwidths, with the exact two-sided p-values for probability 1/2
function _rddbinomial(X, N, Nl, hl, hr, n, nw)
  XL = sort!(-X[1:Nl])
  XR = X[Nl+1:N]
  w1 = max(XL[min(n, Nl)], XR[min(n, N - Nl)])
  wl, wr = [w1], [w1]
  if nw > 1 && w1 < hl && w1 < hr
    wl = [w1 * nw > hl ? w1 + (j - 1) * (hl - w1) / (nw - 1) : j * w1 for j in 1:nw]
    wr = [w1 * nw > hr ? w1 + (j - 1) * (hr - w1) / (nw - 1) : j * w1 for j in 1:nw]
  end
  nl = [count(<=(w), XL) for w in wl]
  nr = [count(<=(w), XR) for w in wr]
  return (left = wl, right = wr, nleft = nl, nright = nr, pvalue = [_binomtest(a, a + b) for (a, b) in zip(nl, nr)])
end

# The exact two-sided binomial test of probability 1/2 for x successes in n trials:
# the total probability of the outcomes at most 1 + 1e-7 times as likely as x. The
# distribution is symmetric and its probabilities increase up to the mode, so that
# these outcomes are the tail of x, i ≤ t for t = min(x, n - x) (or its mirror), and
# on the other side the mirror of i ≤ a, with a ≥ t the last outcome below n/2
# within the tolerance. The outcomes after t come within 1e-7 of x only for n beyond
# 10⁷; the probabilities of Distributions.jl are accurate there only to about n
# times the machine epsilon (3e-7 for n = 10⁸), so that a is found from the exact
# ratios P(j + 1)/P(j) = (n - j)/(j + 1) instead. When the probability of x
# underflows, as for x = 0 and n ≥ 1075, the outcomes after t whose probabilities
# underflow as well count as at most as likely, and a is the last of them, found by
# bisection; the tail of x stays i ≤ t, which changes the p-value only by the
# smallest subnormal number.
function _binomtest(x, n)
  (n == 0 || 2x == n) && return 1.0
  D = Distributions.Binomial(n, 0.5)
  t = min(x, n - x)
  m = (n - 1) ÷ 2               # the last outcome below n/2
  a = t
  if pdf(D, x) > 0
    r = 1.0                     # P(a)/P(t)
    while a < m
      q = r * (n - a) / (a + 1)
      q <= 1 + 1e-7 || break
      a, r = a + 1, q
    end
    # with the mode of an even n within the tolerance, so are all outcomes
    a == m && iseven(n) && r * (n - a) / (a + 1) <= 1 + 1e-7 && return 1.0
  else
    b = m + 1                   # pdf(a) = 0, and pdf(b) > 0 or b beyond m
    while b - a > 1
      j = (a + b) ÷ 2
      pdf(D, j) == 0 ? (a = j) : (b = j)
    end
  end
  # for an odd n with a = (n - 1)/2 the two tails cover all outcomes
  2a + 1 >= n && return 1.0
  return min(1.0, cdf(D, t) + cdf(D, a))
end


## Interface

nobs(m::DensityDiscontinuity) = sum(m.nobs)
bandwidth(m::DensityDiscontinuity) = m.h
StatsAPI.coef(m::DensityDiscontinuity) = [m.density[2] - m.density[1]]
stderror(m::DensityDiscontinuity) = [m.se[3]]

function StatsAPI.confint(m::DensityDiscontinuity; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  d = coef(m)[1]
  return [d - z * m.se[3] d + z * m.se[3]]
end

"""
    coeftable(m::DensityDiscontinuity; level = 0.95)

The estimates of the density on the left and on the right of the cutoff and of the
jump, with their standard errors and intervals, a named tuple of columns `side`,
`estimate`, `se`, `lower` and `upper`.
"""
function StatsAPI.coeftable(m::DensityDiscontinuity; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  z = Distributions.quantile(Distributions.Normal(), (1 + level) / 2)
  est = [m.density[1], m.density[2], m.density[2] - m.density[1]]
  se = collect(m.se)
  return (side = ["left", "right", "jump"], estimate = est, se = se, lower = est .- z .* se, upper = est .+ z .* se)
end

function Base.show(io::IO, m::DensityDiscontinuity)
  print(io, "DensityDiscontinuity(cutoff ", m.cutoff, ", ", nobs(m), " observations, p-value ", _fmt(m.pvalue), ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::DensityDiscontinuity)
  println(io, "Manipulation test at cutoff ", _fmt(m.cutoff), ": local polynomial densities of degree ", m.degree,
          m.restricted ? ", restricted model" : "")
  println(io, "  kernel: ", m.kernel, ", variance: ", m.vce, m.masspoints ? ", ties corrected" : "")
  m.selection.rule === nothing || println(io, "  bandwidth: ", m.selection.rule)
  println(io, "  ", rpad("", 22), rpad("left", 12), "right")
  println(io, "  ", rpad("observations", 22), rpad(m.nobs[1], 12), m.nobs[2])
  println(io, "  ", rpad("in the window", 22), rpad(m.nh[1], 12), m.nh[2])
  println(io, "  ", rpad("bandwidth", 22), rpad(_fmt(m.h[1]), 12), _fmt(m.h[2]))
  println(io, "  ", rpad("density", 22), rpad(_fmt(m.density[1]), 12), _fmt(m.density[2]))
  println(io, "  test of continuity: T = ", _fmt(m.statistic), ", p-value ", _fmt(m.pvalue))
end
