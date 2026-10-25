## Honest intervals for regression discontinuity (Armstrong and Kolesár 2018, 2020)

"""
    RDHonestOptimal(M = :rot; criterion = :mse, level = 0.95, power = 0.8, class = :holder, T0 = 0)

The bandwidth of RDHonest for [`RegressionDiscontinuity`](@ref): the bandwidth,
common to both sides, that minimizes the worst-case mean squared error of the local
linear estimate over the class of regression functions with second derivative
bounded by `M` (`criterion = :mse`, the default), the length of the honest interval
(`:flci`), or the excess length of the one-sided interval at quantile `power`
(`:oci`). The standard errors come from preliminary variances that are constant on
each side, estimated by local linear fits with the bandwidth of Imbens and
Kalyanaraman (2012), as in RDHonest. `M` is a number, a pair `(M_Y, M_T)` for a
fuzzy design, or `:rot` for the rule of thumb of RDHonest; `class` is `:holder` or
`:taylor`, and for a fuzzy design `T0` is the initial value of the effect that the
worst-case bias of the rule uses. The pilot bandwidth is `b = h`.
"""
struct RDHonestOptimal <: BandwidthSelector
  M::Union{Float64,NTuple{2,Float64},Symbol}
  criterion::Symbol
  level::Float64
  power::Float64
  class::Symbol
  T0::Float64

  function RDHonestOptimal(M::Union{Real,Symbol,Tuple{Real,Real}} = :rot; criterion::Symbol = :mse,
                           level::Real = 0.95, power::Real = 0.8, class::Symbol = :holder, T0::Real = 0)
    h = Honest(M; class)   # checks M and class
    criterion in (:mse, :flci, :oci) || throw(ArgumentError("criterion must be :mse, :flci or :oci, got $(repr(criterion))"))
    0 < level < 1 && 0 < power < 1 || throw(ArgumentError("level and power must be between 0 and 1"))
    class === :default && throw(ArgumentError("class must be :holder or :taylor"))
    isfinite(T0) || throw(ArgumentError("T0 must be finite, got $T0"))
    new(h.M, criterion, Float64(level), Float64(power), class, Float64(T0))
  end
end

function Base.show(io::IO, r::RDHonestOptimal)
  print(io, "RDHonestOptimal(", repr(r.M))
  _showkeywords(io, r, RDHonestOptimal(r.M), "; ")
  print(io, ")")
end

selectbandwidth(rule::RDHonestOptimal, d::SmoothingData, k::AbstractKernel; kwargs...) =
  throw(ArgumentError("$rule chooses bandwidths for regression discontinuity; fit a RegressionDiscontinuity"))

"""
    worstcasebias(m::RegressionDiscontinuity, M; class = :holder)

The worst-case bias of the estimate of a [`RegressionDiscontinuity`](@ref) fit, as
in [`Honest`](@ref): over the Hölder class of functions whose second derivative is
bounded by `M` on each side of the cutoff, `M/2 |∑ᵢ ℓᵢ (xᵢ - c)²|` for the weights `ℓᵢ`
of a local linear estimate, which have the same sign on each side; or over the
Taylor class, `M / (p + 1)! ∑ᵢ |ℓᵢ| |xᵢ - c|^(p+1)`. For a fuzzy design, `M = (M_Y, M_T)`
bounds the outcome and the treatment, and the bias is that of `(M_Y + M_T |τ̂|) / |f̂|`
for the estimate `τ̂` and the jump `f̂` in the treatment. `M = :rot` takes the rule of
thumb of RDHonest.
"""
function worstcasebias(m::RegressionDiscontinuity, M; class::Symbol = :holder)
  Honest(M; class)   # checks M and class
  m.ncovariates == 0 || throw(ArgumentError("honest intervals are not implemented with covariates"))
  m.derivative == 0 || throw(ArgumentError("honest intervals are implemented for the jump, derivative = 0"))
  class === :taylor || m.degree == 1 ||
    throw(ArgumentError("the bound over the Hölder class is for local linear fits, degree = 1; use class = :taylor"))
  Mv = M === :rot ? _rdmrot(m.data, m.cutoff, m.fuzzy) : M
  if m.fuzzy
    Mv isa Tuple || throw(ArgumentError("a fuzzy design needs bounds M = (M_Y, M_T) for the outcome and the treatment"))
    Meff = (Mv[1] + Mv[2] * abs(m.estimate)) / abs(m.firststage.estimate::Float64)
  else
    Mv isa Tuple && throw(ArgumentError("a sharp design needs one bound M"))
    Meff = Mv
  end
  ws = map(j -> _rdhweights(m.data[j].data, m.kernel, m.cutoff, m.h[j], m.degree), (1, 2))
  return Meff * _rdhbiasconstant(ws, m.degree, class)
end

function _rdhonestconfint(m::RegressionDiscontinuity, level, method::Honest)
  method.biascorrected && throw(ArgumentError("honest intervals for regression discontinuity use the estimate itself"))
  class = method.class === :default ? :holder : method.class
  B = worstcasebias(m, method.M; class)
  se = m.se
  half = se == 0 ? B : _cvb(B / se, level) * se
  return [m.estimate - half m.estimate + half]
end

# The weights ℓᵢ of the local polynomial estimate of order p of the value at c on one
# side, with bandwidth h, and the distances xᵢ - c, over the observations r with
# positive weight
function _rdhweights(d, k, c, h, p)
  r = _poswindow(d, k, c, h)
  isempty(r) && return (ℓ = Float64[], t = Float64[], r = r)
  S = zeros(2p + 1)
  for i in r
    u = (d.x[i] - c) / h
    _powersums!(S, _weight(d.w, i) * k(u), u)
  end
  C, ok = _gramfactor([S[i + j + 1] for i in 0:p, j in 0:p], nunique(d, r))
  ok || return (ℓ = fill(NaN, length(r)), t = d.x[r] .- c, r = r)
  g = C \ [j == 0 ? 1.0 : 0.0 for j in 0:p]
  t = d.x[r] .- c
  return (ℓ = [_weight(d.w, i) * k((d.x[i] - c) / h) * evalpoly((d.x[i] - c) / h, g) for i in r], t = t, r = r)
end

# the worst-case bias of the jump per unit of M, from the weights on the two sides
function _rdhbiasconstant(ws, p, class)
  if class === :holder
    return abs(sum(w -> sum(w.ℓ .* w.t .^ 2), ws)) / 2
  else
    return sum(w -> sum(abs.(w.ℓ) .* abs.(w.t) .^ (p + 1)), ws) / factorial(p + 1)
  end
end

# The rule of thumb for M of Armstrong and Kolesár (2020): on each side, the largest
# absolute second derivative of a global quartic over the observations of that side,
# and the larger of the two; for a fuzzy design, for the outcome and the treatment
function _rdmrot(sides, c, fuzzy)
  M(col) = maximum(s -> _mrot(s.data.x .- c, s.D[:, col], s.data.w), sides)
  return fuzzy ? (M(1), M(2)) : M(1)
end

# The bound on one side: fit y ≈ r₀ + r₁x + ⋯ + r₄x⁴ by (weighted) least squares, and
# return the largest |q(x)| of the second derivative q(x) = 2r₂ + 6r₃x + 12r₄x² over
# the range [x₁, xₙ] of the sorted x. q is a quadratic, so its largest absolute value
# lies at an end of the range or at its vertex -r₃/(4r₄).
function _mrot(x, y, w)
  _hasdistinct(x, 4) ||
    throw(ArgumentError("the rule of thumb for M fits a quartic on each side and needs at least 4 distinct values of x there; pass M"))
  n = length(x)
  A = Matrix{Float64}(undef, n, 5)
  z = Vector{Float64}(undef, n)
  @inbounds for i in 1:n
    s = w === nothing ? 1.0 : sqrt(w[i])
    for j in 0:4
      A[i, j + 1] = s * x[i]^j
    end
    z[i] = s * y[i]
  end
  r = LinearAlgebra.qr!(A) \ z
  q(u) = 2r[3] + 6u * r[4] + 12u^2 * r[5]
  lo, hi = x[1], x[end]
  M = max(abs(q(lo)), abs(q(hi)))
  if abs(r[5]) > 1e-10
    v = -r[4] / (4r[5])
    lo < v < hi && (M = max(M, abs(q(v))))
  end
  return M
end

# whether the sorted x takes at least k distinct values
function _hasdistinct(x, k)
  m = min(length(x), 1)
  for i in 2:length(x)
    m += x[i] != x[i - 1]
    m >= k && return true
  end
  return m >= k
end


## The bandwidth of RDHonest

function _rdhselect(rule::RDHonestOptimal, sides, c, k, fuzzy, dZ)
  dZ == 0 || throw(ArgumentError("RDHonestOptimal is not implemented with covariates"))
  M = rule.M === :rot ? _rdmrot(sides, c, fuzzy) : rule.M
  fuzzy == (M isa Tuple) || throw(ArgumentError(fuzzy ? "a fuzzy design needs bounds M = (M_Y, M_T)" : "a sharp design needs one bound M"))
  Mopt = fuzzy ? M[1] + M[2] * abs(rule.T0) : M
  # the distances x - c, sorted, on the left and on the right
  t = map(s -> s.data.x .- c, sides)
  σ2 = _rdhprelimvar(sides, c, fuzzy)
  objective(h) = _rdhcriterion(rule, map(j -> _rdhweights(sides[j].data, k, c, h, 1), (1, 2)), sides, σ2, Mopt, fuzzy)
  ul, ur = unique(abs.(t[1])), unique(t[2])
  hmin = max(sort(ur)[min(2, end)], sort(ul)[min(2, end)])
  hmax = max(maximum(abs, t[1]), maximum(t[2]))
  h = if k isa UniformKernel
    supp = sort(unique(vcat(abs.(t[1]), t[2])))
    _rdhgss(objective, supp[supp .>= hmin])
  else
    _fmin(objective, hmin, hmax; tol = eps()^0.75)
  end
  return (h, h), (M = M, criterion = rule.criterion)
end

# The criterion of the rule for the weights ws of the estimate at a bandwidth, with
# the preliminary variances σ2 and the bound Mopt; where a side has no fit, as
# RDHonest, a huge bias and standard error
function _rdhcriterion(rule, ws, sides, σ2, Mopt, fuzzy)
  if any(w -> isempty(w.ℓ) || any(isnan, w.ℓ), ws)
    bias = se = sqrt(floatmax() / 10)
  else
    V = sum(j -> sum(abs2, ws[j].ℓ) .* σ2[j], 1:2)
    if fuzzy
      τ = _rdhjump(ws, sides, 1) / _rdhjump(ws, sides, 2)
      se = sqrt(max(V[1] - 2τ * V[2] + τ^2 * V[3], 0.0))
    else
      se = sqrt(V)
    end
    bias = Mopt * _rdhbiasconstant(ws, 1, rule.class)
  end
  rule.criterion === :mse && return bias^2 + se^2
  rule.criterion === :flci && return 2 * _cvb(bias / se, rule.level) * se
  return 2bias + se * (Distributions.quantile(Distributions.Normal(), rule.level) +
                       Distributions.quantile(Distributions.Normal(), rule.power))
end

# the jump in column col of D: the right estimate minus the left one
_rdhjump(ws, sides, col) = sum(ws[2].ℓ .* sides[2].D[ws[2].r, col]) - sum(ws[1].ℓ .* sides[1].D[ws[1].r, col])

# The minimizer of f over the sorted candidates xs, for a criterion f that is
# unimodal and piecewise constant, such as the criterion of the uniform kernel, which
# changes only where the bandwidth crosses a distance |xᵢ - c|. A golden-section
# search over the indices narrows the bracket [a, b] until it holds at most 101
# candidates, keeping at each step the part on the side of the smaller of the two
# probes; the smallest value in the final bracket wins, the first one on ties (in the
# order of argmin, where NaN counts as the smallest and -0.0 below 0.0). The
# values at the previous probes are kept, since the golden ratio makes one of them
# a probe of the next bracket (up to rounding) and f is deterministic.
function _rdhgss(f, xs)
  φ = (1 + sqrt(5)) / 2
  probes = (0, NaN, 0, NaN)   # the previous probes and their values
  a, b = 1, length(xs)
  while b - a > 100
    lo = round(Int, b - (b - a) / φ)
    hi = round(Int, a + (b - a) / φ)
    flo = _probevalue(f, xs, lo, probes)
    fhi = _probevalue(f, xs, hi, probes)
    probes = (lo, flo, hi, fhi)
    if flo < fhi
      b = hi
    else
      a = lo
    end
  end
  return xs[_bracketargmin(f, xs, a:b, probes)]
end

# the first index i in r that minimizes f(xs[i]), in the order of argmin
_bracketargmin(f, xs, r, probes) = argmin(i -> _probevalue(f, xs, i, probes), r)

# f(xs[i]), or its value at a previous probe
function _probevalue(f, xs, i, probes)
  i == probes[1] && return probes[2]
  i == probes[3] && return probes[4]
  return f(xs[i])
end

# The preliminary variances: on each side, the means of the squared
# residuals (and their products, for the outcome and the treatment of a fuzzy design)
# of local linear fits with the triangular kernel and the bandwidth of Imbens and
# Kalyanaraman, at least the distance of the third distinct value and fourth
# observation on each side
function _rdhprelimvar(sides, c, fuzzy)
  hmin = _rdhminbw(sides, c)
  h = max(_rdikbw(sides, c, hmin), hmin)
  map(sides) do s
    d = s.data
    r = _poswindow(d, TriangularKernel(), c, h)
    X = [((d.x[i] - c))^j for i in r, j in 0:1]
    sw = sqrt.([_weight(d.w, i) * TriangularKernel()((d.x[i] - c) / h) for i in r])
    cols = fuzzy ? (1:2) : (1:1)
    res = [s.D[i, l] for i in r, l in cols]
    for l in cols
      β = LinearAlgebra.qr(sw .* X) \ (sw .* res[:, l])
      res[:, l] .-= X * β
    end
    fuzzy ? [sum(abs2, res[:, 1]), sum(res[:, 1] .* res[:, 2]), sum(abs2, res[:, 2])] ./ length(r) :
            sum(abs2, res[:, 1]) / length(r)
  end
end

function _rdhminbw(sides, c)
  tl, tr = abs.(sides[1].data.x .- c), sides[2].data.x .- c
  ul, ur = sort(unique(tl)), sort(unique(tr))
  return max(ur[min(3, end)], ul[min(3, end)], sort(tr)[min(4, end)], sort(tl)[min(4, end)])
end

# The bandwidth of Imbens and Kalyanaraman (2012, Section 6.2) for the local linear
# estimate of the jump with the triangular kernel, with their constants (C_K = 480^(1/5)
# ≈ 3.4375 for the triangular kernel, 7200^(1/7) ≈ 3.56 for the pilot bandwidths, and
# the regularization 2160σ²/(n h⁴)), in the variant whose results reproduce RDHonest:
# 1. with t = x - c and h₀ = 1.84 sd(t) / N^(1/5), the variance on each side is that of
#    the outcomes with |t| ≤ max(h₀, hmin) about their mean (weighted, with weights);
# 2. the density at c is the share of the observations with |t| ≤ h₀, over 2h₀;
# 3. the third derivative is that of a global cubic with a jump at c, fitted to all data;
# 4. the pilot bandwidths of each side give local quadratic fits, whose second
#    derivatives and the regularization terms give the bandwidth.
# The fits of steps 3 and 4 are unweighted.
function _rdikbw(sides, c, hmin)
  dl, dr = sides[1].data, sides[2].data
  yl, yr = view(sides[1].D, :, 1), view(sides[2].D, :, 1)
  nl, nr = length(dl.x), length(dr.x)
  N = nl + nr
  t = vcat(dl.x, dr.x)
  t .-= c
  s = _sd(t)
  h0 = 1.84 * s / N^(1 / 5)
  h1 = max(h0, hmin)
  # x is sorted, so the observations within h₁ of c are the last ones on the left and
  # the first ones on the right
  vl = _ikvariance(yl, dl.w, _firsttrue(i -> abs(dl.x[i] - c) <= h1, nl):nl)
  vr = _ikvariance(yr, dr.w, 1:(_firsttrue(i -> abs(dr.x[i] - c) > h1, nr) - 1))
  f0 = count(u -> abs(u) <= h0, t) / (2N * h0)
  # the global cubic with a jump at c: columns 1, 1{t ≥ 0}, t, t², t³
  A = Matrix{Float64}(undef, N, 5)
  z = Vector{Float64}(undef, N)
  @inbounds for i in 1:N
    u = t[i]
    A[i, 1] = 1.0
    A[i, 2] = u >= 0 ? 1.0 : 0.0
    A[i, 3] = u
    A[i, 4] = u^2
    A[i, 5] = u^3
    z[i] = i <= nl ? yl[i] : yr[i - nl]
  end
  m3 = 6 * (LinearAlgebra.qr!(A) \ z)[5]
  # the pilot bandwidths, and the local quadratic fits within them
  k7 = 7200^(1 / 7)
  h2l = k7 * (vl / (f0 * m3^2))^(1 / 7) * nl^(-1 / 7)
  h2r = k7 * (vr / (f0 * m3^2))^(1 / 7) * nr^(-1 / 7)
  wl = _firsttrue(i -> dl.x[i] - c >= -h2l, nl):nl
  wr = 1:(_firsttrue(i -> dr.x[i] - c > h2r, nr) - 1)
  m2l = _ikcurvature(dl.x, yl, c, wl)
  m2r = _ikcurvature(dr.x, yr, c, wr)
  rl = 2160 * vl / (length(wl) * h2l^4)
  rr = 2160 * vr / (length(wr) * h2r^4)
  return 480^(1 / 5) * ((vr + vl) / (f0 * N * ((m2r - m2l)^2 + rl + rr)))^(1 / 5)
end

# The variance of the outcomes y[r] about their mean, which is weighted by w[r] (or not,
# with w = nothing), with the denominator length(r) - 1
function _ikvariance(y, w, r)
  yr = view(y, r)
  m = w === nothing ? sum(yr) / length(r) : sum(view(w, r) .* yr) / sum(view(w, r))
  return sum(u -> abs2(u - m), yr) / (length(r) - 1)
end

# twice the coefficient of t² in the least squares fit of y on 1, t, t² with t = x - c,
# over the observations r
function _ikcurvature(x, y, c, r)
  A = Matrix{Float64}(undef, length(r), 3)
  z = Vector{Float64}(undef, length(r))
  @inbounds for (k, i) in enumerate(r)
    u = x[i] - c
    A[k, 1] = 1.0
    A[k, 2] = u
    A[k, 3] = u^2
    z[k] = y[i]
  end
  return 2 * (LinearAlgebra.qr!(A) \ z)[3]
end
