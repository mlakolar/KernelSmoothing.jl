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

# RDHonest's rule of thumb for M: on each side, the largest absolute second derivative
# of a global quartic over the observations of that side, and the larger of the two;
# for a fuzzy design, for the outcome and the treatment
function _rdmrot(sides, c, fuzzy)
  M(col) = maximum(s -> _mrot(s.data.x .- c, s.D[:, col], s.data.w), sides)
  return fuzzy ? (M(1), M(2)) : M(1)
end

# _mrot goes here (see the specification).


## The bandwidth of RDHonest

function _rdhselect(rule::RDHonestOptimal, sides, c, k, fuzzy, dZ)
  dZ == 0 || throw(ArgumentError("RDHonestOptimal is not implemented with covariates"))
  M = rule.M === :rot ? _rdmrot(sides, c, fuzzy) : rule.M
  fuzzy == (M isa Tuple) || throw(ArgumentError(fuzzy ? "a fuzzy design needs bounds M = (M_Y, M_T)" : "a sharp design needs one bound M"))
  Mopt = fuzzy ? M[1] + M[2] * abs(rule.T0) : M
  # RDHonest's data: x - c, sorted, on the left and on the right
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

# _rdhgss goes here (see the specification).

# RDHonest's PrelimVar: the variances on the two sides, the means of the squared
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

# _rdikbw goes here (see the specification).
