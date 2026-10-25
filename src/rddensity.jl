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

selectbandwidth(rule::RDDensityOptimal, d::SmoothingData, k::AbstractKernel; kwargs...) =
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

# _rddfit(X, Y, N, Nl, hl, hr, p, s, k, restricted, vce, ties) goes here (see the specification).

# S[i, j] = ∫_a^b x^(i+j-2) k(x) dx and C[i] = ∫_a^b x^(i+m-1) k(x) dx, i, j = 1, …, p + 1
_rddS(k, p, a, b) = [_kintegral(k, x -> x^(i + j - 2) * k(x), a, b) for i in 1:p+1, j in 1:p+1]
_rddC(k, m, p, a, b) = [_kintegral(k, x -> x^(i + m - 1) * k(x), a, b) for i in 1:p+1]

# probabilists' Hermite polynomials
_hermite(x, p) = p == 0 ? 1.0 : p == 1 ? x : x * _hermite(x, p - 1) - (p - 1) * _hermite(x, p - 2)

# rdbwdensity's constants of the normal-reference pilot bandwidths, for the uniform kernel
const _RDD_CB = (25884.444444494150957, 3430865.4551236177795, 845007948.04262602329, 330631733667.03808594,
                 187774809656037.3125, 145729502641999264, 146013502974449876992)
const _RDD_CC = (4.8000000000000246914, 548.57142857155463389, 100800.00000020420703, 29558225.458100609481,
                 12896196859.612621307, 7890871468221.609375, 6467911284037581)

# _rddbandwidths(X, Y, N, Nl, p, k, restricted, vce, ties, regularize, minobs, minunique) goes here.

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

# _binomtest(x, n) goes here.


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
