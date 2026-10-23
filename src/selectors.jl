## Data-driven bandwidth selectors

"""
    SheatherJones(; method = :ste, exact = 100)

The plug-in bandwidth of Sheather and Jones (1991) for kernel density estimates.
It estimates `ψ₄ = ∫ f''(x)² dx` with a Gaussian pilot kernel and plugs the
estimate into the bandwidth that minimizes the asymptotic mean integrated
squared error,

    h = (R(k) / (μ₂(k)² ψ̂₄(g) n))^(1/5),

where `R(k) = roughness(k)`, `μ₂(k) = moment(k, 2)` and `n` is the
[`effective_nobs`](@ref). With `method = :ste`, "solve the equation", the pilot
bandwidth `g = α₂ h^(5/7)` depends on `h` itself, with `α₂` estimated from `ψ̂₄`
and `ψ̂₆` at normal-reference pilot bandwidths, and the equation is solved to a
relative accuracy of `1e-10`. With `method = :dpi`, the direct plug-in, `g` is
the bandwidth that is optimal for estimating `ψ₄` when `ψ₆` is estimated at its
normal-reference pilot bandwidth.

The estimates are the double sums of Sheather and Jones,

    ψ̂ᵣ(g) = ∑ᵢ ∑ⱼ wᵢ wⱼ φ_g⁽ʳ⁾(xᵢ - xⱼ) / (W² - ∑ᵢ wᵢ²),   W = ∑ᵢ wᵢ,

with the diagonal included, which with equal weights is the definition of R's
`bw.SJ`. They are computed exactly for up to `exact` observations and otherwise
from the data binned linearly once, on a grid with about 100 points per smallest
pilot bandwidth, with the lagged products of the bin weights from an FFT. This
changes the estimates by about `1e-5` and the bandwidth by about `1e-6` relative,
far below its statistical error, and makes the selection take a few milliseconds
for up to `10^6` observations. Data that would need more than `2^20` grid points,
because of far outliers or heavy tails, are split at gaps across which the pilot
kernels are negligible, only the densest part of each run is binned, and the pairs
with the other observations, which are sparse, are summed exactly. R's `bw.SJ` bins differences of
pairs of observations coarsely and stops its search early, so by default it agrees
with this rule to two or three digits only. With fine bins it agrees to about
`1e-6` for `method = :dpi` and `4e-5` for `:ste`, where R rounds the constant
`(2 φ⁽⁴⁾(0) μ₂(k)² / R(k))^(1/7) = 1.35733` of the Gaussian kernel to 1.357. This rule is the recommended default for
densities without boundaries (Jones, Marron and Sheather 1996).
"""
struct SheatherJones <: BandwidthSelector
  method::Symbol
  exact::Int

  function SheatherJones(method::Symbol, exact::Integer)
    method in (:ste, :dpi) || throw(ArgumentError("method must be :ste or :dpi, got :$method"))
    new(method, exact)
  end
end

SheatherJones(; method::Symbol = :ste, exact::Integer = 100) = SheatherJones(method, exact)

function selectbandwidth(rule::SheatherJones, d::SmoothingData, k::AbstractKernel)
  nobs(d) >= 2 || throw(ArgumentError("SheatherJones needs at least two observations"))
  sd, iqr = _sdiqr(d)
  σ = _robustscale(sd, iqr / 1.349)
  _check_scale(σ, rule)
  n = effective_nobs(d)
  μ2 = _mu2centred(k)
  c1 = roughness(k) / (μ2^2 * n)
  exact = nobs(d) <= rule.exact
  # The equations are solved in units of σ, for t = h / σ, with ψ̃ᵣ(t) = σ^(r+1) ψ̂ᵣ(σ t),
  # the estimates for the standardized data, so that no power of a bandwidth
  # underflows or overflows at extreme scales of the data. The normal-reference pilot
  # bandwidths for ψ₄ and ψ₆ (Sheather and Jones 1991) in these units:
  a, b = 1.24 * n^(-1 / 7), 1.23 * n^(-1 / 9)
  # the details, in the units of the data, have one shape in every case, so that the
  # selection is inferable
  details(; pilot = NaN, alpha2 = NaN, psi4 = NaN, psi6 = NaN, iterations = 0, fallback = false) =
    (scale = σ, pilots = (σ * a, σ * b), pilot = σ * Float64(pilot), alpha2 = Float64(alpha2) * σ^(2 / 7),
     psi4 = Float64(psi4) / σ^5, psi6 = Float64(psi6) / σ^7, exact = exact, iterations = iterations,
     fallback = fallback)
  # ψ̃ᵣ(t) from data binned with at least m grid points per pilot bandwidth, from a
  # grid with 500 at the smaller pilot; the root finding evaluates ψ₄ down to about
  # a / 5, and rarely above a few times the larger pilot. The evaluator is one of two
  # closures, so that the type of its values is asserted for inference. (Local
  # functions with default arguments would be boxed.)
  Δ = σ * min(a, b) / 500
  evaluator = _psievaluator(d, Δ, exact, σ * 4 * max(a, b))
  ψm(r, t, m) = evaluator(r, σ * t, σ, m)::Float64
  ψ(r, t) = ψm(r, t, 100)
  TD = -ψ(6, b)
  TD > 0 || return _sparse(rule, d, k, "the estimate of ψ₆ is not negative", details(psi6 = -TD, fallback = true))
  if rule.method == :dpi
    g = (2 * 3 * INV_SQRT_2PI / (n * TD))^(1 / 7)   # 2 φ⁽⁴⁾(0) / (n (-ψ̂₆))
    ψ4 = ψ(4, g)
    ψ4 > 0 || return _sparse(rule, d, k, "the estimate of ψ₄ is not positive",
                             details(pilot = g, psi4 = ψ4, psi6 = -TD, fallback = true))
    return BandwidthSelection(σ * (c1 / ψ4)^(1 / 5), rule, details(pilot = g, psi4 = ψ4, psi6 = -TD))
  end
  SD = ψ(4, a)
  SD > 0 || return _sparse(rule, d, k, "the estimate of ψ₄ is not positive",
                           details(psi4 = SD, psi6 = -TD, fallback = true))
  # g(h) = α₂ h^(5/7) balances the bias and variance of ψ̂₄ at the optimal h
  α2 = (2 * 3 * INV_SQRT_2PI * μ2^2 / roughness(k))^(1 / 7) * (SD / TD)^(1 / 7)
  hmax = (243 * roughness(k) / (35 * μ2^2 * n))^(1 / 5)   # Terrell's oversmoothed bandwidth
  lower, upper = 0.1 * hmax, hmax
  # the bracket needs only the signs, which a coarser grid gives
  fm(h, m) = (c1 / max(ψm(4, α2 * h^(5 / 7), m), floatmin()))^(1 / 5) - h
  f(h) = fm(h, 50)
  tries = 0
  while f(lower) * f(upper) > 0   # widen the bracket, as R's bw.SJ does
    tries < 100 || return _sparse(rule, d, k, "its equation has no solution",
                                  details(alpha2 = α2, psi6 = -TD, fallback = true))
    isodd(tries) ? (lower /= 1.2) : (upper *= 1.2)
    tries += 1
  end
  # with binned data, ψ̂₄ jitters slightly as the grid changes with g, so the
  # bracket stops earlier
  t, iterations = _findroot(f, lower, upper; rtol = exact ? 1e-10 : 1e-8)
  # Clustered or discrete data can put the pilot bandwidth of the root far below the
  # pilots; where the grid has fewer than 100 points per pilot bandwidth there, the
  # root is solved again near it on a grid with at least 250.
  if !exact && σ * α2 * t^(5 / 7) < 100Δ
    lo, hi = t, t
    for _ in 1:10
      lo, hi = lo / 1.02, hi * 1.02
      if fm(lo, 250) * fm(hi, 250) <= 0
        t, more = _findroot(h -> fm(h, 250), lo, hi; rtol = 1e-8)
        iterations += more
        break
      end
    end
  end
  g = α2 * t^(5 / 7)
  return BandwidthSelection(σ * t, rule, details(; pilot = g, alpha2 = α2, psi4 = ψ(4, g), psi6 = -TD, iterations))
end

# For a sample too sparse for the estimates of Sheather and Jones: the normal-reference
# bandwidth, with a warning
function _sparse(rule, d, k, why, details)
  @warn "SheatherJones: $why, so the normal-reference bandwidth is used instead"
  return BandwidthSelection(bandwidth(NormalReference(), d, k), rule, details)
end

# Points and weights for double sums ∑ᵢ ∑ⱼ wᵢ wⱼ k((xⱼ - xᵢ) / g): the observations
# themselves, or the occupied points of a grid with spacing g / 100 onto which the
# observations are binned linearly; `norm` is W² - ∑ wᵢ² of the observations.
struct _PairSums
  x::Vector{Float64}
  w::Vector{Float64}
  norm::Float64
end

function _PairSums(d::SmoothingData, g::Real, exact::Bool)
  w = d.w === nothing ? ones(nobs(d)) : d.w
  norm = d.sumw^2 - sum(abs2, w)
  exact && return _PairSums(d.x, w, norm)
  runs = _runs(d.x, 24g)
  if length(runs) == 1
    gs, gv = _linearbin!(SumsWorkspace(), d.x, w, g / 100)
    return _PairSums(gs, gv, norm)
  end
  # the runs between gaps wider than the kernels reach are binned one by one, so that
  # far outliers do not need a grid over the whole range
  gs, gv, ws = Float64[], Float64[], SumsWorkspace()
  for r in runs
    rs, rv = _linearbin!(ws, view(d.x, r), view(w, r), g / 100)
    append!(gs, rs)
    append!(gv, rv)
  end
  return _PairSums(gs, gv, norm)
end

# the runs of the sorted x between gaps wider than `gap`
function _runs(x, gap)
  runs = UnitRange{Int}[]
  s = 1
  for i in eachindex(x)
    if i == lastindex(x) || x[i + 1] - x[i] > gap
      push!(runs, s:i)
      s = i + 1
    end
  end
  return runs
end

# ∑ᵢ ∑ⱼ wᵢ wⱼ k((xⱼ - xᵢ) / g), the diagonal included, for a symmetric kernel that is
# negligible beyond `cutoff`
function _pairsum(p::_PairSums, k, g, cutoff)
  x, w = p.x, p.w
  n = length(x)
  k0 = k(0.0)
  s = 0.0
  @inbounds for i in 1:n
    si = 0.0
    for j in i+1:n
      u = (x[j] - x[i]) / g
      u > cutoff && break
      si += w[j] * k(u)
    end
    s += w[i] * (2si + w[i] * k0)
  end
  return s
end

# s^(r+1) ψ̂ᵣ(g) with the Gaussian kernel, exactly or from data binned for this g, for
# a unit s of the bandwidths; φ⁽ʳ⁾(u) is below 1e-24 for |u| > 12 and r ≤ 6
function _psi(d::SmoothingData, r, g, exact::Bool, s = 1.0)
  p = _PairSums(d, g, exact)
  return _pairsum(p, GaussianDerivativeKernel(1.0, r), g, 12.0) / (p.norm * (g / s)^(r + 1))
end

# (r, g, s, m) -> s^(r+1) ψ̂ᵣ(g) for Sheather and Jones: exact double sums, or the
# data binned once on a grid of spacing Δ with the lagged products of the bin
# weights, which makes each evaluation cost O(g / Δ). The binning leaves out pairs
# farther apart than 12 gmax. A g above gmax, or below m Δ, where the grid is too
# coarse for the accuracy wanted, takes another grid, of spacing g / 500, which is
# kept for the following evaluations while it is fine enough and reaches far enough.
function _psievaluator(d::SmoothingData, Δ, exact::Bool, gmax)
  exact && return (r, g, s, m) -> _psi(d, r, g, true, s)
  p = _LaggedBins(d, Δ, 12gmax)
  other = Ref(p)
  fits(q, g, m) = 12g <= q.gap && g >= m * q.Δ
  return function (r, g, s, m)
    fits(p, g, m) && return _psi(p, r, g, s)
    fits(other[], g, m) || (other[] = _LaggedBins(d, g / 500, 48g))
    return _psi(other[], r, g, s)
  end
end

# The weights of linearly binned data on a grid of spacing Δ and their lagged
# products c[m + 1] = ∑ₖ bₖ bₖ₊ₘ, so that ∑ᵢ ∑ⱼ wᵢ wⱼ K((xⱼ - xᵢ) / g), the diagonal
# included, is about c[1] K(0) + 2 ∑ₘ c[m + 1] K(m Δ / g) for a symmetric K.
# Data whose range needs more than 2^20 bins, because of far outliers or heavy
# tails, are split at gaps wider than `gap`, across which the pairs are negligible,
# and of a run that is still wider than 2^18 bins only the densest part is binned.
# The pairs with the other observations, which are sparse, are summed exactly.
struct _LaggedBins
  Δ::Float64
  gap::Float64                  # pairs farther apart are left out; Inf for none
  cs::Vector{Vector{Float64}}   # the lagged products of each binned part
  x::Vector{Float64}
  w::Vector{Float64}
  binned::BitVector             # whether an observation is binned
  rest::Vector{Int}             # the observations that are not
  norm::Float64
end

function _LaggedBins(d::SmoothingData, Δ, gap)
  x, n = d.x, nobs(d)
  w = d.w === nothing ? ones(n) : d.w
  norm = d.sumw^2 - sum(abs2, w)
  (x[end] - x[1]) / Δ + 2 <= 2^20 &&
    return _LaggedBins(Δ, Inf, [_binlags(x, w, 1:n, Δ)], x, w, trues(n), Int[], norm)
  binned = falses(n)
  cs = [begin
          part = _densest(x, r, (2^18 - 2) * Δ)
          binned[part] .= true
          _binlags(x, w, part, Δ)
        end for r in _runs(x, gap)]
  return _LaggedBins(Δ, gap, cs, x, w, binned, findall(!, binned), norm)
end

# the longest stretch of the observations r whose values span at most `width`
function _densest(x, r, width)
  best, from, l = 0, first(r), first(r)
  for i in r
    while x[i] - x[l] > width
      l += 1
    end
    i - l + 1 > best && ((best, from) = (i - l + 1, l))
  end
  return from:from+best-1
end

# the lagged products of the observations r, binned linearly on a grid of spacing Δ
# from the first of them
function _binlags(x, w, r, Δ)
  x0 = x[first(r)]
  M = floor(Int, (x[last(r)] - x0) / Δ) + 2
  b = zeros(M)
  for i in r
    t = (x[i] - x0) / Δ
    j = min(floor(Int, t), M - 2)
    f = t - j
    b[j + 1] += (1 - f) * w[i]
    b[j + 2] += f * w[i]
  end
  return _lagproducts(b)
end

_psi(p::_LaggedBins, r, g, s = 1.0) = _lagsum(p, GaussianDerivativeKernel(1.0, r), g, 12.0) / (p.norm * (g / s)^(r + 1))

# ∑ᵢ ∑ⱼ wᵢ wⱼ K((xⱼ - xᵢ) / g), the diagonal included, for a symmetric K that is
# negligible beyond `cutoff`
function _lagsum(p::_LaggedBins, K, g, cutoff)
  s = 0.0
  for c in p.cs
    s += c[1] * K(0.0)
    for m in 1:min(length(c) - 1, floor(Int, cutoff * g / p.Δ))
      s += 2 * c[m + 1] * K(m * p.Δ / g)
    end
  end
  # the pairs with an observation that is not binned, exactly: in both orders with
  # binned ones, and among themselves from both ends
  x, w = p.x, p.w
  for j in p.rest
    for i in searchsortedfirst(x, x[j] - cutoff * g):searchsortedlast(x, x[j] + cutoff * g)
      t = w[i] * w[j] * K((x[i] - x[j]) / g)
      s += p.binned[i] ? 2t : t
    end
  end
  return s
end

# ∑ₖ bₖ bₖ₊ₘ for m = 0, …, length(b) - 1, from the FFT of b padded with zeros: the
# squared moduli are real and even, so that a second forward transform inverts it
function _lagproducts(b)
  M = length(b)
  N = nextpow(2, 2M)
  z = zeros(ComplexF64, N)
  z[1:M] .= b
  _fft!(z)
  z .= abs2.(z)
  _fft!(z)
  return [real(z[m + 1]) / N for m in 0:M-1]
end

# In-place radix-2 FFT, z[k] ← ∑ⱼ z[j] exp(-2πi (j - 1)(k - 1) / N), for N a power of two
function _fft!(z::Vector{ComplexF64})
  N = length(z)
  ispow2(N) || throw(ArgumentError("the length must be a power of two, got $N"))
  j = 0
  for i in 0:N-2   # bit-reversal permutation
    i < j && ((z[i + 1], z[j + 1]) = (z[j + 1], z[i + 1]))
    m = N >> 1
    while m <= j
      j -= m
      m >>= 1
    end
    j += m
  end
  tw = [cispi(-2k / N) for k in 0:max(N ÷ 2 - 1, 0)]
  len = 2
  while len <= N
    half, step = len >> 1, N ÷ len
    for s in 1:len:N, k in 0:half-1
      a, t = z[s + k], tw[k * step + 1] * z[s + k + half]
      z[s + k], z[s + k + half] = a + t, a - t
    end
    len <<= 1
  end
  return z
end

# A zero of f in [a, b], where f changes sign, by the Illinois variant of regula
# falsi: when the same end of the bracket survives twice in a row, its function value
# is halved. Stops when the bracket is shorter than rtol times its larger end, and
# returns the zero and the number of evaluations of f.
function _findroot(f, a, b; rtol = 1e-10, maxevals = 200)
  fa, fb = f(a), f(b)
  evals, side = 2, 0
  while abs(b - a) > rtol * max(abs(a), abs(b)) && evals < maxevals
    c = (fa * b - fb * a) / (fa - fb)
    fc = f(c)
    evals += 1
    if fc * fb > 0
      b, fb = c, fc
      side == -1 && (fa /= 2)
      side = -1
    elseif fc * fa > 0
      a, fa = c, fc
      side == 1 && (fb /= 2)
      side = 1
    else
      return c, evals
    end
  end
  return (a + b) / 2, evals
end

# The minimizer of f between a < b, by golden-section search to a relative accuracy
# rtol
function _goldenmin(f, a, b; rtol = 1e-6)
  φ = (sqrt(5) - 1) / 2
  c, d = b - φ * (b - a), a + φ * (b - a)
  fc, fd = f(c), f(d)
  while b - a > rtol * (abs(a) + abs(b)) / 2
    if fc < fd
      b, d, fd = d, c, fc
      c = b - φ * (b - a)
      fc = f(c)
    else
      a, c, fc = c, d, fd
      d = a + φ * (b - a)
      fd = f(d)
    end
  end
  return (a + b) / 2
end

# The bandwidth chosen from a criterion on a grid: the global minimizer over the
# finite values, or with `largest = true` the largest local minimizer. An interior
# choice is refined between its neighbours, and the refinement is kept only where it
# is no worse. Returns the details: the grid, the criterion values, the local
# minimizers and whether the choice is at an end of the grid.
function _gridminimize(f, hs; largest::Bool)
  cv = [f(h) for h in hs]
  m = length(hs)
  any(isfinite, cv) || throw(ArgumentError("the criterion is not finite at any bandwidth of the grid"))
  interior = [i for i in 2:m-1 if isfinite(cv[i]) && cv[i] < cv[i - 1] && cv[i] <= cv[i + 1]]
  i = largest && !isempty(interior) ? last(interior) : argmin(map(v -> isfinite(v) ? v : Inf, cv))
  atbound = i == 1 || i == m
  h = hs[i]
  if !atbound && isfinite(cv[i - 1]) && isfinite(cv[i + 1])
    t = _goldenmin(f, hs[i - 1], hs[i + 1])
    f(t) <= cv[i] && (h = t)
  end
  return h, (grid = collect(hs), criterion = cv, minimizers = hs[interior], atbound = atbound)
end

"""
    LSCV(; gridsize = 60, exact = 1000)

Least-squares cross-validation for kernel density estimates (Rudemo 1982; Bowman
1984). It minimizes an unbiased estimate of the integrated squared error minus
`∫ f²`,

    CV(h) = ∫ f̂(x)² dx - 2 ∑_{i ≠ j} wᵢ wⱼ k((xᵢ - xⱼ) / h) / (h (W² - ∑ᵢ wᵢ²)),

over `gridsize` bandwidths spaced evenly on the log scale between `h_OS / 100` and
`4 h_OS`, where `h_OS` is the oversmoothed bandwidth of Terrell (1990), an upper
bound for the bandwidth that minimizes the asymptotic mean integrated squared
error. The minimizer of the criterion itself can exceed `h_OS`, as it does for
normal data about half of the time, hence the factor 4. The lower end uses the
robust scale of [`NormalReference`](@ref), so that outliers and heavy tails, which
inflate the standard deviation, do not lift the whole grid. Following Sheather
(2004), it takes the largest local minimizer on the grid and refines it. The
details of the selection record the grid, the criterion, all local minimizers and
whether the minimum was at an end of the grid, which happens, for example, when ties
make the criterion fall without bound as `h` shrinks; the rule warns then.

Polynomial kernels use exact fast sums. Gaussian kernels use exact double sums for
up to `exact` observations and otherwise the data binned once, on a grid of 100
points per smallest bandwidth, with the lagged products of the bin weights, as in
[`SheatherJones`](@ref); binning moves the bandwidth by about `1e-5` relative. Cross-validation converges slowly, at the relative rate
`n^(-1/10)`, and varies a lot between samples, so it is an alternative to
[`SheatherJones`](@ref) rather than a default.
"""
struct LSCV <: BandwidthSelector
  gridsize::Int
  exact::Int

  function LSCV(gridsize::Integer, exact::Integer)
    gridsize >= 3 || throw(ArgumentError("gridsize must be at least 3, got $gridsize"))
    new(gridsize, exact)
  end
end

LSCV(; gridsize::Integer = 60, exact::Integer = 1000) = LSCV(gridsize, exact)

function selectbandwidth(rule::LSCV, d::SmoothingData, k::AbstractKernel)
  nobs(d) >= 2 || throw(ArgumentError("LSCV needs at least two observations"))
  sd, iqr = _sdiqr(d)
  _check_scale(sd, rule)
  n = effective_nobs(d)
  c = (243 * roughness(k) / (35 * _mu2(k)^2 * n))^(1 / 5)
  hmin, hmax = c * _robustscale(sd, iqr / 1.349) / 100, 4 * c * sd
  crit = _lscvcriterion(d, k, nobs(d) <= rule.exact, hmin)
  hs = exp.(range(log(hmin), log(hmax); length = rule.gridsize))
  h, details = _gridminimize(crit, hs; largest = true)
  details.atbound && @warn "LSCV: the criterion is smallest at an end of the grid of bandwidths, h = $h"
  return BandwidthSelection(h, rule, details)
end

# h ↦ CV(h); ∫ f̂² is a double sum of the convolution of k with its reflection
function _lscvcriterion(d, k::PolynomialKernel, exact, hmin)
  kk = convolve(k, _reflect(k))
  w = d.w === nothing ? ones(nobs(d)) : d.w
  W, sw2 = d.sumw, sum(abs2, w)
  ws = SumsWorkspace()
  return function (h)
    s1 = sum(w .* vec(kernelsums(d, kk, h; workspace = ws)))
    s2 = sum(w .* vec(kernelsums(d, k, h; workspace = ws))) - sw2 * k(0.0)
    return s1 / (h * W^2) - 2 * s2 / (h * (W^2 - sw2))
  end
end

# with more than `exact` observations, the data are binned once on a grid of 100
# points per smallest standard deviation of the kernel, as for Sheather and Jones,
# unless that grid would need more than 2^20 points; then for each h
function _lscvcriterion(d, k::GaussianFamily, exact, hmin)
  σ, r = _scale_order(k)
  r == 0 || throw(ArgumentError("LSCV needs a density kernel, got $k"))
  kk = convolve(k, k)
  Δ = hmin * σ / 100
  if !exact && (d.x[end] - d.x[1]) / Δ + 2 <= 2^20
    p = _LaggedBins(d, Δ, Inf)
    W, sw2 = d.sumw, d.sumw^2 - p.norm
    return function (h)
      s1 = _lagsum(p, kk, h, 12 * kk.σ)
      s2 = _lagsum(p, k, h, 12σ) - sw2 * k(0.0)
      return s1 / (h * W^2) - 2 * s2 / (h * p.norm)
    end
  end
  return function (h)
    p = _PairSums(d, h * σ, exact)
    W, sw2 = d.sumw, d.sumw^2 - p.norm
    s1 = _pairsum(p, kk, h, 12 * kk.σ)
    s2 = _pairsum(p, k, h, 12σ) - sw2 * k(0.0)
    return s1 / (h * W^2) - 2 * s2 / (h * p.norm)
  end
end

_lscvcriterion(d, k::AbstractKernel, exact, hmin) =
  throw(ArgumentError("LSCV needs the convolution of the kernel with itself, which is not available for $k"))

"""
    CrossValidation(; gridsize = 40)

Leave-one-out cross-validation for local polynomial regression: the bandwidth that
minimizes

    CV(h) = ∑ᵢ wᵢ (yᵢ - m̂₋ᵢ(xᵢ))² / ∑ᵢ wᵢ,   m̂₋ᵢ(xᵢ) = (m̂(xᵢ) - ℓᵢᵢ yᵢ) / (1 - ℓᵢᵢ),

where `m̂₋ᵢ(xᵢ)` is the fit at `xᵢ` without observation `i` and `ℓᵢᵢ` the weight
of `yᵢ` in the fit at `xᵢ`. The fits at all observations come from
[`kernelsums`](@ref), so that a bandwidth costs `O(n)` operations for polynomial
kernels. The grid has `gridsize` bandwidths spaced evenly on the log scale, from
the smallest one that leaves `p + 1` other distinct values in every window, below
which leave-one-out fits break down, to the range of the data. The global minimizer
on the grid is refined, and the details record the grid and the criterion.
Cross-validation estimates the regression function, so it needs `derivative = 0`.
"""
struct CrossValidation <: BandwidthSelector
  gridsize::Int

  function CrossValidation(gridsize::Integer)
    gridsize >= 3 || throw(ArgumentError("gridsize must be at least 3, got $gridsize"))
    new(gridsize)
  end
end

CrossValidation(; gridsize::Integer = 40) = CrossValidation(gridsize)

function selectbandwidth(rule::CrossValidation, d::SmoothingData, k::AbstractKernel;
                         degree::Integer = 1, derivative::Integer = 0, points = nothing, kwargs...)
  d.y === nothing && throw(ArgumentError("CrossValidation needs responses; use SmoothingData(x, y)"))
  derivative == 0 || throw(ArgumentError(
    "CrossValidation chooses bandwidths for the regression function, derivative = 0"))
  p = Int(degree)
  nunique(d) >= p + 2 || throw(ArgumentError("CrossValidation needs at least $(p + 2) distinct values"))
  hmin = 1.01 * maximum(knn_distance(d, x, p + 2; distinct = true) for x in d.values)
  hmax = d.x[end] - d.x[1]
  hmin < hmax || (hmin = hmax / 2)
  ws = SumsWorkspace()
  crit(h) = _cvcriterion(d, k, h, p, ws)
  h, details = _gridminimize(crit, exp.(range(log(hmin), log(hmax); length = rule.gridsize));
                             largest = false)
  return BandwidthSelection(h, rule, details)
end

function _cvcriterion(d, k, h, p, ws)
  S = kernelsums(d, k, h; kmax = 2p, workspace = ws)
  T = kernelsums(d, k, h; kmax = p, z = d.y, workspace = ws)
  k0 = k(0.0)
  # chunks of a fixed size, run as tasks and summed in order, so that the criterion
  # does not depend on the number of threads
  parts = _fetchall([Threads.@spawn _cvpart(d, S, T, k0, p, r) for r in Iterators.partition(1:nobs(d), 8192)])
  return sum(parts) / d.sumw
end

# the leave-one-out squared errors of the observations in r, or Inf if a fit there
# does not exist
function _cvpart(d, S, T, k0, p, r)
  A, e = zeros(p + 1, p + 1), zeros(p + 1)
  s = 0.0
  for i in r
    for b in 0:p, a in 0:p
      A[a + 1, b + 1] = S[a + b + 1, i]
    end
    _solvefirst!(e, A) || return Inf
    fit = 0.0
    for j in 1:p+1
      fit += e[j] * T[j, i]
    end
    wi = _weight(d.w, i)
    lii = wi * k0 * e[1]
    lii < 1 - 1e-10 || return Inf
    s += wi * (d.y[i] - (fit - lii * d.y[i]) / (1 - lii))^2
  end
  return s
end

# e = A⁻¹ e₁ for a symmetric positive definite A, by a Cholesky factorization in the
# lower triangle of A, without allocating; false if a pivot is not positive
function _solvefirst!(e, A)
  n = size(A, 1)
  for j in 1:n
    t = A[j, j]
    for l in 1:j-1
      t -= A[j, l]^2
    end
    t > 0 || return false
    A[j, j] = sqrt(t)
    for i in j+1:n
      u = A[i, j]
      for l in 1:j-1
        u -= A[i, l] * A[j, l]
      end
      A[i, j] = u / A[j, j]
    end
  end
  for j in 1:n   # L y = e₁
    u = j == 1 ? 1.0 : 0.0
    for l in 1:j-1
      u -= A[j, l] * e[l]
    end
    e[j] = u / A[j, j]
  end
  for j in n:-1:1   # Lᵀ e = y
    u = e[j]
    for l in j+1:n
      u -= A[l, j] * e[l]
    end
    e[j] = u / A[j, j]
  end
  return true
end
