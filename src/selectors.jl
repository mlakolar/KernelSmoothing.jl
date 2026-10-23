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
for up to `10^6` observations. R's `bw.SJ` bins differences of
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
  c1 = roughness(k) / (_mu2(k)^2 * n)
  exact = nobs(d) <= rule.exact
  # normal-reference pilot bandwidths for ψ₄ and ψ₆ (Sheather and Jones 1991)
  a, b = 1.24 * σ * n^(-1 / 7), 1.23 * σ * n^(-1 / 9)
  # the root finding evaluates ψ₄ at bandwidths down to about a / 5
  ψ = _psievaluator(d, min(a, b) / 500, exact)
  TD = -ψ(6, b)
  TD > 0 || return _sparse(rule, d, k, "the estimate of ψ₆ is not negative")
  if rule.method == :dpi
    g = (2 * 3 * INV_SQRT_2PI / (n * TD))^(1 / 7)   # 2 φ⁽⁴⁾(0) / (n (-ψ̂₆))
    ψ4 = ψ(4, g)
    ψ4 > 0 || return _sparse(rule, d, k, "the estimate of ψ₄ is not positive")
    return BandwidthSelection((c1 / ψ4)^(1 / 5), rule,
      (scale = σ, pilot = g, psi4 = ψ4, psi6 = -TD, exact = exact, fallback = false))
  end
  SD = ψ(4, a)
  SD > 0 || return _sparse(rule, d, k, "the estimate of ψ₄ is not positive")
  # g(h) = α₂ h^(5/7) balances the bias and variance of ψ̂₄ at the optimal h
  α2 = (2 * 3 * INV_SQRT_2PI * _mu2(k)^2 / roughness(k))^(1 / 7) * (SD / TD)^(1 / 7)
  hmax = (243 * roughness(k) / (35 * _mu2(k)^2 * n))^(1 / 5) * σ   # Terrell's oversmoothed bandwidth
  lower, upper = 0.1 * hmax, hmax
  f(h) = (c1 / max(ψ(4, α2 * h^(5 / 7)), floatmin()))^(1 / 5) - h
  tries = 0
  while f(lower) * f(upper) > 0   # widen the bracket, as R's bw.SJ does
    tries < 100 || return _sparse(rule, d, k, "its equation has no solution")
    isodd(tries) ? (lower /= 1.2) : (upper *= 1.2)
    tries += 1
  end
  # with binned data, ψ̂₄ jitters slightly as the grid changes with g, so the
  # bracket stops earlier
  h, iterations = _findroot(f, lower, upper; rtol = exact ? 1e-10 : 1e-8)
  return BandwidthSelection(h, rule, (scale = σ, pilots = (a, b), alpha2 = α2,
    psi4 = ψ(4, α2 * h^(5 / 7)), psi6 = -TD, exact = exact,
    iterations = iterations, fallback = false))
end

# For a sample too sparse for the estimates of Sheather and Jones: the normal-reference
# bandwidth, with a warning
function _sparse(rule, d, k, why)
  @warn "SheatherJones: $why, so the normal-reference bandwidth is used instead"
  s = selectbandwidth(NormalReference(), d, k)
  return BandwidthSelection(s.h, rule, (s.details..., fallback = true))
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
  gs, gv = _linearbin!(SumsWorkspace(), d.x, w, g / 100)
  return _PairSums(gs, gv, norm)
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

# ψ̂ᵣ(g) with the Gaussian kernel, exactly or from data binned for this g; φ⁽ʳ⁾(u) is
# below 1e-24 for |u| > 12 and r ≤ 6
function _psi(d::SmoothingData, r, g, exact::Bool)
  p = _PairSums(d, g, exact)
  return _pairsum(p, GaussianDerivativeKernel(1.0, r), g, 12.0) / (p.norm * g^(r + 1))
end

# (r, g) -> ψ̂ᵣ(g) for Sheather and Jones: exact double sums, or the data binned once
# on a grid of spacing Δ with the lagged products of the bin weights, which makes each
# evaluation cost O(g / Δ). A grid of more than 2^20 points, for data with far
# outliers, falls back to binning for each g, which visits only the occupied points.
function _psievaluator(d::SmoothingData, Δ, exact::Bool)
  exact && return (r, g) -> _psi(d, r, g, true)
  M = (d.x[end] - d.x[1]) / Δ + 2
  M <= 2^20 || return (r, g) -> _psi(d, r, g, false)
  p = _LaggedBins(d, Δ, floor(Int, M))
  return (r, g) -> _psi(p, r, g)
end

# The weights of linearly binned data on a grid of spacing Δ and their lagged
# products c[m + 1] = ∑ₖ bₖ bₖ₊ₘ, so that ∑ᵢ ∑ⱼ wᵢ wⱼ K((xⱼ - xᵢ) / g), the diagonal
# included, is about c[1] K(0) + 2 ∑ₘ c[m + 1] K(m Δ / g) for a symmetric K
struct _LaggedBins
  Δ::Float64
  c::Vector{Float64}
  norm::Float64
end

function _LaggedBins(d::SmoothingData, Δ, M)
  w = d.w === nothing ? ones(nobs(d)) : d.w
  b = zeros(M)
  for i in eachindex(d.x)
    t = (d.x[i] - d.x[1]) / Δ
    j = min(floor(Int, t), M - 2)
    f = t - j
    b[j + 1] += (1 - f) * w[i]
    b[j + 2] += f * w[i]
  end
  return _LaggedBins(Δ, _lagproducts(b), d.sumw^2 - sum(abs2, w))
end

function _psi(p::_LaggedBins, r, g)
  k = GaussianDerivativeKernel(1.0, r)
  s = p.c[1] * k(0.0)
  for m in 1:min(length(p.c) - 1, floor(Int, 12g / p.Δ))
    s += 2 * p.c[m + 1] * k(m * p.Δ / g)
  end
  return s / (p.norm * g^(r + 1))
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

# The bandwidth chosen from a criterion on a grid: the largest (or global) local
# minimizer on the grid, refined between its neighbours, with the grid, the
# criterion values, the local minimizers and whether the minimum is at an end.
function _gridminimize(f, hs; largest::Bool)
  cv = [f(h) for h in hs]
  m = length(hs)
  interior = [i for i in 2:m-1 if cv[i] < cv[i - 1] && cv[i] <= cv[i + 1] && isfinite(cv[i])]
  if isempty(interior)
    i = argmin(map(v -> isfinite(v) ? v : Inf, cv))
    atbound = true
  else
    i = largest ? last(interior) : interior[argmin(cv[interior])]
    atbound = false
  end
  h = atbound ? hs[i] : _goldenmin(f, hs[i - 1], hs[i + 1])
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
normal data about half of the time, hence the factor 4. Following Sheather (2004),
it takes the largest local minimizer on the grid and refines it. The details of the selection record the grid, the criterion,
all local minimizers and whether the minimum was at an end of the grid, which
happens, for example, when ties make the criterion fall without bound as `h`
shrinks.

Polynomial kernels use exact fast sums. Gaussian kernels use exact double sums for
up to `exact` observations and binned ones otherwise, as in
[`SheatherJones`](@ref); since the criterion is flat near its minimum, binning moves
the bandwidth by about `1e-4` relative. Cross-validation converges slowly, at the relative rate
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
  σ = _std(d)
  _check_scale(σ, rule)
  n = effective_nobs(d)
  hos = (243 * roughness(k) / (35 * _mu2(k)^2 * n))^(1 / 5) * σ
  crit = _lscvcriterion(d, k, hos / 100, nobs(d) <= rule.exact)
  hs = exp.(range(log(hos / 100), log(4hos); length = rule.gridsize))
  h, details = _gridminimize(crit, hs; largest = true)
  return BandwidthSelection(h, rule, details)
end

# h ↦ CV(h); ∫ f̂² is a double sum of the convolution of k with its reflection
function _lscvcriterion(d, k::PolynomialKernel, hmin, exact)
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

function _lscvcriterion(d, k::GaussianFamily, hmin, exact)
  σ, r = _scale_order(k)
  r == 0 || throw(ArgumentError("LSCV needs a density kernel, got $k"))
  kk = convolve(k, k)
  return function (h)
    p = _PairSums(d, h * σ, exact)
    W, sw2 = d.sumw, d.sumw^2 - p.norm
    s1 = _pairsum(p, kk, h, 12 * kk.σ)
    s2 = _pairsum(p, k, h, 12σ) - sw2 * k(0.0)
    return s1 / (h * W^2) - 2 * s2 / (h * p.norm)
  end
end

_lscvcriterion(d, k::AbstractKernel, hmin, exact) =
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
