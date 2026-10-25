## Data-driven bandwidth selectors

"""
    SheatherJones(; method = :ste, exact = 100, compat = false)

The plug-in bandwidth of Sheather and Jones (1991) for kernel density estimates.
It estimates `ψ₄ = ∫ f''(x)² dx` with a Gaussian pilot kernel and plugs the
estimate into the bandwidth that minimizes the asymptotic mean integrated
squared error,

    h = (R(k) / (μ₂(k)² ψ̂₄(g) n))^(1/5),

where `R(k) = roughness(k)`, `μ₂(k) = moment(k, 2)` and `n` is the
[`effective_nobs`](@ref). With `method = :ste`, "solve the equation", the pilot
bandwidth `g = α₂ h^(5/7)` depends on `h` itself, with `α₂` estimated from `ψ̂₄`
and `ψ̂₆` at normal-reference pilot bandwidths, and the equation is solved to a
relative accuracy of `1e-10`, or `1e-8` from binned data. With `method = :dpi`, the direct plug-in, `g` is
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
for `10^6` observations of a smooth density; heavy tails take longer, since their
sparse stretches are summed exactly. Data that would need more than `2^20` grid points,
because of far outliers or heavy tails, are split at gaps across which the pilot
kernels are negligible, and each run into stretches of at most `2^18` grid points,
of which the densest one and the others that are dense enough are binned; the pairs
with the other observations, which are sparse, and the pairs across the boundaries
of binned stretches are summed exactly. R's `bw.SJ` bins differences of
pairs of observations coarsely and stops its search early, so by default it agrees
with this rule to two or three digits only. With fine bins it agrees to about
`1e-6` for `method = :dpi` and `4e-5` for `:ste`, where R rounds the constant
`(2 φ⁽⁴⁾(0) μ₂(k)² / R(k))^(1/7) = 1.35733` of the Gaussian kernel to 1.357. This rule is the recommended default for
densities without boundaries (Jones, Marron and Sheather 1996).

With rounded data, pairs of equal values make `ψ̂₄(g)` grow like `g⁻⁵` as the pilot
bandwidth shrinks, and the equation gains roots at bandwidths that resolve the grid
of the values: for `10⁵` normal observations rounded to `0.1`, the roots were `0.0026`,
`0.017` and `0.106`, and R's `bw.SJ` takes the smallest, at which the estimate is a
comb of spikes. The rule therefore searches only bandwidths with `h sd(k) ≥ r/2`,
where `sd(k) = √μ₂(k)` and `r` is the median gap between neighbouring distinct
values, which for data that are not rounded is of the order of their range over
`n` and changes nothing. Where no root is left, or where the direct plug-in falls
below this bound, the normal-reference bandwidth is used, raised to the bound if it
is smaller, with a warning; for integers with a standard deviation of 2, the normal
reference alone gave 0.25 for the Gaussian kernel, which still resolves the integers.

`compat = true` gives R's `bw.SJ` exactly, as `density(x, bw = "SJ", kernel = …)`
uses it: the bandwidth for the Gaussian kernel, from R's binned pair counts, rounded
constants and root search, divided by `√μ₂(k)` for other kernels, so that the kernel
has the standard deviation of R's bandwidth. Like `bw.SJ`, it takes no weights and
throws where `bw.SJ` stops. For data far from zero relative to their range, beyond
about `2 × 10⁶` times the range (2³¹ bins of 1.01/1000 of it), R 4.5's integer bins overflow and put every pair
at distance zero; this rule bins from the minimum, as R's development sources do.
"""
struct SheatherJones <: BandwidthSelector
  method::Symbol
  exact::Int
  compat::Bool    # R's bw.SJ

  function SheatherJones(method::Symbol, exact::Integer, compat::Bool)
    method in (:ste, :dpi) || throw(ArgumentError("method must be :ste or :dpi, got :$method"))
    new(method, exact, compat)
  end
end

SheatherJones(; method::Symbol = :ste, exact::Integer = 100, compat::Bool = false) =
  SheatherJones(method, exact, compat)

# The details of a selection, in the units of the data, of one shape in every case,
# so that the selection is inferable
_sjdetails(; scale, pilots, pilot = NaN, alpha2 = NaN, psi4 = NaN, psi6 = NaN, exact, iterations = 0,
           fallback = false, resolution) =
  (scale = Float64(scale), pilots = Float64.(pilots), pilot = Float64(pilot), alpha2 = Float64(alpha2),
   psi4 = Float64(psi4), psi6 = Float64(psi6), exact = exact, iterations = Int(iterations), fallback = fallback,
   resolution = Float64(resolution))

# The median gap between neighbouring distinct values, the resolution of rounded data,
# from about 1000 gaps evenly spread over them: for rounded data most gaps equal the
# rounding unit, and for other data only the order of the median matters, so that a
# sort of all gaps, which took ten times the rest of the rule at n = 10⁶, is not needed
function _mediangap(d::SmoothingData)
  U = nunique(d)
  U < 2 && return 0.0
  v = d.values
  g = [v[i + 1] - v[i] for i in 1:max(1, (U - 1) ÷ 1000):U-1]
  m = length(g)
  return isodd(m) ? partialsort!(g, (m + 1) ÷ 2) : (partialsort!(g, m ÷ 2) + partialsort!(g, m ÷ 2 + 1)) / 2
end

function selectbandwidth(rule::SheatherJones, d::SmoothingData, k::AbstractKernel)
  nobs(d) >= 2 || throw(ArgumentError("SheatherJones needs at least two observations"))
  _check_mass(k, rule)
  rule.compat && return _sjcompat(rule, d, k)
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
  # bandwidths that resolve the grid of rounded data, t σ sd(k) < r/2, are left out
  r = _mediangap(d)
  tres = r / (2σ * sqrt(μ2))
  details(; pilot = NaN, alpha2 = NaN, psi4 = NaN, psi6 = NaN, iterations = 0, fallback = false) =
    _sjdetails(; scale = σ, pilots = (σ * a, σ * b), pilot = σ * pilot, alpha2 = alpha2 * σ^(2 / 7),
               psi4 = psi4 / σ^5, psi6 = psi6 / σ^7, exact, iterations, fallback, resolution = r)
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
    t = (c1 / ψ4)^(1 / 5)
    t >= tres || return _sparse(rule, d, k, "its bandwidth resolves the grid of the rounded data",
                                details(pilot = g, psi4 = ψ4, psi6 = -TD, fallback = true); floor = σ * tres)
    return BandwidthSelection(σ * t, rule, details(pilot = g, psi4 = ψ4, psi6 = -TD))
  end
  SD = ψ(4, a)
  SD > 0 || return _sparse(rule, d, k, "the estimate of ψ₄ is not positive",
                           details(psi4 = SD, psi6 = -TD, fallback = true))
  # g(h) = α₂ h^(5/7) balances the bias and variance of ψ̂₄ at the optimal h
  α2 = (2 * 3 * INV_SQRT_2PI * μ2^2 / roughness(k))^(1 / 7) * (SD / TD)^(1 / 7)
  hmax = (243 * roughness(k) / (35 * μ2^2 * n))^(1 / 5)   # Terrell's oversmoothed bandwidth
  lower = max(0.1 * hmax, tres)
  upper = max(hmax, 1.2 * lower)
  # the bracket needs only the signs, which a coarser grid gives
  fm(h, m) = (c1 / max(ψm(4, α2 * h^(5 / 7), m), floatmin()))^(1 / 5) - h
  f(h) = fm(h, 50)
  tries = 0
  while f(lower) * f(upper) > 0   # widen the bracket, as R's bw.SJ does, not below tres
    tries < 100 || return _sparse(rule, d, k, lower > tres ? "its equation has no solution" :
                                  "its equation has no solution at bandwidths that do not resolve the grid of the rounded data",
                                  details(alpha2 = α2, psi6 = -TD, fallback = true); floor = σ * tres)
    isodd(tries) && lower / 1.2 >= tres ? (lower /= 1.2) : (upper *= 1.2)
    tries += 1
  end
  # with binned data, ψ̂₄ jitters slightly as the grid changes with g, so the
  # bracket stops earlier
  t, iterations = _findroot(f, lower, upper; rtol = exact ? 1e-10 : 1e-8)
  # Clustered or discrete data can put the pilot bandwidth of the root far below the
  # pilots; where the grid has fewer than 100 points per pilot bandwidth there, the
  # root is solved again on a grid with at least 250, within a quarter of the first
  # root, which is accurate to about 1e-3 there
  if !exact && σ * α2 * t^(5 / 7) < 100Δ
    lo, hi = max(t / 1.25, tres), t * 1.25
    if fm(lo, 250) * fm(hi, 250) <= 0
      t, more = _findroot(h -> fm(h, 250), lo, hi; rtol = 1e-8)
      iterations += more
    end
  end
  g = α2 * t^(5 / 7)
  return BandwidthSelection(σ * t, rule, details(; pilot = g, alpha2 = α2, psi4 = ψ(4, g), psi6 = -TD, iterations))
end

## SheatherJones(compat = true), which reproduces R's bw.SJ (R 4.5)

# The pair counts use a grid of this many bins over 1.01 times the range of the data.
# Up to `_SJR_PAIRWISE` observations each pair is placed on the grid by the bins of its
# two observations; above it the observations are binned first (see `_sjpaircounts`).
const _SJR_NBINS = 1000
const _SJR_PAIRWISE = 500

# The selector of Sheather and Jones (1991) with the Gaussian kernel as both the kernel
# and the pilot kernel, computed as R's bw.SJ computes it for `density(x, bw = "SJ")`:
# the density functionals ψ₄ = ∫ f⁽⁴⁾ f and ψ₆ = ∫ f⁽⁶⁾ f are estimated from the
# numbers of pairs of observations at each distance on a coarse grid, with the
# normal-scale pilot bandwidths and the constants of Sheather and Jones rounded to
# three or four digits, and the equation of the solve-the-equation rule is solved only
# to a tenth of the lower end of its bracket. The bandwidth h of the Gaussian kernel is
# divided by √μ₂(k), so that the kernel k has the standard deviation h, as in density().
function _sjcompat(rule::SheatherJones, d::SmoothingData, k::AbstractKernel)
  d.w === nothing || throw(ArgumentError(
    "SheatherJones(compat = true) follows R's bw.SJ, which takes no weights; use compat = false with weights"))
  x, n = d.x, nobs(d)
  x[end] > x[1] || throw(ArgumentError("SheatherJones(compat = true) needs data with at least two distinct values"))
  sdk = sqrt(_mu2centred(k))
  # the normal scale: the smaller of the standard deviation and IQR / 1.349, the
  # interquartile range of the standard normal distribution
  σ = min(_std(d), (_unweightedquantile(x, 0.75) - _unweightedquantile(x, 0.25)) / 1.349)
  Δ = 1.01 * (x[end] - x[1]) / _SJR_NBINS
  pairwise = n <= _SJR_PAIRWISE
  counts = _sjpaircounts(x, Δ, pairwise)
  ψ4(g) = _sjpsi(counts, n, Δ, g, Val(4))
  ψ6(g) = _sjpsi(counts, n, Δ, g, Val(6))
  # the normal-scale pilot bandwidths for ψ₄ and ψ₆ (Sheather and Jones 1991, sec. 3)
  a = 1.24 * σ * n^(-1 / 7)
  b = 1.23 * σ * n^(-1 / 9)
  T = -ψ6(b)   # estimates ∫ (f''')² > 0
  isfinite(T) && T > 0 || throw(ArgumentError(
    "SheatherJones(compat = true): the estimate of ψ₆ is not negative, the sample is too sparse"))
  Rn = 1 / (2 * sqrt(π) * n)   # the roughness of the Gaussian kernel over n
  if rule.method == :dpi
    # the pilot bandwidth that is optimal for ψ̂₄ when ψ₆ = -T, (2 φ⁽⁴⁾(0) / (n T))^(1/7)
    # (Wand and Jones 1995, sec. 3.6), with 2 φ⁽⁴⁾(0) = 6 / √(2π) rounded to 2.394
    g = (2.394 / (n * T))^(1 / 7)
    S = ψ4(g)
    return BandwidthSelection(_fifthroot(Rn / S) / sdk, rule,
                              _sjdetails(; scale = σ, pilots = (a, b), pilot = g, psi4 = S, psi6 = -T,
                                         exact = pairwise, resolution = _mediangap(d)))
  end
  # Solve the equation h = (R(φ) / (n ψ̂₄(g(h))))^(1/5) with the pilot bandwidth
  # g(h) = α₂ h^(5/7) (Sheather and Jones 1991, eq. 8 and sec. 3), where
  # α₂ = (2 φ⁽⁴⁾(0) / R(φ))^(1/7) (ψ₄ / -ψ₆)^(1/7), its constant 1.35733 rounded to 1.357
  # and ψ₄ estimated at the pilot bandwidth a
  ratio = ψ4(a) / T
  α2 = ratio >= 0 ? 1.357 * ratio^(1 / 7) : NaN
  isfinite(α2) || throw(ArgumentError(
    "SheatherJones(compat = true): the pilot constant α₂ is not finite, the sample is too sparse"))
  F(h) = _fifthroot(Rn / ψ4(α2 * h^(5 / 7))) - h
  # a bracket of a tenth of the oversmoothed bandwidth of the Gaussian kernel up to it,
  # widened alternately upward and downward by factors of 1.2 until F changes sign
  hmax = 1.144 * σ * n^(-1 / 5)
  lo, hi = 0.1 * hmax, hmax
  flo, fhi = F(lo), F(hi)
  widenings = 0
  while flo * fhi > 0
    widenings < 99 || throw(ArgumentError(
      "SheatherJones(compat = true): the equation has no solution within 99 widenings of its bracket"))
    widenings += 1
    if isodd(widenings)
      hi *= 1.2
      fhi = F(hi)
    else
      lo /= 1.2
      flo = F(lo)
    end
  end
  h, iterations = _brentzero(F, lo, hi, flo, fhi, 0.1 * lo)
  g = α2 * h^(5 / 7)
  return BandwidthSelection(h / sdk, rule,
                            _sjdetails(; scale = σ, pilots = (a, b), pilot = g, alpha2 = α2, psi4 = ψ4(g),
                                       psi6 = -T, exact = pairwise, iterations, resolution = _mediangap(d)))
end

# the fifth root, NaN for negative numbers, as floating-point powers give it
_fifthroot(t) = t >= 0 ? t^(1 / 5) : NaN

# The numbers c[ℓ + 1] of the pairs of distinct observations whose bins on a grid of
# width Δ are ℓ apart, ℓ = 0, …, _SJR_NBINS - 1, from the sorted x; the bins truncate
# toward zero, so that the bin around zero is twice as wide as the others.
# - Binned (`pairwise = false`): observation i is in bin trunc(xᵢ / Δ), counted from the
#   bin of x₁, and the observations beyond the last bin of the grid are left out.
# - Pairwise: observation i is in bin trunc((xᵢ - o) / Δ), with o = 0 unless the bins of
#   x₁ or xₙ would be outside the range of 32-bit integers, and o = x₁ then.
# Both are counted in the same way. The bins of the sorted observations do not decrease,
# so that they form runs of equal bins, and the pairs within a run of m observations
# number m (m - 1) / 2, those between two runs the product of their lengths.
function _sjpaircounts(x::Vector{Float64}, Δ::Float64, pairwise::Bool)
  N, n = _SJR_NBINS, length(x)
  o = pairwise && (x[1] / Δ < -2.0^31 || x[end] / Δ > 2.0^31 - 1) ? x[1] : 0.0
  # each run is stored as one integer, its length times N plus its bin counted from the
  # first, in 32 bits where they suffice; there are at most min(n, N) runs, since these
  # bins are 0, …, N - 1
  K = min(n, N)
  runs = (n + 1) * N <= typemax(Int32) ? Vector{Int32}(undef, K) : Vector{Int}(undef, K)
  return _sjpaircounts!(zeros(Int, N), runs, x, Δ, o)
end

function _sjpaircounts!(c::Vector{Int}, runs::Vector{<:Integer}, x, Δ, o)
  N = length(c)
  bin1 = trunc((x[1] - o) / Δ)
  nruns, current = 0, -1
  for xi in x
    t = trunc((xi - o) / Δ) - bin1
    t < N || break   # beyond the grid; for the pairwise bins, (xₙ - x₁) / Δ < N - 1 anyway
    j = Int(t)
    if j == current
      runs[nruns] += N
    else
      nruns += 1
      runs[nruns] = N + j
      current = j
    end
  end
  @inbounds for r in 1:nruns
    m, j = divrem(Int(runs[r]), N)
    c[1] += m * (m - 1) ÷ 2
    for s in r+1:nruns
      ms, js = divrem(Int(runs[s]), N)
      c[js - j + 1] += m * ms
    end
  end
  return c
end

# The estimates of Sheather and Jones (1991, sec. 2) for r = 4, 6, with the diagonal,
#   ψ̂ᵣ(g) = ∑ᵢ ∑ⱼ φ_g⁽ʳ⁾(xᵢ - xⱼ) / (n (n - 1)),   φ_g⁽ʳ⁾(u) = Heᵣ(u / g) φ(u / g) / g^(r+1),
# with the distances of the pairs taken from their bins, ℓΔ for the c[ℓ + 1] pairs ℓ bins
# apart (Wand and Jones 1995, sec. 3.6), and Heᵣ the probabilists' Hermite polynomials:
#   ψ̂ᵣ(g) = [n Heᵣ(0) + 2 ∑ₗ c[ℓ + 1] Heᵣ(zₗ) exp(-zₗ² / 2)] / (√(2π) n (n - 1) g^(r+1)),   zₗ = ℓΔ / g.
# The sum stops at the first ℓ with zₗ² ≥ 1000, where exp(-zₗ² / 2) ≤ e⁻⁵⁰⁰.
function _sjpsi(c::Vector{Int}, n::Int, Δ::Float64, g::Float64, r::Val{R}) where {R}
  s = 0.0
  @inbounds for i in eachindex(c)
    z = (i - 1) * Δ / g
    z2 = z * z
    z2 >= 1000 && break
    ci = c[i]
    ci == 0 || (s += ci * (_hermite(r, z2) * exp(-z2 / 2)))
  end
  return (n * _hermite(r, 0.0) + 2 * s) / (n * (n - 1) * g^(R + 1)) * INV_SQRT_2PI
end

# He₄ and He₆ as functions of z²
_hermite(::Val{4}, z2) = z2 * z2 - 6 * z2 + 3
_hermite(::Val{6}, z2) = z2 * z2 * z2 - 15 * z2 * z2 + 45 * z2 - 15

# Brent's zero finder (Brent 1973, ch. 4) for f on [lo, hi], given flo = f(lo) and
# fhi = f(hi) of opposite signs or one of them zero, to the absolute tolerance tol. It
# returns the zero and the number of iterations, or -1 for them if maxit iterations did
# not converge. This is the variant that R's bw.SJ uses, whose iterates matter there since
# its coarse tolerance stops the search early: it tests whether to interpolate with the
# most recent step, where Brent's algorithm uses the step before it.
function _brentzero(f, lo, hi, flo, fhi, tol; maxit::Integer = 1000)
  flo == 0 && return lo, 0
  fhi == 0 && return hi, 0
  # b is the estimate, of the smallest |f| so far; f(b) and f(c) have opposite signs, so
  # that the zero is between b and c; a is the previous estimate
  a, fa = lo, flo
  b, fb = hi, fhi
  c, fc = lo, flo
  for k in 1:maxit+1
    laststep = b - a
    if abs(fc) < abs(fb)
      a, fa = b, fb
      b, fb = c, fc
      c, fc = a, fa
    end
    τ = 2 * eps(Float64) * abs(b) + tol / 2
    halfway = (c - b) / 2
    (abs(halfway) <= τ || fb == 0) && return b, k - 1
    step = halfway   # bisection, unless an interpolation is accepted
    if abs(laststep) >= τ && abs(fa) > abs(fb)
      cb = c - b
      if a == c   # through two points: the secant
        t = fb / fa
        p, q = cb * t, 1 - t
      else        # through three: inverse quadratic interpolation
        u, t1, t2 = fa / fc, fb / fc, fb / fa
        p = t2 * (cb * u * (u - t1) - (b - a) * (t1 - 1))
        q = (u - 1) * (t1 - 1) * (t2 - 1)
      end
      if p > 0
        q = -q
      else
        p = -p
      end
      # the step p / q must stay within three quarters of the way to c and be shorter
      # than half the last step
      if p < 0.75 * cb * q - abs(τ * q) / 2 && p < abs(laststep * q / 2)
        step = p / q
      end
    end
    abs(step) < τ && (step = step > 0 ? τ : -τ)
    a, fa = b, fb
    b += step
    fb = f(b)
    if (fb > 0 && fc > 0) || (fb < 0 && fc < 0)
      c, fc = a, fa
    end
  end
  return b, -1
end

# For a sample too sparse for the estimates of Sheather and Jones: the normal-reference
# bandwidth, with a warning; for rounded data at least `floor`, the bandwidth at which
# the estimate stops resolving the grid of the values
function _sparse(rule, d, k, why, details; floor = 0.0)
  h = bandwidth(NormalReference(), d, k)
  @warn "SheatherJones: $why, so the normal-reference bandwidth" *
        (floor > h ? ", raised to half the resolution of the rounded data," : "") * " is used instead"
  return BandwidthSelection(max(h, floor), rule, details)
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
# and each run into stretches of at most 2^18 bins: its densest stretch, which is
# binned, and consecutive ones on either side, which are binned where their pairs
# within reach of a pilot bandwidth would outnumber their bins. The pairs with the
# observations of the other stretches, which are sparse, are summed exactly, and so
# are the pairs across the boundary of two neighbouring binned stretches; with only
# the densest stretch binned, heavy tails made the exact pairs grow quadratically.
struct _LaggedBins
  Δ::Float64
  gap::Float64                  # pairs farther apart are left out; Inf for none
  cs::Vector{Vector{Float64}}   # the lagged products of each binned part
  parts::Vector{UnitRange{Int}} # the observations of each binned part, in increasing order
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
    return _LaggedBins(Δ, Inf, [_binlags(x, w, 1:n, Δ)], [1:n], x, w, trues(n), Int[], norm)
  binned = falses(n)
  cs, parts = Vector{Float64}[], UnitRange{Int}[]
  width = (2^18 - 2) * Δ
  for r in _runs(x, gap)
    D = _densest(x, r, width)
    for part in [_stretches(x, first(r):first(D)-1, width); [D]; _stretches(x, last(D)+1:last(r), width)]
      # binning costs a multiple of the number of bins once, the exact pairs their
      # number at each evaluation
      M = (x[last(part)] - x[first(part)]) / Δ + 2
      if part == D || 8 * _pairsnear(x, part, 1200Δ) >= M
        binned[part] .= true
        push!(parts, part)
        push!(cs, _binlags(x, w, part, Δ))
      end
    end
  end
  return _LaggedBins(Δ, gap, cs, parts, x, w, binned, findall(!, binned), norm)
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

# the number of ordered pairs of the observations r within `reach` of each other
function _pairsnear(x, r, reach)
  pairs, l = 0, first(r)
  for i in r
    while x[i] - x[l] > reach
      l += 1
    end
    pairs += 2 * (i - l) + 1
  end
  return pairs
end

# consecutive stretches of the observations r, each spanning at most `width`
function _stretches(x, r, width)
  out = UnitRange{Int}[]
  s = first(r)
  while s <= last(r)
    e = min(searchsortedlast(x, x[s] + width), last(r))
    push!(out, s:e)
    s = e + 1
  end
  return out
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
  x, w = p.x, p.w
  reach = cutoff * g
  # the pairs across the boundary of two neighbouring binned parts, which neither part
  # holds, in both orders; parts that are not neighbours are farther apart than that
  for k in 1:length(p.parts)-1
    A, B = p.parts[k], p.parts[k + 1]
    x[first(B)] - x[last(A)] <= reach || continue
    for i in max(first(A), searchsortedfirst(x, x[first(B)] - reach)):last(A)
      for j in first(B):min(last(B), searchsortedlast(x, x[i] + reach))
        s += 2 * w[i] * w[j] * K((x[j] - x[i]) / g)
      end
    end
  end
  # the pairs with an observation that is not binned, exactly: in both orders with
  # binned ones, and among themselves from both ends
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
# finite values, or with `largest = true` the largest local minimizer whose
# criterion is within half of the lowest local minimum, if there is one. An interior
# choice is refined between its neighbours, to the relative accuracy rtol, and the
# refinement is kept only where it is no worse. Returns the details: the grid, the
# criterion values, the local minimizers and whether the choice is at an end of the
# grid.
function _gridminimize(f, hs; largest::Bool, rtol = 1e-6)
  cv = [f(h) for h in hs]
  m = length(hs)
  any(isfinite, cv) || throw(ArgumentError("the criterion is not finite at any bandwidth of the grid"))
  interior = [i for i in 2:m-1 if isfinite(cv[i]) && cv[i] < cv[i - 1] && cv[i] <= cv[i + 1]]
  i = if largest && !isempty(interior)
    lowest = minimum(cv[interior])
    last(filter(i -> cv[i] <= lowest + abs(lowest) / 2, interior))
  else
    argmin(map(v -> isfinite(v) ? v : Inf, cv))
  end
  atbound = i == 1 || i == m
  h = hs[i]
  if !atbound && isfinite(cv[i - 1]) && isfinite(cv[i + 1])
    t = _goldenmin(f, hs[i - 1], hs[i + 1]; rtol)
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
(2004), the rule prefers the largest local minimizer, since the criterion often has
spurious minimizers at small bandwidths: with rounded data, it falls without bound
as `h` shrinks. With compact kernels, however, clustered data give local minimizers
at the scale of the gaps between the clusters, where the criterion is far above its
lowest local minimum; for two normal samples 10 apart, the largest minimizer was at
13.3, with a criterion of -0.045 against -0.126 at 0.95. The rule therefore takes
the largest local minimizer whose criterion is within half of the lowest local
minimum, and refines it. Where the criterion is lowest at the lower end of the grid
and the data have no ties, the grid first extends downward by up to three decades,
as for lognormal data, whose minimizer lay at a fifth of the lower end. The details
of the selection record the grid, the criterion, all local minimizers and whether
the choice was at an end of the grid, and the rule warns then.

Polynomial kernels use exact fast sums. Gaussian kernels use exact double sums for
up to `exact` observations and otherwise the data binned once, on a grid of 100
points per smallest bandwidth, with the lagged products of the bin weights, as in
[`SheatherJones`](@ref), or binned again for each bandwidth where that grid would
need more than `2^20` points; binning moves the bandwidth by about `1e-5` relative. Cross-validation converges slowly, at the relative rate
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
  _check_mass(k, rule)
  sd, iqr = _sdiqr(d)
  _check_scale(sd, rule)
  n = effective_nobs(d)
  c = (243 * roughness(k) / (35 * _mu2(k)^2 * n))^(1 / 5)
  hmin, hmax = c * _robustscale(sd, iqr / 1.349) / 100, 4 * c * sd
  exact = nobs(d) <= rule.exact
  hs = exp.(range(log(hmin), log(hmax); length = rule.gridsize))
  h, details = _gridminimize(_lscvcriterion(d, k, exact, hmin), hs; largest = true)
  # the lowest criterion at the lower end: without ties, which make the criterion
  # fall without bound there, the grid extends downward, with as many points per
  # decade, to three decades lower or to the smallest gap between distinct values
  lo = max(hmin / 1000, minimum(diff(d.values)))
  if argmin(map(v -> isfinite(v) ? v : Inf, details.criterion)) == 1 && nunique(d) == nobs(d) && lo < hmin
    m = rule.gridsize + ceil(Int, (rule.gridsize - 1) * log(hmin / lo) / log(hmax / hmin))
    hs = exp.(range(log(lo), log(hmax); length = m))
    h, details = _gridminimize(_lscvcriterion(d, k, exact, lo), hs; largest = true)
  end
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
    CrossValidation(; gridsize = 40, exact = 1000)

Leave-one-out cross-validation for local polynomial regression: the bandwidth that
minimizes

    CV(h) = ∑ᵢ wᵢ (yᵢ - m̂₋ᵢ(xᵢ))² / ∑ᵢ wᵢ,   m̂₋ᵢ(xᵢ) = (m̂(xᵢ) - ℓᵢᵢ yᵢ) / (1 - ℓᵢᵢ),

where `m̂₋ᵢ(xᵢ)` is the fit at `xᵢ` without observation `i` and `ℓᵢᵢ` the weight
of `yᵢ` in the fit at `xᵢ`. The fits at all observations come from
[`kernelsums`](@ref), so that a bandwidth costs `O(n)` operations for polynomial
kernels. Gaussian kernels use exact sums for up to `exact` observations, and
otherwise, where it is faster, the data binned linearly on a grid of 50 points per
standard deviation of the kernel, with the sums from FFT convolutions; this changes
the criterion by about `1e-5` relative, which moves the bandwidth by less than its
refinement tolerance.

The grid has `gridsize` bandwidths spaced evenly on the log scale, from the smallest
one that leaves `p + 1` other distinct values in the window of each scored
observation, below which leave-one-out fits break down, to the range of the data,
or to twice that smallest one if it is larger. The criterion scores every
observation except isolated ones, whose distance to their `(p + 1)`-th nearest
other distinct value is far out among these distances, more than three
interquartile ranges above the upper quartile; they still enter the fits of the
others. A single isolated observation used to set the floor
of the grid for all: one `x` at 10 among 1000 uniform ones moved the default
regression bandwidth from 0.08 to 4.7. The global minimizer on the grid is refined
to a relative accuracy of `1e-3`, far below the statistical error of the bandwidth,
and the rule warns when it lies at the lower end of the grid. The details record
the grid, the criterion and the number of scored observations. Cross-validation
estimates the regression function, so it needs `derivative = 0`.
"""
struct CrossValidation <: BandwidthSelector
  gridsize::Int
  exact::Int

  function CrossValidation(gridsize::Integer, exact::Integer)
    gridsize >= 3 || throw(ArgumentError("gridsize must be at least 3, got $gridsize"))
    new(gridsize, exact)
  end
end

CrossValidation(; gridsize::Integer = 40, exact::Integer = 1000) = CrossValidation(gridsize, exact)

function selectbandwidth(rule::CrossValidation, d::SmoothingData, k::AbstractKernel;
                         degree::Integer = 1, derivative::Integer = 0, points = nothing,
                         vce = nothing, minobs::Integer = 0)
  # (the fits pass vce and minobs to every regression rule; this one does not use them)
  d.y === nothing && throw(ArgumentError("CrossValidation needs responses; use SmoothingData(x, y)"))
  _check_degree(degree, derivative)
  derivative == 0 || throw(ArgumentError(
    "CrossValidation chooses bandwidths for the regression function, derivative = 0"))
  p = Int(degree)
  nunique(d) >= p + 2 || throw(ArgumentError("CrossValidation needs at least $(p + 2) distinct values"))
  # the distance at which each distinct value has p + 1 other distinct values in its
  # window; isolated values are left out of the criterion and of the floor
  t = [knn_distance(d, x, p + 2; distinct = true) for x in d.values]
  ts = sort(t)
  q1, q3 = _unweightedquantile(ts, 0.25), _unweightedquantile(ts, 0.75)
  scoredvalue = t .<= q3 + 3 * (q3 - q1)   # Tukey's far-out fence
  scored = scoredvalue[d.valueindex]
  hmin = 1.01 * maximum(t[scoredvalue])
  # the grid reaches at least twice its floor, where every scored leave-one-out fit
  # holds p + 1 other distinct values with positive weight; with few distinct values
  # the floor can exceed the range, as for three values of x and a local linear fit
  hmax = max(d.x[end] - d.x[1], 2hmin)
  ws = SumsWorkspace()
  binned = nobs(d) > rule.exact
  crit(h) = _cvcriterion(d, k, h, p, ws, scored, binned)
  # the minimum is refined to 1e-3, far below the statistical error of the bandwidth,
  # with about half the evaluations of 1e-6
  h, details = _gridminimize(crit, exp.(range(log(hmin), log(hmax); length = rule.gridsize));
                             largest = false, rtol = 1e-3)
  details.atbound && h == details.grid[1] && @warn "CrossValidation: the criterion is smallest at " *
    "the lower end of its grid, h = $h, below which some leave-one-out fits fail; the minimum may lie lower"
  return BandwidthSelection(h, rule, (details..., scored = count(scored)))
end

function _cvcriterion(d, k, h, p, ws, scored = trues(nobs(d)), binned::Bool = false)
  S, T = _sumsyz(d, k, h, p, ws; binned)
  k0 = k(0.0)
  # chunks of a fixed size, run as tasks and summed in order, so that the criterion
  # does not depend on the number of threads
  parts = _fetchall([Threads.@spawn _cvpart(d, S, T, k0, p, r, scored)
                     for r in Iterators.partition(1:nobs(d), 8192)], Float64)
  return sum(parts) / sum(i -> scored[i] ? _weight(d.w, i) : 0.0, 1:nobs(d))
end

# the leave-one-out squared errors of the scored observations in r, or Inf if a fit
# there does not exist
function _cvpart(d, S, T, k0, p, r, scored)
  A, e = zeros(p + 1, p + 1), zeros(p + 1)
  s = 0.0
  for i in r
    scored[i] || continue
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
