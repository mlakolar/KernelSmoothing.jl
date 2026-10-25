## Conditional density estimation

"""
    ConditionalDensity

A kernel estimate of the conditional density of `y` given `x` (Rosenblatt 1969;
Hyndman, Bashtannyk and Grunwald 1996; Fan, Yao and Tong 1996),

    f̂(y | x) = ∑ᵢ ℓᵢ(x) K((y - yᵢ)/b) / b,

where `K` is the kernel in `y` with bandwidth `b`, and `ℓᵢ(x)` are the weights of the
local polynomial fit of order `degree` at `x` with the kernel `k` and bandwidth `h`
in `x`: the estimate smooths the kernels of the responses as a local polynomial
regression smooths the responses. With `degree = 0`, the weights are those of
Nadaraya and Watson, `wᵢ k((xᵢ - x)/h) / ∑ⱼ wⱼ k((xⱼ - x)/h)`, nonnegative and summing
to one, so that `f̂(⋅ | x)` is a density, and [`conditional`](@ref)`(m, x)` returns it as
a [`KDE`](@ref). With `degree = 1` (Fan, Yao and Tong 1996), the bias at a boundary
of `x` is smaller, but the weights can be negative, and so can the estimate.

Fitted with [`fit`](@ref)`(ConditionalDensity, x, y)`; `pdf(m, y, x)` and `cdf(m, y, x)`
evaluate the estimate and its distribution function, `quantile(m, p, x)` the
conditional quantiles, and [`bandwidth`](@ref)`(m)` returns `(h, b)`.
"""
struct ConditionalDensity{D<:SmoothingData,K<:AbstractKernel,L<:AbstractKernel,S<:BandwidthSelection}
  data::D
  kernel::K
  ykernel::L
  degree::Int
  hx::Float64
  hy::Float64
  selection::S
end

"""
    fit(ConditionalDensity, x, y; bandwidth = ConditionalCV(), degree = 0,
        kernel = GaussianKernel(), ykernel = kernel, weights = nothing)

The kernel estimate of the conditional density of `y` given `x`, see
[`ConditionalDensity`](@ref). `bandwidth` is a pair `(h, b)` of bandwidths in `x`
and in `y`, or a rule; by default [`ConditionalCV`](@ref), which maximizes the
cross-validated likelihood.
"""
function fit(::Type{ConditionalDensity}, x::AbstractVector, y::AbstractVector; weights = nothing, kwargs...)
  return fit(ConditionalDensity, SmoothingData(x, y; weights); kwargs...)
end

function fit(::Type{ConditionalDensity}, d::SmoothingData; bandwidth = ConditionalCV(), degree::Integer = 0,
             kernel::AbstractKernel = GaussianKernel(), ykernel::AbstractKernel = kernel)
  _noclusters(d, "ConditionalDensity")
  d.y === nothing && throw(ArgumentError("a conditional density needs responses; use SmoothingData(x, y)"))
  _check_degree(degree, 0)
  _isnonnegative(ykernel) && isapprox(moment(ykernel, 0), 1; rtol = 1e-10) ||
    throw(ArgumentError("the kernel in y must be a probability density, got $ykernel"))
  bandwidth === nothing && (bandwidth = ConditionalCV())
  sel = if bandwidth isa ConditionalCV
    select_bandwidth(bandwidth, d, kernel, ykernel; degree)
  else
    (bandwidth isa Union{Tuple,AbstractVector} && length(bandwidth) == 2 && all(v -> v isa Real, bandwidth)) ||
      throw(ArgumentError("bandwidth must be a pair (h, b) of positive numbers or ConditionalCV(), got $(repr(bandwidth))"))
    BandwidthSelection([_bandwidths(bandwidth[1], 1), _bandwidths(bandwidth[2], 1)], nothing, NamedTuple())
  end
  return ConditionalDensity(d, kernel, ykernel, Int(degree), sel.h[1], sel.h[2], sel)
end

function Base.show(io::IO, m::ConditionalDensity)
  print(io, "ConditionalDensity(degree = ", m.degree, ", bandwidth = (", round(m.hx; sigdigits = 4), ", ",
        round(m.hy; sigdigits = 4), "), ", nobs(m.data), " observations)")
end

bandwidth(m::ConditionalDensity) = (m.hx, m.hy)
nobs(m::ConditionalDensity) = nobs(m.data)

# the window and the local polynomial weights at x, which sum to one
_cdweights(m::ConditionalDensity, x) = _smootherrow(m.data, m.kernel, Float64(x), m.hx, m.degree, 0)

"""
    pdf(m::ConditionalDensity, y, x)

The estimate `f̂(y | x)` of the conditional density of `y` given `x`; `NaN` where the
window of `x` does not identify the local fit.
"""
function pdf(m::ConditionalDensity, y::Real, x::Real)
  r, ℓ = _cdweights(m, x)
  d, K, b = m.data, m.ykernel, m.hy
  return sum(ℓ[k] * K((y - d.y[i]) / b) for (k, i) in enumerate(r); init = 0.0) / b
end

"""
    cdf(m::ConditionalDensity, y, x)

The estimate `∑ᵢ ℓᵢ(x) 𝒦((y - yᵢ)/b)` of the conditional distribution function, where
`𝒦` is the distribution function of the kernel in `y`.
"""
function cdf(m::ConditionalDensity, y::Real, x::Real)
  r, ℓ = _cdweights(m, x)
  d, K, b = m.data, m.ykernel, m.hy
  return sum(ℓ[k] * cdf(K, (y - d.y[i]) / b) for (k, i) in enumerate(r); init = 0.0)
end

"""
    conditional(m::ConditionalDensity, x)

The estimate of the conditional distribution of `y` given `x` of a local constant
fit (`degree = 0`), as a [`KDE`](@ref) of the responses with the weights `ℓᵢ(x)`: a
distribution, with `pdf`, `cdf`, `quantile`, `rand`, `mean` and the rest.
"""
function conditional(m::ConditionalDensity, x::Real)
  m.degree == 0 || throw(ArgumentError("the estimate of a local fit of degree $(m.degree) need not be a density; " *
    "conditional needs degree = 0"))
  r, ℓ = _cdweights(m, x)
  any(isnan, ℓ) && throw(ArgumentError("the window of x = $x holds no observation"))
  keep = [k for k in eachindex(ℓ) if ℓ[k] > 0]
  isempty(keep) && throw(ArgumentError("the window of x = $x holds no observation"))
  ys = m.data.y[r]
  return fit(KDE, ys[keep]; weights = ℓ[keep], kernel = m.ykernel, bandwidth = m.hy)
end

"""
    quantile(m::ConditionalDensity, p, x)

The quantile of order `p` of the estimated conditional distribution of `y` given
`x`, for a local constant fit: an estimate of the conditional quantile.
"""
quantile(m::ConditionalDensity, p::Real, x::Real) = quantile(conditional(m, x), p)

"""
    ConditionalCV(; criterion = :likelihood, gridsize = 7)

Cross-validation for the bandwidths `(h, b)` of a [`ConditionalDensity`](@ref) with
`degree = 0`. With `criterion = :likelihood` (the default), it maximizes the
leave-one-out log-likelihood

    ∑ᵢ wᵢ log f̂₋ᵢ(yᵢ | xᵢ),

where `f̂₋ᵢ` leaves observation `i` out of the numerator and the denominator of the
local constant estimate. With `criterion = :leastsquares`, it minimizes the
estimate of the integrated squared error of Hall, Racine and Li (2004),

    ∑ᵢ wᵢ ∫ f̂₋ᵢ(y | xᵢ)² dy / W - 2 ∑ᵢ wᵢ f̂₋ᵢ(yᵢ | xᵢ) / W,

whose cost grows as `n³`. The search starts on a grid of `gridsize × gridsize` pairs,
from an eighth to four times the normal-reference bandwidths `1.06 σ̂ n^(-1/6)` of each
variable, and refines the best one by the method of Nelder and Mead on the
logarithms of the bandwidths.
"""
struct ConditionalCV <: BandwidthSelector
  criterion::Symbol
  gridsize::Int

  function ConditionalCV(criterion::Symbol, gridsize::Integer)
    criterion in (:likelihood, :leastsquares) ||
      throw(ArgumentError("criterion must be :likelihood or :leastsquares, got :$criterion"))
    gridsize >= 2 || throw(ArgumentError("gridsize must be at least 2, got $gridsize"))
    new(criterion, gridsize)
  end
end

ConditionalCV(; criterion::Symbol = :likelihood, gridsize::Integer = 7) = ConditionalCV(criterion, gridsize)

function select_bandwidth(rule::ConditionalCV, d::SmoothingData, k::AbstractKernel, K::AbstractKernel = k; degree::Integer = 0)
  degree == 0 || throw(ArgumentError("ConditionalCV chooses the bandwidths of local constant estimates, degree = 0"))
  d.y === nothing && throw(ArgumentError("ConditionalCV needs responses; use SmoothingData(x, y)"))
  n = effective_nobs(d)
  nobs(d) >= 3 || throw(ArgumentError("cross-validation needs at least three observations"))
  sx = _std(d)
  sy = sqrt(_wvar(d.y, d.w))
  sx > 0 && sy > 0 || throw(ArgumentError("the data have no spread, so ConditionalCV cannot choose bandwidths"))
  # bandwidths as standard deviations of the kernels
  cx, cy = sqrt(moment(k, 2)), sqrt(moment(K, 2))
  h0, b0 = 1.06 * sx * n^(-1 / 6) / cx, 1.06 * sy * n^(-1 / 6) / cy
  crit(t) = _cdcv(rule.criterion, d, k, K, exp(t[1]), exp(t[2]))
  factors = exp.(range(log(1 / 8), log(4); length = rule.gridsize))
  grid = [[log(h0 * f), log(b0 * g)] for f in factors, g in factors]
  values = _pointwise(crit, vec(grid))
  any(isfinite, values) || throw(ArgumentError("the cross-validation criterion is not finite on the grid"))
  start = vec(grid)[argmin(map(v -> isfinite(v) ? v : Inf, values))]
  t, value = _neldermead(crit, start, log(factors[2] / factors[1]) / 2)
  return BandwidthSelection([exp(t[1]), exp(t[2])], rule, (criterion = value, start = exp.(start)))
end

# the weighted variance of y about its weighted mean, with Kish's correction
function _wvar(y, w)
  W = w === nothing ? Float64(length(y)) : sum(w)
  m = w === nothing ? sum(y) / W : sum(w .* y) / W
  s = w === nothing ? sum(abs2, y .- m) / W : sum(w .* abs2.(y .- m)) / W
  n = w === nothing ? W : W^2 / sum(abs2, w)
  return s * n / (n - 1)
end

# the criterion to minimize: minus the leave-one-out log-likelihood per unit weight,
# or the least-squares criterion of Hall, Racine and Li
function _cdcv(criterion, d, k, K, h, b)
  n = nobs(d)
  x, y = d.x, d.y
  # ∫ K((u - y₁)/b) K((u - y₂)/b) du / b² = (K⋆K)((y₁ - y₂)/b) / b
  Kc = criterion === :leastsquares ? convolve(K, K) : K
  kv = zeros(n)
  s = 0.0
  for i in 1:n
    wi = _weight(d.w, i)
    wi == 0 && continue
    den = num = 0.0
    for j in 1:n
      kv[j] = j == i ? 0.0 : _weight(d.w, j) * k((x[j] - x[i]) / h)
      den += kv[j]
      num += kv[j] * K((y[i] - y[j]) / b) / b
    end
    den > 0 || return Inf
    if criterion === :likelihood
      num > 0 || return Inf
      s -= wi * log(num / den)
    else
      sq = 0.0
      for j in 1:n
        kv[j] == 0 && continue
        t = 0.0
        for l in 1:n
          kv[l] == 0 || (t += kv[l] * Kc((y[j] - y[l]) / b))
        end
        sq += kv[j] * t
      end
      s += wi * (sq / (b * den^2) - 2 * num / den)
    end
  end
  return s / d.sumw
end

# Minimizes f over ℝ² by the method of Nelder and Mead from x0 with the initial
# simplex of size δ, until the values of the simplex agree to 1e-14 relative or its
# size falls below 1e-10
function _neldermead(f, x0, δ; maxiter = 400)
  P = [copy(x0), x0 .+ [δ, 0.0], x0 .+ [0.0, δ]]
  F = [f(p) for p in P]
  for _ in 1:maxiter
    o = sortperm(F)
    P, F = P[o], F[o]
    (abs(F[3] - F[1]) <= 1e-14 * (abs(F[1]) + 1e-300) || maximum(p -> maximum(abs, p .- P[1]), P) < 1e-10) && break
    c = (P[1] .+ P[2]) ./ 2
    r = c .+ (c .- P[3])
    fr = f(r)
    if fr < F[1]
      e = c .+ 2 .* (c .- P[3])
      fe = f(e)
      fe < fr ? ((P[3], F[3]) = (e, fe)) : ((P[3], F[3]) = (r, fr))
    elseif fr < F[2]
      P[3], F[3] = r, fr
    else
      q = fr < F[3] ? c .+ (r .- c) ./ 2 : c .+ (P[3] .- c) ./ 2
      fq = f(q)
      if fq < min(fr, F[3])
        P[3], F[3] = q, fq
      else   # shrink toward the best point
        for j in 2:3
          P[j] = P[1] .+ (P[j] .- P[1]) ./ 2
          F[j] = f(P[j])
        end
      end
    end
  end
  i = argmin(F)
  return P[i], F[i]
end
