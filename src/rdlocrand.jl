## Local randomization inference for regression discontinuity designs

"""
    LocalRandomization

Randomization inference for a regression discontinuity design under local
randomization (Cattaneo, Frandsen and Titiunik 2015), as rdlocrand's `rdrandinf`:
in a window `[wl, wr]` around the cutoff, where treatment is taken to be assigned
as if at random, the observations above the cutoff are compared with those below
by a test statistic, whose distribution under the sharp null hypothesis
`Yᵢ(1) - Yᵢ(0) = τ₀` for every unit comes from permuting the treatment labels within
the window, with the number treated fixed. Fitted with
[`fit`](@ref)`(LocalRandomization, x, y; window)`.

`m.statistic` names the statistics, `m.observed` holds their values, `m.pvalue` the
randomization p-values and `m.asymptotic` the asymptotic p-values (for the
Kolmogorov–Smirnov statistic in small samples, its exact p-value); `coeftable(m)`
reports them.
"""
struct LocalRandomization
  cutoff::Float64
  window::NTuple{2,Float64}
  nulltau::Float64
  reps::Int
  ncontrol::Int
  ntreated::Int
  means::NTuple{2,Float64}       # of the outcomes below and above the cutoff in the window
  sds::NTuple{2,Float64}
  statistic::Vector{Symbol}
  observed::Vector{Float64}
  pvalue::Vector{Float64}
  asymptotic::Vector{Float64}
end

"""
    fit(LocalRandomization, x, y; window, cutoff = 0, statistic = :diffmeans, nulltau = 0,
        reps = 1000, rng = Random.default_rng())

Randomization inference in the `window`, a pair `(wl, wr)` with `wl < cutoff ≤ wr`
or a number `w` for `(cutoff - w, cutoff + w)`, for the sharp null hypothesis that the
treatment changes every outcome by `nulltau`; the observations with `x ≥ cutoff`
are treated. The statistic is

- `:diffmeans`, the difference in means `ȳ₁ - ȳ₀`, with the asymptotic p-value of
  `(ȳ₁ - ȳ₀) / √(s₁²/n₁ + s₀²/n₀)` against the normal distribution;
- `:ksmirnov`, the two-sample Kolmogorov–Smirnov statistic, with its exact p-value
  when `n₀ n₁ < 10000`, given the ties if there are any, and the limiting one
  otherwise, as R's `ks.test`;
- `:ranksum`, the standardized rank sum of the observations below the cutoff,
  `(W₀ - n₀(n + 1)/2) / √(n₀ n₁ (n + 1)/12)`, against the normal distribution;
- `:all`, the three.

The randomization p-value is the fraction of the `reps` random permutations of the
labels whose statistic is at least as large in absolute value as the observed one,
drawn with `rng`. `m.means` and `m.sds` hold the means and standard deviations of
the outcomes below and above the cutoff in the window.
"""
function fit(::Type{LocalRandomization}, x::AbstractVector, y::AbstractVector; window, cutoff::Real = 0,
             statistic::Symbol = :diffmeans, nulltau::Real = 0, reps::Integer = 1000,
             rng::AbstractRNG = Random.default_rng())
  length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
  _check_values(x, "x")
  _check_values(y, "y")
  c = Float64(cutoff)
  isfinite(c) || throw(ArgumentError("the cutoff must be finite, got $cutoff"))
  isfinite(nulltau) || throw(ArgumentError("nulltau must be finite, got $nulltau"))
  window isa Real || length(window) == 2 ||
    throw(ArgumentError("the window must be a number or a pair (wl, wr), got $(repr(window))"))
  wl, wr = window isa Real ? (c - window, c + window) : (Float64(window[1]), Float64(window[2]))
  wl < c <= wr || throw(ArgumentError("the window must contain the cutoff, got ($wl, $wr) for the cutoff $c"))
  statistic in (:diffmeans, :ksmirnov, :ranksum, :all) ||
    throw(ArgumentError("statistic must be :diffmeans, :ksmirnov, :ranksum or :all, got :$statistic"))
  reps >= 1 || throw(ArgumentError("reps must be positive, got $reps"))
  idx = findall(t -> wl <= t <= wr, x)
  treated = [x[i] >= c for i in idx]
  # under the null hypothesis the treated outcomes without the effect
  yy = [Float64(y[i]) - (x[i] >= c ? nulltau : 0.0) for i in idx]
  n1, n0 = count(treated), count(!, treated)
  n1 >= 2 && n0 >= 2 || throw(ArgumentError("the window needs at least two observations on each side"))
  stats = statistic === :all ? [:diffmeans, :ksmirnov, :ranksum] : [statistic]
  pre = _lrprep(yy)
  observed = [_lrstatistic(s, pre, treated) for s in stats]
  asymptotic = [_lrasymptotic(s, pre, treated, t) for (s, t) in zip(stats, observed)]
  # the permutation distribution, with the number treated fixed
  exceed = zeros(Int, length(stats))
  perm = copy(treated)
  for _ in 1:reps
    Random.shuffle!(rng, perm)
    for (k, s) in enumerate(stats)
      abs(_lrstatistic(s, pre, perm)) >= abs(observed[k]) * (1 - 1e-12) && (exceed[k] += 1)
    end
  end
  y0 = [Float64(y[i]) for i in idx if x[i] < c]
  y1 = [Float64(y[i]) for i in idx if x[i] >= c]
  return LocalRandomization(c, (wl, wr), Float64(nulltau), Int(reps), n0, n1, (mean(y0), mean(y1)),
                            (std(y0), std(y1)), stats, observed, exceed ./ reps, asymptotic)
end

# The outcomes, their order, their midranks and the ends of the runs of tied values
# in that order, computed once so that each statistic of a labelling takes one pass
# without allocations
function _lrprep(y)
  o = sortperm(y)
  n = length(y)
  r = zeros(n)
  ends = falses(n)
  i = 1
  while i <= n
    j = i
    while j < n && y[o[j + 1]] == y[o[i]]
      j += 1
    end
    for k in i:j
      r[o[k]] = (i + j) / 2
    end
    ends[j] = true
    i = j + 1
  end
  return (y = y, order = o, ranks = r, ends = ends)
end

function _lrstatistic(s::Symbol, pre, treated)
  y = pre.y
  n = length(y)
  n1 = count(treated)
  n0 = n - n1
  if s === :diffmeans
    s1 = s0 = 0.0
    for i in eachindex(y)
      treated[i] ? (s1 += y[i]) : (s0 += y[i])
    end
    return s1 / n1 - s0 / n0
  elseif s === :ksmirnov
    # sup |F₁ - F₀|, evaluated where the runs of tied values end
    i = j = 0
    D = 0.0
    for k in 1:n
      treated[pre.order[k]] ? (i += 1) : (j += 1)
      pre.ends[k] && (D = max(D, abs(i / n1 - j / n0)))
    end
    return D
  else
    # the standardized rank sum of the controls, with midranks for ties
    W0 = 0.0
    for i in eachindex(y)
      treated[i] || (W0 += pre.ranks[i])
    end
    return (W0 - n0 * (n + 1) / 2) / sqrt(n0 * n1 * (n + 1) / 12)
  end
end

function _lrasymptotic(s::Symbol, pre, treated, t)
  y = pre.y
  N = Distributions.Normal()
  if s === :diffmeans
    a, b = y[treated], y[.!treated]
    va = sum(abs2, a .- sum(a) / length(a)) / (length(a) - 1)
    vb = sum(abs2, b .- sum(b) / length(b)) / (length(b) - 1)
    return 2 * Distributions.ccdf(N, abs(t) / sqrt(va / length(a) + vb / length(b)))
  elseif s === :ranksum
    return 2 * Distributions.ccdf(N, abs(t))
  else
    n1, n0 = count(treated), count(!, treated)
    return n1 * n0 < 10_000 ? _ksexact(y[pre.order], n1, t) : _kslimit(sqrt(n1 * n0 / (n1 + n0)) * t)
  end
end

# P(D ≥ d) for the two-sample Kolmogorov–Smirnov statistic when n₁ of the sorted
# pooled values v are treated at random: the probability that the path of the labels
# in the order of v reaches |i/n₁ - j/n₀| ≥ d, with i treated and j controls among the
# first i + j values, at the end of a run of tied values, where the empirical
# distribution functions are evaluated (Schröer and Trenkler 1995). The mass of the
# paths is moved forward step by step, and the mass that reaches the boundary is
# summed, so that small p-values keep their relative accuracy.
function _ksexact(v, n1, d)
  n = length(v)
  n0 = n - n1
  # p[i + 1] is the probability of i treated among the first k, without an exit
  p, q = zeros(n1 + 1), zeros(n1 + 1)
  p[1] = 1.0
  tol = d * (1 - 1e-12)
  exit = 0.0
  for k in 1:n
    fill!(q, 0.0)
    for i in max(0, k - 1 - n0):min(k - 1, n1)
      j = k - 1 - i
      i < n1 && (q[i + 2] += p[i + 1] * (n1 - i) / (n - k + 1))
      j < n0 && (q[i + 1] += p[i + 1] * (n0 - j) / (n - k + 1))
    end
    if k == n || v[k + 1] != v[k]
      for i in max(0, k - n0):min(k, n1)
        if abs(i / n1 - (k - i) / n0) >= tol
          exit += q[i + 1]
          q[i + 1] = 0.0
        end
      end
    end
    p, q = q, p
  end
  return min(exit, 1.0)
end

# P(K ≥ t) for the Kolmogorov distribution, the limit of √(n₁n₀/n) D: from
# 1 - (√(2π)/t) ∑ₖ exp(-(2k-1)²π²/(8t²)) for t < 1, and from the alternating series
# 2 ∑ₖ (-1)^(k-1) exp(-2k²t²) for t ≥ 1, each of which converges in a few terms there
function _kslimit(t)
  t <= 0 && return 1.0
  if t < 1
    s = 0.0
    for k in 1:10
      s += exp(-(2k - 1)^2 * π^2 / (8 * t^2))
    end
    return clamp(1 - sqrt(2π) / t * s, 0.0, 1.0)
  end
  s = 0.0
  for k in 1:20
    s += 2 * (-1)^(k - 1) * exp(-2 * k^2 * t^2)
  end
  return clamp(s, 0.0, 1.0)
end

function Base.show(io::IO, m::LocalRandomization)
  print(io, "LocalRandomization(window (", m.window[1], ", ", m.window[2], "), ", m.ncontrol, " and ", m.ntreated,
        " observations, ", join(m.statistic, ", "), ")")
end

"""
    coeftable(m::LocalRandomization)

The results as `rdrandinf` reports them, a named tuple of columns with one row per
statistic: `statistic`, `observed`, `pvalue` (by randomization) and `asymptotic`.
"""
StatsAPI.coeftable(m::LocalRandomization) =
  (statistic = string.(m.statistic), observed = copy(m.observed), pvalue = copy(m.pvalue), asymptotic = copy(m.asymptotic))

nobs(m::LocalRandomization) = m.ncontrol + m.ntreated
