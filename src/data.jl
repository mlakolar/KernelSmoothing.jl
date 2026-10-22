"""
    SmoothingData(x; weights = nothing)
    SmoothingData(x, y; weights = nothing)

A sample prepared for kernel smoothing: the observations `x`, sorted once,
together with the responses `y` of a regression and the weights, in the same
order. Ties are kept and recorded: the distinct values of `x` and where each one
starts in the sorted order.

Sorting once makes local questions cheap. [`window`](@ref) finds the
observations within a bandwidth of a point in `O(log n)` time,
[`nunique`](@ref) and [`weightsum`](@ref) count the distinct values and add up
the weights among them in `O(1)` time, and [`knn_distance`](@ref) finds the
distance to the `k`-th nearest observation in `O(log n + k)` time. Estimators and
bandwidth selectors will accept a `SmoothingData`, so a sample used for many fits
is sorted and checked once.

All values must be finite real numbers. Vectors containing `missing` are
rejected, so remove incomplete observations first. Weights must be nonnegative
with a positive sum; only their ratios matter, and observations with zero weight
are dropped. Values are stored as `Float64`, with `-0.0` stored as `0.0`.

The fields, all in sorted order:
- `x`: the observations
- `y`: the responses, or `nothing`
- `w`: the weights, or `nothing` when all weights are equal
- `perm`: the original indices, `x[i] == x_input[perm[i]]`
- `values`: the distinct values of `x`
- `starts`: observations `starts[j]:starts[j+1]-1` equal `values[j]`
- `valueindex`: `x[i] == values[valueindex[i]]`
- `cumw`: cumulative weights, `cumw[i + 1] = w[1] + ⋯ + w[i]` and `cumw[1] = 0`

# Examples
```julia
julia> d = SmoothingData([0.3, 0.1, 0.3, 0.2]);

julia> d.x, d.perm
([0.1, 0.2, 0.3, 0.3], [2, 4, 1, 3])

julia> nobs(d), nunique(d)
(4, 3)

julia> window(d, 0.25, 0.1)
2:4
```
"""
struct SmoothingData{Y<:Union{Nothing,Vector{Float64}},W<:Union{Nothing,Vector{Float64}}}
  x::Vector{Float64}        # sorted observations
  y::Y                      # responses in the order of x, or nothing
  w::W                      # weights in the order of x, or nothing for equal weights
  perm::Vector{Int}         # x[i] == x_input[perm[i]]
  values::Vector{Float64}   # distinct values of x
  starts::Vector{Int}       # x[starts[j]:starts[j+1]-1] .== values[j]
  valueindex::Vector{Int}   # x[i] == values[valueindex[i]]
  cumw::Vector{Float64}     # cumw[i + 1] = w[1] + ⋯ + w[i], cumw[1] = 0
  sumw::Float64             # total weight
  neff::Float64             # Kish's effective sample size
end

SmoothingData(x::AbstractVector; weights = nothing) = _smoothingdata(x, nothing, weights)
SmoothingData(x::AbstractVector, y::AbstractVector; weights = nothing) =
  _smoothingdata(x, y, weights)

function _smoothingdata(x, y, weights)
  _check_values(x, "x")
  isempty(x) && throw(ArgumentError("x must contain at least one observation"))
  if y !== nothing
    _check_values(y, "y")
    length(y) == length(x) ||
      throw(DimensionMismatch("y has $(length(y)) values for $(length(x)) observations"))
  end
  keep = eachindex(x)
  if weights !== nothing
    _check_values(weights, "weights")
    length(weights) == length(x) || throw(DimensionMismatch(
      "weights has $(length(weights)) values for $(length(x)) observations"))
    any(<(0), weights) && throw(ArgumentError("weights must be nonnegative"))
    keep = findall(>(0), weights)
    isempty(keep) && throw(ArgumentError("weights must have a positive sum"))
  end

  perm = _sortperm(x, keep)
  xs = [Float64(x[i]) + 0.0 for i in perm]   # adding 0.0 turns -0.0 into 0.0
  ys = y === nothing ? nothing : Float64[y[i] for i in perm]
  ws = weights === nothing ? nothing : Float64[weights[i] for i in perm]

  starts = [1]
  valueindex = Vector{Int}(undef, length(xs))
  for i in eachindex(xs)
    i > 1 && xs[i] != xs[i - 1] && push!(starts, i)
    valueindex[i] = length(starts)
  end
  values = xs[starts]
  push!(starts, length(xs) + 1)

  n = length(xs)
  cumw = ws === nothing ? collect(0.0:n) : _cumsum0(ws)
  neff = ws === nothing ? Float64(n) : cumw[end]^2 / sum(abs2, ws)
  return SmoothingData(xs, ys, ws, perm, values, starts, valueindex, cumw, cumw[end], neff)
end

# The stable sorting permutation of x[keep], as indices of x. Each observation gets
# a 128-bit key: the high half maps x[i] to an unsigned integer in the same order
# (flip the sign bit of positive numbers, all bits of negative ones), the low half
# is i, so sorting the keys orders by value and then by index. This is faster
# than sortperm because integers sort by radix sort.
function _sortperm(x, keep)
  keys = Vector{UInt128}(undef, length(keep))
  for (j, i) in enumerate(keep)
    v = Float64(x[i]) + 0.0   # -0.0 sorts with 0.0
    u = reinterpret(UInt64, v)
    keys[j] = UInt128(signbit(v) ? ~u : u | 0x8000000000000000) << 64 | UInt128(i)
  end
  sort!(keys)
  return [Int(k % UInt64) for k in keys]
end

# [0, w[1], w[1] + w[2], …] by compensated summation
function _cumsum0(w)
  c = Vector{Float64}(undef, length(w) + 1)
  c[1] = s = comp = 0.0
  for i in eachindex(w)
    y = w[i] - comp
    t = s + y
    comp = (t - s) - y
    s = t
    c[i + 1] = s
  end
  return c
end

function _check_values(v, name)
  Base.require_one_based_indexing(v)
  Missing <: eltype(v) && any(ismissing, v) &&
    throw(ArgumentError("$name contains missing values; remove the incomplete observations first"))
  eltype(v) <: Real || all(e -> e isa Real, v) ||
    throw(ArgumentError("$name must contain real numbers"))
  i = findfirst(!isfinite, v)
  i === nothing || throw(ArgumentError("$name[$i] = $(v[i]) is not finite"))
  return nothing
end

function Base.show(io::IO, d::SmoothingData)
  print(io, "SmoothingData(", nobs(d), " observations, ", nunique(d), " distinct values",
        d.w === nothing ? "" : ", weighted", d.y === nothing ? "" : ", with responses", ")")
end

nobs(d::SmoothingData) = length(d.x)

"""
    effective_nobs(d::SmoothingData)

Kish's effective sample size `(∑ w)^2 / ∑ w^2`, which equals the number of
observations when all weights are equal. Rules of thumb that scale with the
sample size should use it for weighted samples. (For weights that count repeated
observations, the effective sample size is instead the total weight.)
"""
effective_nobs(d::SmoothingData) = d.neff

"""
    nunique(d::SmoothingData)
    nunique(d::SmoothingData, r::AbstractUnitRange)

The number of distinct values of `x`, overall or among the sorted observations
`r`, such as a [`window`](@ref), in `O(1)` time.
"""
nunique(d::SmoothingData) = length(d.values)
nunique(d::SmoothingData, r::AbstractUnitRange{<:Integer}) =
  isempty(r) ? 0 : d.valueindex[last(r)] - d.valueindex[first(r)] + 1

"""
    weightsum(d::SmoothingData)
    weightsum(d::SmoothingData, r::AbstractUnitRange)

The total weight of the observations, overall or among the sorted observations
`r`, such as a [`window`](@ref), in `O(1)` time from the cumulative weights.
Without weights every observation counts one.
"""
weightsum(d::SmoothingData) = d.sumw
weightsum(d::SmoothingData, r::AbstractUnitRange{<:Integer}) =
  d.cumw[last(r) + 1] - d.cumw[first(r)]

"""
    window(d::SmoothingData, x0, h)

The range of sorted observations within the bandwidth `h` of the point `x0`: the
indices `i` with `abs((d.x[i] - x0) / h) <= 1`, the same test that decides
whether a compactly supported kernel is in reach. `length(window(d, x0, h))` is
the number of observations in the window and
[`nunique`](@ref)`(d, window(d, x0, h))` the number of distinct values. Takes
`O(log n)` time; the range is empty when no observation is that close.
"""
function window(d::SmoothingData, x0::Real, h::Real)
  _check_point(x0)
  h > 0 || throw(ArgumentError("bandwidth h must be positive, got $h"))
  x = d.x
  # (x[i] - x0) / h is nondecreasing in i, also after rounding
  lo = _firsttrue(i -> (x[i] - x0) / h >= -1, length(x))
  hi = _firsttrue(i -> (x[i] - x0) / h > 1, length(x)) - 1
  return lo:hi
end

# the first i in 1:n with pred(i), or n + 1, for a predicate that is false and then true
function _firsttrue(pred, n)
  lo, hi = 1, n + 1
  while lo < hi
    mid = (lo + hi) >>> 1
    pred(mid) ? (hi = mid) : (lo = mid + 1)
  end
  return lo
end

_check_point(x0) = isfinite(x0) || throw(ArgumentError("the point x0 must be finite, got $x0"))

"""
    knn_distance(d::SmoothingData, x0, k; distinct = false)

The distance from `x0` to its `k`-th nearest observation, counting tied
observations separately, or, with `distinct = true`, to its `k`-th nearest
distinct value. It is the smallest bandwidth `h` for which
[`window`](@ref)`(d, x0, h)` holds at least `k` observations (or `k` distinct
values). That makes it the floor for a bandwidth that must leave enough data in
every window, and the bandwidth of a `k`-nearest-neighbour estimator. It is zero
when `k` or more observations equal `x0`. Takes `O(log n + k)` time.
"""
function knn_distance(d::SmoothingData, x0::Real, k::Integer; distinct::Bool = false)
  _check_point(x0)
  v = distinct ? d.values : d.x
  1 <= k <= length(v) || throw(ArgumentError("k must be between 1 and the number of " *
    (distinct ? "distinct values" : "observations") * ", $(length(v)), got $k"))
  # take the k closest by merging outward from x0: v[l] < x0 <= v[r]
  r = searchsortedfirst(v, x0)
  l = r - 1
  δ = 0.0
  for _ in 1:k
    if r > length(v) || (l >= 1 && x0 - v[l] <= v[r] - x0)
      δ = x0 - v[l]
      l -= 1
    else
      δ = v[r] - x0
      r += 1
    end
  end
  return δ
end
