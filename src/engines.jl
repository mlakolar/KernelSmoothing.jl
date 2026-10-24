"""
    SumEngine

Supertype of the algorithms that compute [`kernelsums`](@ref).
[`DirectSums`](@ref), [`WindowSums`](@ref) and [`FastSums`](@ref) are exact up to
rounding; [`BinnedSums`](@ref) approximates. [`isexact`](@ref) tells them apart.
All engines work on several points at once when Julia runs with several threads.
"""
abstract type SumEngine end

"""
    DirectSums()

Sums over all observations for every point: `O(n m)` time for `n` observations
and `m` points, for any kernel. The other engines are tested against it.
"""
struct DirectSums <: SumEngine end

"""
    WindowSums()

Sums over the observations in each point's window, which a search in the sorted
data finds: `O(m log n)` time plus the total size of the windows. For compactly
supported kernels; it adds the same terms in the same order as
[`DirectSums`](@ref), so the sums are identical.
"""
struct WindowSums <: SumEngine end

"""
    FastSums()

Exact sums for polynomial kernels in `O(n + m log m)` time, independent of the
bandwidth. On each polynomial piece of the kernel the sum is a combination of
power sums of the observations, so prefix sums answer every point in constant
time (fast sum updating; Seifert et al. 1994, Langrené and Warin 2019).

Power sums about one far-away centre lose all accuracy when the data are far from
zero or the bandwidth is small compared with their range. Here the data are split
into blocks at most one window wide and the power sums are taken about the midpoint
of the observations in each block. The rounding error grows with the width of the blocks and the
highest power of the sums, the degree of the kernel plus `kmax`, so that the blocks
narrow as the power grows. The error is then at most
about `1e-13` times the total absolute weight of the observations near the point,
whatever the location and scale of the data and the bandwidth; for kernels of degree `d` with
`d + kmax ≤ 6`, such as the named kernels other than the triweight with
`kmax ≤ 2`, the blocks are one window wide. A sum that is much smaller than that,
because the kernel nearly vanishes at every observation in the window, has a larger
relative error.

With a bandwidth per point, the blocks are sized for the smallest bandwidth, and a
point with a larger bandwidth spans more of them.
"""
struct FastSums <: SumEngine end

"""
    BinnedSums(; binsperh = 50)

Approximate sums from linearly binned data: each observation's value is split
between the two nearest points of a grid with spacing `h / binsperh`, in
proportion to its distance from them, and the sums are computed over the occupied
grid points. Gaussian kernels are cut off at 12 standard deviations. Useful for
Gaussian kernels with many observations, where [`DirectSums`](@ref) is slow. With
a bandwidth per point, the grid spacing uses the smallest bandwidth.

Binning replaces each observation's term by its linear interpolation between grid
points, so the relative error is of order `binsperh^-2` for continuous kernels,
about `1e-3` at the default. For kernels that jump, such as the uniform kernel and
the derivatives of the triangular and Epanechnikov kernels, the error comes from
the observations within one grid spacing of a jump and decreases much more slowly
as `binsperh` grows.
"""
struct BinnedSums <: SumEngine
  binsperh::Int   # grid points per bandwidth

  function BinnedSums(binsperh::Integer)
    binsperh >= 1 || throw(ArgumentError("binsperh must be at least 1, got $binsperh"))
    new(binsperh)
  end
end

BinnedSums(; binsperh::Integer = 50) = BinnedSums(binsperh)

"""
    isexact(engine::SumEngine)

Whether the engine computes the kernel sums exactly, up to rounding. Inference
should use exact engines.
"""
isexact(::SumEngine) = true
isexact(::BinnedSums) = false

"""
    SumsWorkspace()

Buffers that [`kernelsums!`](@ref) can reuse: the values `wᵢ zᵢ`, the order of the
points, the blocks and power sums of [`FastSums`](@ref) and the grid of
[`BinnedSums`](@ref). Passing the same workspace to repeated calls, as in a
bandwidth search, saves allocating them every time. A workspace must not be used
by two calls at the same time.
"""
mutable struct SumsWorkspace
  vals::Vector{Float64}       # wᵢ zᵢ in the sorted order of the data
  order::Vector{Int}          # the points in increasing order
  prefixes::Vector{Matrix{Float64}}   # FastSums: power sums within blocks, one per vector of values
  starts::Vector{Int}         # FastSums: the first observation of each block
  centers::Vector{Float64}    # FastSums: the centre of each block
  blockof::Vector{Int}        # FastSums: the block of each observation
  grid::Vector{Float64}       # BinnedSums: occupied grid points
  gridvals::Vector{Float64}   # BinnedSums: their binned values
  gridindex::Vector{Int}      # BinnedSums: their indices on the grid
  ones::Vector{Float64}       # values of one, for sums without weights and z

  SumsWorkspace() = new(Float64[], Int[], Matrix{Float64}[], Int[], Float64[],
                        Int[], Float64[], Float64[], Int[], Float64[])
end

Base.show(io::IO, ::SumsWorkspace) = print(io, "SumsWorkspace()")

const _Bandwidths = Union{Real,AbstractVector{<:Real}}

"""
    kernelsums(d::SmoothingData, k, h, points; kmax = 0, z = nothing, engine, workspace)
    kernelsums(d::SmoothingData, k, h; kmax = 0, z = nothing, engine, workspace)

The kernel sums

    S[r + 1, j] = ∑ᵢ wᵢ zᵢ k(uᵢⱼ) uᵢⱼ^r,   uᵢⱼ = (xᵢ - points[j]) / hⱼ,   r = 0, …, kmax,

over the observations `xᵢ` of `d` with weights `wᵢ`, as a
`(kmax + 1) × length(points)` matrix. The bandwidth `h` is a number, or a vector
with one bandwidth per point, as pointwise bandwidth selectors give. Without
`points` the sums are evaluated at the observations themselves, `d.x`, as
cross-validation needs. `z` holds one value per observation in the sorted order of
`d`, such as the responses `d.y`; by default every `zᵢ = 1`.

Estimators are built from these sums. For a symmetric kernel,
`S[1, j] / (h * d.sumw)` is the kernel density estimate at `points[j]`; with
`kmax = 2p`, and with `z = d.y` and `kmax = p`, the sums give the normal
equations of a local polynomial fit of order `p` in the scaled variable `u`.

`engine` chooses the algorithm. By default it is the faster exact one: for
polynomial kernels [`WindowSums`](@ref) when the windows hold few observations
and [`FastSums`](@ref) otherwise, by a model of their costs; for Gaussian kernels
sums over the observations within 38.62 standard deviations, beyond which the
kernel underflows to zero, so that they equal [`DirectSums`](@ref) exactly; and
[`DirectSums`](@ref) for other kernels. [`BinnedSums`](@ref) approximates. A
[`SumsWorkspace`](@ref) passed as `workspace` provides reusable buffers.

# Examples
```julia
julia> d = SmoothingData([0.0, 0.5, 1.0]);

julia> kernelsums(d, EpanechnikovKernel(), 1.0, [0.5]; kmax = 2)
3×1 Matrix{Float64}:
 1.875
 0.0
 0.28125
```
"""
function kernelsums(d::SmoothingData, k::AbstractKernel, h::_Bandwidths,
                    points::AbstractVector{<:Real}; kmax::Integer = 0, kwargs...)
  kmax >= 0 || throw(ArgumentError("kmax must be nonnegative, got $kmax"))
  out = Matrix{Float64}(undef, kmax + 1, length(points))
  return kernelsums!(out, d, k, h, points; kwargs...)
end

kernelsums(d::SmoothingData, k::AbstractKernel, h::_Bandwidths; kwargs...) =
  kernelsums(d, k, h, d.x; kwargs...)

"""
    kernelsums!(out, d::SmoothingData, k, h, points; z = nothing, engine, workspace)

[`kernelsums`](@ref) into the `Float64` matrix `out`, whose number of rows sets
`kmax = size(out, 1) - 1`.
"""
function kernelsums!(out::AbstractMatrix{Float64}, d::SmoothingData, k::AbstractKernel,
                     h::_Bandwidths, points::AbstractVector{<:Real}; z = nothing,
                     engine::Union{Nothing,SumEngine} = nothing,
                     workspace::SumsWorkspace = SumsWorkspace())
  Base.require_one_based_indexing(out, points)
  all(isfinite, points) || throw(ArgumentError("points must be finite"))
  size(out, 1) >= 1 || throw(DimensionMismatch("out must have at least one row, for kmax = 0"))
  size(out, 2) == length(points) || throw(DimensionMismatch(
    "out must have length(points) = $(length(points)) columns, got size $(size(out))"))
  # the engines write the columns of out from several tasks, which only a strided
  # array allows; others, such as a sparse matrix, receive a copy
  out isa StridedMatrix || return copyto!(out, kernelsums!(Matrix{Float64}(undef, size(out)), d, k, h, points;
                                                         z, engine, workspace))
  hs = _bandwidths(h, length(points))
  vals = _sumvalues!(workspace, d, z)
  isempty(points) && return out
  kmax = size(out, 1) - 1
  e = engine === nothing ? _choose_engine(k, d.x, hs, points, kmax, _sortedorder!(workspace, points)) : engine
  _kernelsums!(out, e, d.x, vals, k, hs, points, kmax, workspace)
  return out
end

# one bandwidth as a Float64, or a vector of them, checked
function _bandwidths(h::Real, m)
  isfinite(h) && h > 0 || throw(ArgumentError("bandwidth h must be positive and finite, got $h"))
  return Float64(h)
end

function _bandwidths(h::AbstractVector{<:Real}, m)
  Base.require_one_based_indexing(h)
  length(h) == m || throw(DimensionMismatch("h has $(length(h)) bandwidths for $m points"))
  all(b -> isfinite(b) && b > 0, h) ||
    throw(ArgumentError("bandwidths must be positive and finite"))
  return Vector{Float64}(h)
end

_bw(h::Float64, j) = h
_bw(h::Vector{Float64}, j) = @inbounds h[j]
_hmin(h::Float64) = h
_hmin(h::Vector{Float64}) = minimum(h)

# wᵢ zᵢ in the sorted order of d
function _sumvalues!(ws::SumsWorkspace, d::SmoothingData, z)
  n = nobs(d)
  if z === nothing
    d.w === nothing || return d.w
    # filled once for repeated calls, as in a bandwidth search; nothing else writes it
    length(ws.ones) == n || fill!(resize!(ws.ones, n), 1.0)
    return ws.ones
  end
  _check_values(z, "z")
  length(z) == n || throw(DimensionMismatch("z has $(length(z)) values for $n observations"))
  vals = resize!(ws.vals, n)
  d.w === nothing ? (vals .= z) : (vals .= d.w .* z)
  return vals
end

# The faster exact engine for a polynomial kernel, from a model of the costs in
# nanoseconds measured on Apple M-series machines: a windowed sum costs a term per
# observation in each window, 0.8 for the named kernels and 2.7 for others, whose
# pieces are vectors; fast sums cost 0.65 per power sum of each observation, and per
# point a constant plus the pieces of the kernel times the blocks its window spans.
# Both are estimated from 32 of the points, evenly spaced in their sorted `order`, so
# that a grid repeated in the points does not put the whole sample in one place.
_choose_engine(::AbstractKernel, xs, h, points, kmax, order) = DirectSums()
_choose_engine(k::AbstractKernel, xs, h, points, kmax) = _choose_engine(k, xs, h, points, kmax, sortperm(points))

function _choose_engine(k::PolynomialKernel, xs, h, points, kmax, order)
  m, n = length(points), length(xs)
  b, c = pieces(k)
  lo, hi = first(b), last(b)
  np, len = length(c), maximum(length, c)
  f = _blockfactor(len - 1 + kmax)
  h0 = _hmin(h)
  sample = view(order, 1:max(1, m ÷ 32):m)
  terms = blocks = 0.0
  for j in sample
    p, hj = points[j], _bw(h, j)
    terms += _firsttrue(i -> (xs[i] - p) / hj > hi, n) - _firsttrue(i -> (xs[i] - p) / hj >= lo, n)
    blocks += f * hj / h0 + 1
  end
  windowcost = (k isa NamedPolynomialKernel ? 0.8 : 2.7) * (1 + kmax / 20) * terms / length(sample) * m
  fastcost = 0.65 * n * (len + kmax) + (45 * length(sample) + 3.5 * np * len * blocks) * (1 + kmax / 4) / length(sample) * m
  return windowcost <= fastcost ? WindowSums() : FastSums()
end

# Gaussian kernels: the sums over the observations within the radius where the
# kernel underflows to zero, which equal the direct sums term by term, when the
# windows hold fewer than about half of the observations
function _choose_engine(k::GaussianFamily, xs, h, points, kmax, order)
  m, n = length(points), length(xs)
  r = _zeroradius(k)
  sample = view(order, 1:max(1, m ÷ 32):m)
  terms = 0
  for j in sample
    p, hj = points[j], _bw(h, j)
    terms += _firsttrue(i -> (xs[i] - p) / hj > r, n) - _firsttrue(i -> (xs[i] - p) / hj >= -r, n)
  end
  return terms / length(sample) <= n / 2 ? _ZeroWindowSums() : DirectSums()
end

# The windowed sums of a Gaussian kernel over |u| ≤ _zeroradius(k), outside which it
# is exactly zero; the automatic choice only, since they are DirectSums by another
# route
struct _ZeroWindowSums <: SumEngine end

_kernelsums!(out, ::_ZeroWindowSums, xs, vals, k, h, points, kmax, ws) =
  _windowsums!(out, xs, vals, k, h, points, Val(kmax), -_zeroradius(k), _zeroradius(k), _sortedorder!(ws, points))

# exp(-z²/2) is zero in Float64 for |z| > 38.604, which holds for |u| ≥ 38.62 σ
_zeroradius(k::GaussianFamily) = 38.62 * first(_scale_order(k))

# the interval outside which a kernel is exactly zero in Float64: its support, or for
# Gaussian kernels the interval where the exponential has not underflowed
_zerosupport(k::AbstractKernel) = support(k)
_zerosupport(k::GaussianFamily) = (r = _zeroradius(k); (-r, r))

# The factor by which the blocks of fast sums are narrower than a window, for powers
# up to s. Measured over the named kernels, their convolutions and up to s = 19, the
# relative error of blocks one window wide grows about as 2.6^s eps, to 5e-14 at
# s = 6 and 8e-7 at s = 19, while halving the blocks takes off about s / 1.4 bits;
# with these factors it stayed below 5e-14.
_blockfactor(s) = 2.0^cld(max(s - 6, 0), 7)

# the indices of the points in increasing order of the points; an order already in
# the workspace that sorts the points, as from the engine choice, is kept
function _sortedorder!(ws::SumsWorkspace, points)
  order = ws.order
  length(order) == length(points) && issorted(view(points, order)) && return order
  resize!(order, length(points))
  return issorted(points) ? copyto!(order, eachindex(points)) : sortperm!(order, points)
end

const _MINCHUNK = 256   # the fewest points worth a task of their own

# f(chunk) for the chunks of the index vector `order`, as parallel tasks when there
# are enough points; with one thread the chunks run one after the other
function _foreachchunk(f, order)
  m = length(order)
  nchunks = min(4 * Threads.nthreads(), cld(m, _MINCHUNK))
  if nchunks <= 1
    f(order)
  else
    _fetchall([Threads.@spawn f(chunk) for chunk in Iterators.partition(order, cld(m, nchunks))])
  end
  return nothing
end

# [f(p) for p in points] for an f that is expensive at each point, one task per point,
# which run in parallel when Julia runs with several threads; the results have the
# type that inference gives f, so that callers infer too
_pointwise(f, points) = _fetchall([Threads.@spawn f(p) for p in points], Base.promote_op(f, eltype(points)))

# The results of the tasks, once all of them have finished, as a vector with element
# type T where that is a type their values can have. An error in a task is thrown as
# itself, not wrapped in a TaskFailedException, so that its type does not depend on
# whether the work ran in tasks. Anything else that interrupts the wait, such as
# Ctrl-C, is thrown at once.
function _fetchall(tasks, T::Type = Any)
  for t in tasks
    try
      wait(t)
    catch e
      e isa TaskFailedException || rethrow()
    end
  end
  for t in tasks
    istaskfailed(t) && throw(t.result)
  end
  T === Union{} && return [fetch(t) for t in tasks]
  return T[fetch(t)::T for t in tasks]
end

# The first i with pred(i), or n + 1, for a predicate that is false and then true on
# 1:n, starting from the answer for the previous point. With one bandwidth the
# answers never decrease as the points increase; with a bandwidth per point they can.
_search(pred, start, n, ::Float64) = _gallop(pred, start, n)
_search(pred, start, n, ::Vector{Float64}) = _gallop2(pred, start, n)

# the first i ≥ start with pred(i), or n + 1; galloping from start takes
# O(log(answer - start)) evaluations
function _gallop(pred, start, n)
  start > n && return n + 1
  pred(start) && return start
  lo, step = start, 1   # pred(lo) is false
  hi = lo + step
  while hi <= n && !pred(hi)
    lo = hi
    step *= 2
    hi = lo + step
  end
  hi = min(hi, n + 1)   # pred(hi) is true, or hi == n + 1
  while hi - lo > 1
    mid = (lo + hi) >>> 1
    pred(mid) ? (hi = mid) : (lo = mid)
  end
  return hi
end

# the first i with pred(i), or n + 1, galloping from start in either direction
function _gallop2(pred, start, n)
  start = clamp(start, 1, n + 1)
  start <= n && !pred(start) && return _gallop(pred, start, n)
  hi, step = start, 1   # pred(hi) is true, or hi == n + 1
  lo = hi - step
  while lo >= 1 && pred(lo)
    hi = lo
    step *= 2
    lo = hi - step
  end
  lo = max(lo, 0)       # pred(lo) is false, or lo == 0
  while hi - lo > 1
    mid = (lo + hi) >>> 1
    pred(mid) ? (hi = mid) : (lo = mid)
  end
  return hi
end

# acc .+ (v, v u, v u^2, …) for a tuple of accumulators, kept in registers
@inline _accum(acc::Tuple{}, v, u) = ()
@inline _accum(acc::Tuple, v, u) = (acc[1] + v, _accum(Base.tail(acc), v * u, u)...)

@inline function _store!(out, j, acc)
  @inbounds for r in eachindex(acc)
    out[r, j] = acc[r]
  end
  return out
end

# Julia stops optimizing tuples of more than about 32 elements, where the sums with
# kmax = 32 took 60 times as long as with kmax = 31; beyond this many powers, the
# accumulators are the column of out itself, with the terms added in the same order
const _MAXTUPLE = 16

# ∑ vals[i] k(uᵢ) (1, uᵢ, …, uᵢ^K) over i in r into out[:, j], uᵢ = (xs[i] - p) / hj;
# with `finite`, u is kept finite, so that a u that overflows to ±Inf, where the
# kernel vanishes, adds zero terms instead of NaN (0 · u stays 0), without a branch
@inline function _pointsums!(out, j, xs, vals, k, p, hj, r, ::Val{K}, finite::Bool) where K
  if K < _MAXTUPLE
    acc = ntuple(_ -> 0.0, Val(K + 1))
    @inbounds for i in r
      u = (xs[i] - p) / hj
      v = vals[i] * k(u)
      finite && (u = clamp(u, -floatmax(), floatmax()))
      acc = _accum(acc, v, u)
    end
    _store!(out, j, acc)
  else
    @inbounds for s in 1:K+1
      out[s, j] = 0.0
    end
    @inbounds for i in r
      u = (xs[i] - p) / hj
      v = vals[i] * k(u)
      finite && (u = clamp(u, -floatmax(), floatmax()))
      for s in 1:K+1
        out[s, j] += v
        v *= u
      end
    end
  end
  return out
end


## Direct and windowed sums

_kernelsums!(out, ::DirectSums, xs, vals, k, h, points, kmax, ws) =
  _directsums!(out, xs, vals, k, h, points, Val(kmax))

function _directsums!(out, xs, vals, k, h, points, kmax::Val)
  _foreachchunk(eachindex(points)) do chunk
    for j in chunk
      p, hj = points[j], _bw(h, j)
      # u overflows only where the farthest observation is beyond floatmax bandwidths
      if isfinite(max(abs(xs[1] - p), abs(xs[end] - p)) / hj)
        _pointsums!(out, j, xs, vals, k, p, hj, eachindex(xs), kmax, false)
      else
        _pointsums!(out, j, xs, vals, k, p, hj, eachindex(xs), kmax, true)
      end
    end
  end
  return out
end

function _kernelsums!(out, ::WindowSums, xs, vals, k, h, points, kmax, ws)
  lo, hi = support(k)
  isfinite(lo) && isfinite(hi) || throw(ArgumentError(
    "WindowSums needs a compactly supported kernel; use DirectSums or BinnedSums for $k"))
  return _windowsums!(out, xs, vals, k, h, points, Val(kmax), lo, hi, _sortedorder!(ws, points))
end

# sums over the observations with lo ≤ u ≤ hi, which is all of them where k is
# nonzero when (lo, hi) is the support of k; the points are taken in `order`
function _windowsums!(out, xs, vals, k, h, points, kmax::Val, lo, hi, order)
  n = length(xs)
  _foreachchunk(order) do chunk
    a = b = 1   # the first index with u ≥ lo, the first index with u > hi
    for j in chunk
      p, hj = points[j], _bw(h, j)
      a = _search(i -> (xs[i] - p) / hj >= lo, a, n, h)
      b = _search(i -> (xs[i] - p) / hj > hi, b, n, h)
      _pointsums!(out, j, xs, vals, k, p, hj, a:b-1, kmax, false)
    end
  end
  return out
end

# windowed sums for two vectors of values in one pass, sharing the windows and the
# kernel values; each output gets the terms of a separate pass, in the same order
function _windowsums2!(S, T, xs, v1, v2, k, h, points, ::Val{K1}, ::Val{K2}, lo, hi, order) where {K1,K2}
  n = length(xs)
  _foreachchunk(order) do chunk
    a = b = 1
    for j in chunk
      p, hj = points[j], _bw(h, j)
      a = _search(i -> (xs[i] - p) / hj >= lo, a, n, h)
      b = _search(i -> (xs[i] - p) / hj > hi, b, n, h)
      acc1, acc2 = ntuple(_ -> 0.0, Val(K1 + 1)), ntuple(_ -> 0.0, Val(K2 + 1))
      @inbounds for i in a:b-1
        u = (xs[i] - p) / hj
        kv = k(u)
        acc1 = _accum(acc1, v1[i] * kv, u)
        acc2 = _accum(acc2, v2[i] * kv, u)
      end
      _store!(S, j, acc1)
      _store!(T, j, acc2)
    end
  end
  return S, T
end

# The kernel sums at the observations with and without the responses,
# S = kernelsums(d, k, h; kmax = 2p) and T = kernelsums(d, k, h; kmax = p, z = d.y),
# in one pass where the engine allows it
function _sumsyz(d::SmoothingData, k::AbstractKernel, h, p, ws::SumsWorkspace)
  n = nobs(d)
  S, T = Matrix{Float64}(undef, 2p + 1, n), Matrix{Float64}(undef, p + 1, n)
  order = _sortedorder!(ws, d.x)
  e = _choose_engine(k, d.x, h, d.x, 2p, order)
  vS, vT = _sumvalues!(ws, d, nothing), _sumvalues!(ws, d, d.y)
  if e isa FastSums
    _fastsums!((S, T), d.x, (vS, vT), k, h, d.x, (2p, p), ws)
  elseif e isa Union{WindowSums,_ZeroWindowSums} && 2p < _MAXTUPLE
    lo, hi = e isa WindowSums ? support(k) : _zerosupport(k)
    _windowsums2!(S, T, d.x, vS, vT, k, h, d.x, Val(2p), Val(p), lo, hi, order)
  else
    kernelsums!(S, d, k, h, d.x; engine = e, workspace = ws)
    kernelsums!(T, d, k, h, d.x; z = d.y, engine = e, workspace = ws)
  end
  return S, T
end


## Fast sums

_kernelsums!(out, ::FastSums, xs, vals, k::AbstractKernel, h, points, kmax, ws) =
  throw(ArgumentError("FastSums needs a polynomial kernel; use DirectSums or BinnedSums for $k"))

function _kernelsums!(out, ::FastSums, xs, vals, k::PolynomialKernel, h, points, kmax, ws)
  _fastsums!((out,), xs, (vals,), k, h, points, (kmax,), ws)
  return out
end

# Fast sums for several vectors of values at once, each into its own output with its
# own kmax, sharing the blocks, the searches and the polynomials of each piece, as
# cross-validation needs for the sums with and without the responses
function _fastsums!(outs::NTuple{N,Any}, xs, valss::NTuple{N,Any}, k::PolynomialKernel, h, points,
                    kmaxs::NTuple{N,Int}, ws) where N
  b, c = pieces(k)
  order = _sortedorder!(ws, points)
  h0 = _hmin(h)
  K = maximum(kmaxs)
  len0 = maximum(length, c)
  smax = len0 - 1 + K   # the highest power of t needed
  # block width: a window at the smallest bandwidth, narrower for wide supports and
  # high powers, whose rounding grows with the width
  Δ = (last(b) - first(b)) * h0 / _blockfactor(smax)
  # below the resolution of the data, blocks would hold single values; sum directly
  if !(Δ > 64 * eps(max(abs(xs[1]), abs(xs[end]))))
    for l in 1:N
      _windowsums!(outs[l], xs, valss[l], k, h, points, Val(kmaxs[l]), first(b), last(b), order)
    end
    return outs
  end
  starts, centers, blockof = _blocks!(ws, xs, Δ)
  prefixes = ntuple(l -> _prefix!(ws, l, xs, valss[l], starts, centers, h0, len0 - 1 + kmaxs[l]), N)
  n, np = length(xs), length(c)
  _foreachchunk(order) do chunk
    ptr = ones(Int, np + 1)                         # piece π holds observations ptr[π]:ptr[π+1]-1
    L = ntuple(l -> zeros(size(prefixes[l], 1)), N)  # power sums of t over part of a block
    g = zeros(smax + 1)                             # a piece of k times u^r, in powers of t
    for j in chunk
      p, hj = points[j], _bw(h, j)
      ρ = h0 / hj   # u = ρ t + δ with t = (x - centre) / h0
      # piece π covers b[π] ≤ u < b[π+1], and the last piece also u = b[end], as in
      # k(u); the predicates capture only variables assigned once, so nothing is boxed
      for π in 1:np
        ptr[π] = _search(i -> (xs[i] - p) / hj >= b[π], ptr[π], n, h)
      end
      ptr[np + 1] = _search(i -> (xs[i] - p) / hj > b[np + 1], ptr[np + 1], n, h)
      for l in 1:N, r in 1:kmaxs[l]+1
        outs[l][r, j] = 0.0
      end
      for π in 1:np
        i1, i2 = ptr[π], ptr[π + 1] - 1
        i1 <= i2 || continue
        m = _mid(b, π)
        @inbounds for β in blockof[i1]:blockof[i2]
          s, e = max(i1, starts[β]), min(i2, starts[β + 1] - 1)
          for l in 1:N
            P, Ll = prefixes[l], L[l]
            for q in eachindex(Ll)
              Ll[q] = P[q, e] - (s > starts[β] ? P[q, s - 1] : 0.0)
            end
          end
          δ = (centers[β] - p) / hj
          len = length(c[π])
          for q in 1:len
            g[q] = c[π][q]
          end
          _shift!(g, len, δ - m)        # the piece, a polynomial in u - m, in powers of ρ t
          ρ == 1 || _scale!(g, len, ρ)  # in powers of t
          for r in 1:K+1
            for l in 1:N
              r <= kmaxs[l] + 1 || continue
              Ll = L[l]
              acc = 0.0
              for q in 1:len
                acc = muladd(g[q], Ll[q], acc)
              end
              outs[l][r, j] += acc
            end
            r <= K && (len = _mulaffine!(g, len, ρ, δ))
          end
        end
      end
    end
  end
  return outs
end

# Splits the sorted xs into blocks of width Δ and returns the first index and the
# centre of each block and the block of each observation; the arrays live in the
# workspace. The centre is the midpoint of the block's own observations: the midpoint
# of its cell of the grid can lie half a cell from them, and when the bandwidth, and
# with it the cell, is much wider than the data, the power sums about it cancel.
function _blocks!(ws::SumsWorkspace, xs, Δ)
  n = length(xs)
  starts, centers = empty!(ws.starts), empty!(ws.centers)
  blockof = resize!(ws.blockof, n)
  q0 = -1.0
  @inbounds for i in 1:n
    q = floor((xs[i] - xs[1]) / Δ)
    if i == 1 || q != q0
      push!(starts, i)
      q0 = q
    end
    blockof[i] = length(starts)
  end
  push!(starts, n + 1)
  @inbounds for β in 1:length(starts)-1
    s, e = starts[β], starts[β + 1] - 1
    push!(centers, xs[s] + (xs[e] - xs[s]) / 2)
  end
  return starts, centers, blockof
end

# The power sums prefix[q, i] = ∑ vals[l] t_l^(q-1), q = 1, …, smax + 1, over the
# observations l of i's block up to i, where t_l = (xs[l] - centre) / h, in the l-th
# prefix matrix of the workspace
function _prefix!(ws::SumsWorkspace, l, xs, vals, starts, centers, h, smax)
  n = length(xs)
  l <= length(ws.prefixes) || push!(ws.prefixes, Matrix{Float64}(undef, 0, 0))
  size(ws.prefixes[l]) == (smax + 1, n) || (ws.prefixes[l] = Matrix{Float64}(undef, smax + 1, n))
  prefix = ws.prefixes[l]
  @inbounds for β in 1:length(starts)-1
    s, e, c = starts[β], starts[β + 1] - 1, centers[β]
    for i in s:e
      t = (xs[i] - c) / h
      v = vals[i]
      for r in 1:smax+1
        prefix[r, i] = (i == s ? 0.0 : prefix[r, i - 1]) + v
        v *= t
      end
    end
  end
  return prefix
end


## Binned sums

function _kernelsums!(out, e::BinnedSums, xs, vals, k, h, points, kmax, ws)
  gs, gv = _linearbin!(ws, xs, vals, _hmin(h) / e.binsperh)
  lo, hi = _binned_support(k)
  return _windowsums!(out, gs, gv, k, h, points, Val(kmax), lo, hi, _sortedorder!(ws, points))
end

_binned_support(k::AbstractKernel) = support(k)
_binned_support(k::GaussianFamily) = (s = 12 * first(_scale_order(k)); (-s, s))

# Linear binning onto the grid xs[1] + l δ: each value is split between the two
# neighbouring grid points in proportion to its distance from them. Returns the
# occupied grid points, in increasing order, and their totals.
function _linearbin!(ws::SumsWorkspace, xs, vals, δ)
  (xs[end] - xs[1]) / δ < 2.0^50 || throw(ArgumentError(
    "BinnedSums: the grid would need more than 2^50 points for this bandwidth; use an exact engine, " *
    "FastSums or WindowSums for polynomial kernels and DirectSums otherwise"))
  gl, gv = empty!(ws.gridindex), empty!(ws.gridvals)
  for i in eachindex(xs)
    s = (xs[i] - xs[1]) / δ
    l = floor(Int, s)
    f = s - l
    _addbin!(gl, gv, l, (1 - f) * vals[i])
    _addbin!(gl, gv, l + 1, f * vals[i])
  end
  gs = resize!(ws.grid, length(gl))
  gs .= xs[1] .+ gl .* δ
  return gs, gv
end

# adds v to grid point l; points arrive in nondecreasing order, at most one behind
function _addbin!(gl, gv, l, v)
  v == 0 && return
  n = length(gl)
  if n >= 1 && gl[n] == l
    gv[n] += v
  elseif n >= 2 && gl[n - 1] == l
    gv[n - 1] += v
  else
    push!(gl, l)
    push!(gv, v)
  end
  return
end
