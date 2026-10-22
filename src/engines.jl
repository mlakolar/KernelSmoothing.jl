"""
    SumEngine

Supertype of the algorithms that compute [`kernelsums`](@ref).
[`DirectSums`](@ref), [`WindowSums`](@ref) and [`FastSums`](@ref) are exact up to
rounding; [`BinnedSums`](@ref) approximates. [`isexact`](@ref) tells them apart.
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
supported kernels; gives the same sums as [`DirectSums`](@ref).
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
into blocks about one window wide and the power sums are taken about the centre
of each block. The rounding error is then about `1e-13` times the total absolute
weight of the observations near the point, whatever the location and scale of the
data. A sum that is much smaller than that, because the kernel nearly vanishes at
every observation in the window, has a larger relative error.
"""
struct FastSums <: SumEngine end

"""
    BinnedSums(; binsperh = 50)

Approximate sums from linearly binned data: each observation's value is split
between the two nearest points of a grid with spacing `h / binsperh`, in
proportion to its distance from them, and the sums are computed over the occupied
grid points. Gaussian kernels are cut off at 12 standard deviations. Useful for
Gaussian kernels with many observations, where [`DirectSums`](@ref) is slow.

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

_default_engine(::PolynomialKernel) = FastSums()
_default_engine(::AbstractKernel) = DirectSums()

"""
    kernelsums(d::SmoothingData, k, h, points; kmax = 0, z = nothing, engine)
    kernelsums(d::SmoothingData, k, h; kmax = 0, z = nothing, engine)

The kernel sums

    S[r + 1, j] = ∑ᵢ wᵢ zᵢ k(uᵢⱼ) uᵢⱼ^r,   uᵢⱼ = (xᵢ - points[j]) / h,   r = 0, …, kmax,

over the observations `xᵢ` of `d` with weights `wᵢ`, as a
`(kmax + 1) × length(points)` matrix. Without `points` the sums are evaluated at
the observations themselves, `d.x`, as cross-validation needs. `z` holds one value
per observation in the sorted order of `d`, such as the responses `d.y`; by
default every `zᵢ = 1`.

Estimators are built from these sums. For a symmetric kernel,
`S[1, j] / (h * d.sumw)` is the kernel density estimate at `points[j]`; with
`kmax = 2p`, and with `z = d.y` and `kmax = p`, the sums give the normal
equations of a local polynomial fit of order `p` in the scaled variable `u`.

`engine` chooses the algorithm: [`FastSums`](@ref) for polynomial kernels and
[`DirectSums`](@ref) for others by default, or [`WindowSums`](@ref) or
[`BinnedSums`](@ref).

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
function kernelsums(d::SmoothingData, k::AbstractKernel, h::Real,
                    points::AbstractVector{<:Real}; kmax::Integer = 0, kwargs...)
  kmax >= 0 || throw(ArgumentError("kmax must be nonnegative, got $kmax"))
  out = Matrix{Float64}(undef, kmax + 1, length(points))
  return kernelsums!(out, d, k, h, points; kwargs...)
end

kernelsums(d::SmoothingData, k::AbstractKernel, h::Real; kwargs...) =
  kernelsums(d, k, h, d.x; kwargs...)

"""
    kernelsums!(out, d::SmoothingData, k, h, points; z = nothing, engine)

[`kernelsums`](@ref) into the matrix `out`, whose number of rows sets
`kmax = size(out, 1) - 1`.
"""
function kernelsums!(out::AbstractMatrix, d::SmoothingData, k::AbstractKernel, h::Real,
                     points::AbstractVector{<:Real}; z = nothing,
                     engine::SumEngine = _default_engine(k))
  isfinite(h) && h > 0 || throw(ArgumentError("bandwidth h must be positive and finite, got $h"))
  Base.require_one_based_indexing(out, points)
  all(isfinite, points) || throw(ArgumentError("points must be finite"))
  size(out, 1) >= 1 && size(out, 2) == length(points) || throw(DimensionMismatch(
    "out must have length(points) = $(length(points)) columns, got size $(size(out))"))
  _kernelsums!(out, engine, d.x, _sumvalues(d, z), k, float(h), points, size(out, 1) - 1)
  return out
end

# wᵢ zᵢ in the sorted order of d
function _sumvalues(d::SmoothingData, z)
  z === nothing && return d.w === nothing ? ones(nobs(d)) : d.w
  _check_values(z, "z")
  length(z) == nobs(d) ||
    throw(DimensionMismatch("z has $(length(z)) values for $(nobs(d)) observations"))
  return d.w === nothing ? Vector{Float64}(z) : d.w .* z
end

# the indices of points in increasing order of the points
_sortedorder(points) = issorted(points) ? collect(eachindex(points)) : sortperm(points)

# the first i ≥ start with pred(i), or n + 1, for a predicate that is false and then
# true on 1:n; galloping from start takes O(log(answer - start)) evaluations
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


## Direct and windowed sums

function _kernelsums!(out, ::DirectSums, xs, vals, k, h, points, kmax)
  fill!(out, 0.0)
  for j in eachindex(points), i in eachindex(xs)
    u = (xs[i] - points[j]) / h
    v = vals[i] * k(u)
    for r in 1:kmax+1
      out[r, j] += v
      v *= u
    end
  end
  return out
end

function _kernelsums!(out, ::WindowSums, xs, vals, k, h, points, kmax)
  lo, hi = support(k)
  isfinite(lo) && isfinite(hi) || throw(ArgumentError(
    "WindowSums needs a compactly supported kernel; use DirectSums or BinnedSums for $k"))
  return _windowsums!(out, xs, vals, k, h, points, kmax, lo, hi)
end

# sums over the observations with lo ≤ u ≤ hi, which is all of them where k is
# nonzero when (lo, hi) is the support of k
function _windowsums!(out, xs, vals, k, h, points, kmax, lo, hi)
  fill!(out, 0.0)
  n = length(xs)
  a, b = 1, 1   # first index with u ≥ lo, first index with u > hi
  for j in _sortedorder(points)
    p = points[j]
    a = _gallop(i -> (xs[i] - p) / h >= lo, a, n)
    b = _gallop(i -> (xs[i] - p) / h > hi, b, n)
    for i in a:b-1
      u = (xs[i] - p) / h
      v = vals[i] * k(u)
      for r in 1:kmax+1
        out[r, j] += v
        v *= u
      end
    end
  end
  return out
end


## Fast sums

_kernelsums!(out, ::FastSums, xs, vals, k::AbstractKernel, h, points, kmax) =
  throw(ArgumentError("FastSums needs a polynomial kernel; use DirectSums or BinnedSums for $k"))

function _kernelsums!(out, ::FastSums, xs, vals, k::PolynomialKernel, h, points, kmax)
  b, c = pieces(k)
  Δ = (last(b) - first(b)) * h / 2   # block width: about half a window
  # below the resolution of the data, blocks would hold single values; sum directly
  Δ > 64 * eps(max(abs(xs[1]), abs(xs[end]))) ||
    return _windowsums!(out, xs, vals, k, h, points, kmax, first(b), last(b))
  smax = maximum(length, c) - 1 + kmax   # the highest power of t needed
  starts, centers, blockof, prefix = _blockprefix(xs, vals, Δ, h, smax)
  n, np = length(xs), length(c)
  ptr = ones(Int, np + 1)   # piece π holds observations ptr[π]:ptr[π+1]-1
  L = zeros(smax + 1)       # power sums of t over part of a block
  g = zeros(smax + 1)       # a piece of k times u^r, in powers of t
  fill!(out, 0.0)
  for j in _sortedorder(points)
    p = points[j]
    # piece π covers b[π] ≤ u < b[π+1], and the last piece also u = b[end], as in k(u)
    # (the predicates capture only variables assigned once, so nothing is boxed)
    for π in 1:np
      ptr[π] = _gallop(i -> (xs[i] - p) / h >= b[π], ptr[π], n)
    end
    ptr[np + 1] = _gallop(i -> (xs[i] - p) / h > b[np + 1], ptr[np + 1], n)
    for π in 1:np
      i1, i2 = ptr[π], ptr[π + 1] - 1
      i1 <= i2 || continue
      m = _mid(b, π)
      for β in blockof[i1]:blockof[i2]
        s, e = max(i1, starts[β]), min(i2, starts[β + 1] - 1)
        for q in 1:smax+1
          L[q] = prefix[q, e] - (s > starts[β] ? prefix[q, s - 1] : 0.0)
        end
        # u = t + δ with t = (x - centre) / h; the piece is a polynomial in u - m
        δ = (centers[β] - p) / h
        len = length(c[π])
        for q in 1:len
          g[q] = c[π][q]
        end
        _shift!(g, len, δ - m)
        for r in 1:kmax+1
          acc = 0.0
          for q in 1:len
            acc = muladd(g[q], L[q], acc)
          end
          out[r, j] += acc
          r <= kmax && (len = _mulaffine!(g, len, δ))
        end
      end
    end
  end
  return out
end

# Splits the sorted xs into blocks of width Δ and returns the first index and the
# centre of each block, the block of each observation, and the power sums
# prefix[q, i] = ∑ vals[l] t_l^(q-1) over the observations l of i's block up to i,
# where t_l = (xs[l] - centre) / h.
function _blockprefix(xs, vals, Δ, h, smax)
  n = length(xs)
  starts, centers = Int[], Float64[]
  blockof = Vector{Int}(undef, n)
  prefix = Matrix{Float64}(undef, smax + 1, n)
  q0 = -1.0
  for i in 1:n
    q = floor((xs[i] - xs[1]) / Δ)
    newblock = i == 1 || q != q0
    if newblock
      push!(starts, i)
      push!(centers, xs[1] + (q + 0.5) * Δ)
      q0 = q
    end
    β = length(starts)
    blockof[i] = β
    t = (xs[i] - centers[β]) / h
    v = vals[i]
    for s in 1:smax+1
      prefix[s, i] = (newblock ? 0.0 : prefix[s, i - 1]) + v
      v *= t
    end
  end
  push!(starts, n + 1)
  return starts, centers, blockof, prefix
end


## Binned sums

function _kernelsums!(out, e::BinnedSums, xs, vals, k, h, points, kmax)
  gs, gv = _linearbin(xs, vals, h / e.binsperh)
  lo, hi = _binned_support(k)
  return _windowsums!(out, gs, gv, k, h, points, kmax, lo, hi)
end

_binned_support(k::AbstractKernel) = support(k)
_binned_support(k::GaussianFamily) = (s = 12 * first(_scale_order(k)); (-s, s))

# Linear binning onto the grid xs[1] + l δ: each value is split between the two
# neighbouring grid points in proportion to its distance from them. Returns the
# occupied grid points, in increasing order, and their totals.
function _linearbin(xs, vals, δ)
  (xs[end] - xs[1]) / δ < 2.0^50 || throw(ArgumentError(
    "BinnedSums: the grid would need more than 2^50 points for this bandwidth; use FastSums"))
  gl, gv = Int[], Float64[]
  for i in eachindex(xs)
    s = (xs[i] - xs[1]) / δ
    l = floor(Int, s)
    f = s - l
    _addbin!(gl, gv, l, (1 - f) * vals[i])
    _addbin!(gl, gv, l + 1, f * vals[i])
  end
  return xs[1] .+ gl .* δ, gv
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
