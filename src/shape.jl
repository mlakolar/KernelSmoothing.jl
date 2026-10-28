## Shape constraints: rearrangement and projections onto cones of shapes

"""
    Rearranged

The rearrangement of the estimates of a fit, see [`rearrange`](@ref): `coef`,
`confint` and `conf_band` return the rearranged estimates and the rearranged ends of
the intervals and bands of the fit, at its points.
"""
struct Rearranged{M}
  model::M
  decreasing::Bool
  points::Vector{Float64}
  estimate::Vector{Float64}
end

"""
    rearrange(m; decreasing = false)

The increasing rearrangement of Chernozhukov, Fernández-Val and Galichon (2009) of
the estimates of a fit `m` at its points, which must be sorted: the estimates
sorted in increasing order, or in decreasing order with `decreasing = true`. For a
monotone function, the rearranged estimate is never farther from it than the
original one in the ``L^p`` distance over the points, for every ``p ≥ 1``; the
bound is on the distance over all points, not at each point. The rearrangement
counts each point equally: for the equally spaced default points of
[`LocalPolynomialRegression`](@ref) it is the rearrangement over an interval, and
for the quantiles that are the default points of the density fits, with respect to
the distribution of the data.

`confint(r; kwargs...)` and `conf_band(r; kwargs...)` rearrange the lower and upper
ends of the intervals and bands of `m` separately, over the points where neither end
is `NaN`. A band that covers a monotone function at the points still covers it after
the rearrangement (Chernozhukov, Fernández-Val and Galichon 2009, Proposition 2),
and the ``L^p`` norm of its width over the points does not increase; pointwise
intervals keep no such guarantee. Works for any fit whose `coef` holds one estimate
per point, such as [`LocalPolynomialRegression`](@ref), [`PointwiseDensity`](@ref),
[`LocalPolynomialDensity`](@ref) and [`LocalLikelihood`](@ref), for which the
estimates are those of the linear predictor (which the rearrangement of the mean
equals, since the link is monotone) and which has no bands.
"""
function rearrange(m; decreasing::Bool = false)
  pts, est = _pointestimates(m)
  issorted(pts) || throw(ArgumentError("rearrangement needs sorted points"))
  any(isnan, est) && throw(ArgumentError("the fit has unidentified estimates (NaN), which cannot be rearranged"))
  return Rearranged(m, decreasing, pts, sort(est; rev = decreasing))
end

function Base.show(io::IO, r::Rearranged)
  print(io, "Rearranged(", r.decreasing ? "decreasing" : "increasing", ", ", length(r.points), " points, ",
        nameof(typeof(r.model)), ")")
end

coef(r::Rearranged) = copy(r.estimate)

# the points and the estimates of a fit, which must hold one estimate per point
function _pointestimates(m)
  pts = Vector{Float64}(m.points)
  est = Vector{Float64}(coef(m))
  length(est) == length(pts) ||
    throw(ArgumentError("the fit has $(length(est)) estimates at $(length(pts)) points; shapes apply to fits with one estimate per point"))
  return pts, est
end

# the ends of intervals sorted over the points where neither is NaN, the others left in
# place
function _rearrangeends(lo, hi, decreasing)
  lo, hi = copy(lo), copy(hi)
  ok = findall(i -> !isnan(lo[i]) && !isnan(hi[i]), eachindex(lo))
  lo[ok] = sort(lo[ok]; rev = decreasing)
  hi[ok] = sort(hi[ok]; rev = decreasing)
  return lo, hi
end

function confint(r::Rearranged; kwargs...)
  c = confint(r.model; kwargs...)
  return hcat(_rearrangeends(c[:, 1], c[:, 2], r.decreasing)...)
end

function conf_band(r::Rearranged; kwargs...)
  b = conf_band(r.model; kwargs...)
  lo, hi = _rearrangeends(b.lower, b.upper, r.decreasing)
  return merge(b, (lower = lo, upper = hi))
end

const _SHAPES = (:increasing, :decreasing, :convex, :concave, :increasing_convex, :increasing_concave,
                 :decreasing_convex, :decreasing_concave)

"""
    shape_projection(values, points; shape = :increasing, weights = nothing)

The weighted least-squares projection of `values` at the sorted `points` onto the
cone of the sequences of the given `shape`,

    θ̂ = argmin ∑ₖ vₖ (θₖ - valuesₖ)²   over the θ of that shape,

with `v = weights` (by default equal): `:increasing` or `:decreasing` (isotonic
regression), `:convex` or `:concave` (nonnegative or nonpositive second divided
differences at the points), or a monotone convex or concave shape such as
`:increasing_concave`. Increasing and decreasing projections with equal weights use
pooling adjacent violators in O(m) time through ProximalBase. Otherwise, θ is
written in the generators of the cone, such as the steps `1{t ≥ tⱼ}` for increasing
sequences and the hinges `(t - tⱼ)₊` for convex ones, plus a linear part, and the
nonnegative coefficients of the generators solve a nonnegative least-squares problem
by the active-set method of Lawson and Hanson (1974), with a QR factorization of the
active generators updated as generators enter and leave. Both algorithms are exact
up to floating-point tolerance; the active-set method stops when no generator
improves the fit beyond that tolerance and warns at its iteration limit. The cones
do not change under increasing affine maps of the points, and the projection is
equivariant under the scale of the values. On one thread, pooling takes
microseconds. For the other projections, the time grows with the number of
generators in the solution: about 10–20 ms for a smooth estimate at 1000 points.
When all generators enter, the time grows as m³, taking about 5 ms at 200 points,
30 ms at 400 and 0.3 s at 800.
"""
function shape_projection(values::AbstractVector{<:Real}, points::AbstractVector{<:Real}; shape::Symbol = :increasing,
                          weights = nothing)
  shape in _SHAPES || throw(ArgumentError("shape must be one of $(join(repr.(_SHAPES), ", ")), got :$shape"))
  m = length(values)
  length(points) == m || throw(DimensionMismatch("$m values at $(length(points)) points"))
  m >= 1 || throw(ArgumentError("there must be at least one value"))
  _check_values(values, "values")
  _check_values(points, "points")
  t = Vector{Float64}(points)
  all(i -> t[i] < t[i + 1], 1:m-1) || throw(ArgumentError("the points must be strictly increasing"))
  v = weights === nothing ? ones(m) : Vector{Float64}(weights)
  length(v) == m || throw(DimensionMismatch("$(length(v)) weights for $m values"))
  all(w -> w > 0 && isfinite(w), v) || throw(ArgumentError("weights must be positive and finite"))
  if shape in (:increasing, :decreasing) && all(==(v[1]), v)
    y, s = _scaled_values(values)
    θ = similar(y)
    ProximalBase.prox!(θ, ProximalBase.IndIsotonic(; decreasing = shape === :decreasing), y, 1.0)
    return θ .*= s
  end
  # the cones are the same for points mapped onto [0, 1], which keeps the generators
  # well scaled whatever the units of the points
  z = m >= 2 ? (t .- t[1]) ./ (t[end] - t[1]) : zeros(m)
  F, E = _conegenerators(shape, z)
  return _coneproject(values, F, E, v)
end

# scale the values before pooling or projecting to avoid overflow in the sums
function _scaled_values(values)
  y = Vector{Float64}(values)
  s = maximum(abs, y)
  s = s > 0 ? s : 1.0
  y ./= s
  return y, s
end

# The linear part F and the generators E (as columns) of the cone of a shape at the
# points t: θ = F a + E c with c ≥ 0
function _conegenerators(shape, t)
  m = length(t)
  sign = startswith(string(shape), "decreasing") || shape === :concave ? -1.0 : 1.0
  one_ = ones(m, 1)
  if shape in (:increasing, :decreasing)
    return one_, sign .* [Float64(t[k] >= t[j]) for k in 1:m, j in 2:m]
  elseif shape in (:convex, :concave)
    return hcat(ones(m), t), sign .* [max(t[k] - t[j], 0.0) for k in 1:m, j in 2:m-1]
  elseif shape in (:increasing_convex, :decreasing_concave)
    # slopes that start nonnegative and increase
    return one_, sign .* [max(t[k] - t[j], 0.0) for k in 1:m, j in 1:m-1]
  else   # :increasing_concave, :decreasing_convex: slopes that end nonnegative and decrease
    s = shape === :increasing_concave ? 1.0 : -1.0
    return one_, s .* [min(t[k], t[j]) - t[1] for k in 1:m, j in 2:m]
  end
end

# the projection of y with the weights v onto {F a + E c : c ≥ 0}
function _coneproject(y, F, E, v)
  y, s = _scaled_values(y)
  m = length(y)
  size(E, 2) == 0 && return s .* (F * (F \ y))
  sw = sqrt.(v)
  Fw, Ew, yw = sw .* F, sw .* E, sw .* y
  Q = Matrix(LinearAlgebra.qr(Fw).Q)[:, 1:size(F, 2)]
  # the generators and the values with the linear part projected out
  Er = Ew .- Q * (Q' * Ew)
  yr = yw .- Q * (Q' * yw)
  c = _nnls(Er, yr)
  a = Fw \ (yw .- Ew * c)
  return s .* (F * a .+ E * c)
end

# A thin QR factorization of the ordered passive columns, with spare columns that
# grow as needed. Only the first length(P) columns of Q and the leading block of R
# are live.
mutable struct _NNLSQR
  P::Vector{Int}
  Q::Matrix{Float64}
  R::Matrix{Float64}
  h::Vector{Float64}
  max_capacity::Int
  updates::Int
  rebuilds::Int
  since_rebuild::Int
end

function _NNLSQR(m, n)
  max_capacity = min(m, n)
  capacity = min(max_capacity, 16)
  return _NNLSQR(Int[], zeros(m, capacity), zeros(capacity, capacity), zeros(capacity), max_capacity, 0, 0, 0)
end

# Grow only after an insertion passes all rejection checks. Copy live blocks and
# leave the update and rebuild counters alone: resizing does not change the factors.
function _nnls_qr_grow!(factor, required)
  capacity = size(factor.Q, 2)
  required <= capacity && return factor
  capacity = min(factor.max_capacity, max(required, 2 * capacity))
  p = length(factor.P)
  Q = zeros(size(factor.Q, 1), capacity)
  R, h = zeros(capacity, capacity), zeros(capacity)
  Q[:, 1:p] .= view(factor.Q, :, 1:p)
  R[1:p, 1:p] .= view(factor.R, 1:p, 1:p)
  h[1:p] .= view(factor.h, 1:p)
  factor.Q, factor.R, factor.h = Q, R, h
  return factor
end

function _nnls_qr_factors(A, b, P)
  F = LinearAlgebra.qr(A[:, P])
  Q, R = Matrix(F.Q), Matrix(F.R)
  return Q, R, Q' * b
end

function _nnls_qr_rebuild!(factor, A, b)
  p = length(factor.P)
  if p > 0
    Q, R, h = _nnls_qr_factors(A, b, factor.P)
    factor.Q[:, 1:p] .= Q
    factor.R[1:p, 1:p] .= R
    factor.h[1:p] .= h
  end
  factor.rebuilds += 1
  factor.since_rebuild = 0
  return factor
end

# Append by modified Gram–Schmidt with one reorthogonalization pass. All tests
# precede the commit, so a refused column leaves the entire factorization unchanged.
function _nnls_qr_insert!(factor, A, b, j; positive = false)
  p = length(factor.P)
  p < factor.max_capacity || return false
  v = copy(view(A, :, j))
  na = LinearAlgebra.norm(v)
  r = zeros(p)
  for _ in 1:2
    for i in 1:p
      q = view(factor.Q, :, i)
      t = LinearAlgebra.dot(q, v)
      r[i] += t
      v .-= t .* q
    end
  end
  d = LinearAlgebra.norm(v)
  # sqrt(m) converts a per-coordinate cutoff of 1e-10 to a Euclidean norm
  # cutoff, allowing for residual error spread across the m coordinates.
  rank_tol = 1e-10 * sqrt(size(A, 1))
  d > 0 && d >= rank_tol * na || return false
  v ./= d
  h = LinearAlgebra.dot(v, b)
  positive && h / d <= 0 && return false
  # A small new diagonal amplifies roundoff in the passive solve. Use fresh
  # Householder factors there, and otherwise amortize rebuilds over max(p, 16) updates.
  rebuild = factor.since_rebuild + 1 >= max(p + 1, 16) || d < sqrt(eps()) * na
  if rebuild
    Q, R, rhs = _nnls_qr_factors(A, b, [factor.P; j])
    abs(R[end, end]) >= rank_tol * na || return false
    positive && rhs[end] / R[end, end] <= 0 && return false
    _nnls_qr_grow!(factor, p + 1)
    factor.Q[:, 1:p+1] .= Q
    factor.R[1:p+1, 1:p+1] .= R
    factor.h[1:p+1] .= rhs
    factor.rebuilds += 1
    factor.since_rebuild = 0
  else
    _nnls_qr_grow!(factor, p + 1)
    factor.Q[:, p+1] .= v
    factor.R[1:p, p+1] .= r
    factor.R[p+1, 1:p] .= 0
    factor.R[p+1, p+1] = d
    factor.h[p+1] = h
    factor.since_rebuild += 1
  end
  push!(factor.P, j)
  factor.updates += 1
  return true
end

# Remove a column of R, then chase its subdiagonal bulge with Givens rotations.
# If R becomes G R, then h becomes G h and Q becomes Q Gᵀ.
function _nnls_qr_delete!(factor, A, b, position)
  p = length(factor.P)
  Q, R, h = factor.Q, factor.R, factor.h
  for j in position:p-1
    R[1:p, j] .= view(R, 1:p, j + 1)
  end
  for i in position:p-1
    d = hypot(R[i, i], R[i+1, i])
    c, s = d == 0 ? (1.0, 0.0) : (R[i, i] / d, R[i+1, i] / d)
    R[i, i], R[i+1, i] = d, 0.0
    for j in i+1:p-1
      a, b_ = R[i, j], R[i+1, j]
      R[i, j], R[i+1, j] = c * a + s * b_, -s * a + c * b_
    end
    a, b_ = h[i], h[i+1]
    h[i], h[i+1] = c * a + s * b_, -s * a + c * b_
    for j in axes(Q, 1)
      a, b_ = Q[j, i], Q[j, i+1]
      Q[j, i], Q[j, i+1] = c * a + s * b_, -s * a + c * b_
    end
  end
  deleteat!(factor.P, position)
  factor.updates += 1
  factor.since_rebuild += 1
  factor.since_rebuild >= max(p - 1, 16) && _nnls_qr_rebuild!(factor, A, b)
  return factor
end

function _nnls_passive_solve!(z, factor, A, b, fallback)
  fill!(z, 0)
  P = factor.P
  p = length(P)
  if p > 0
    z[P] = fallback ? A[:, P] \ b :
           LinearAlgebra.UpperTriangular(view(factor.R, 1:p, 1:p)) \ view(factor.h, 1:p)
  end
  return z
end

# min ‖A x - b‖ over x ≥ 0 by Lawson and Hanson (1974, Chapter 23), with each
# nonzero column of A and b scaled to unit norm. The ordered passive columns have
# A_s[:, P] = Q R and h = Qᵀ b_s; passive solves are triangular. Insertions use
# reorthogonalized Gram–Schmidt and deletions use Givens rotations, with occasional
# rebuilds. With amortized rebuilds, O(m) updates and n = O(m) cost O(m³), including
# dual vectors. Accuracy rebuilds and the rank-revealing fallback can require extra
# factorizations.
# An entering index needs w_i > tol and a positive tentative coefficient. Rejections
# last only until x changes. Backtracking zeros the blocking index explicitly and
# removes every nonpositive coefficient before solving again. Rejected indices are
# included in the final dual check; a violation switches to rank-revealing passive
# solves through \ before stopping. A negative tol exercises the coefficient guard,
# rather than imposing a negative optimality bound. The iteration limit warns when
# optimality was not reached. Optional internal stats record updates and rebuilds.
function _nnls(A, b; maxiter = 3 * size(A, 2) + 50, tol = nothing, stats = nothing)
  m, n = size(A)
  if stats !== nothing
    for key in (:updates, :rebuilds, :rejections, :reconsiderations, :fallbacks, :deletions, :max_blockers)
      stats[key] = 0
    end
  end
  nb = LinearAlgebra.norm(b)
  nb > 0 || return zeros(n)
  s = [LinearAlgebra.norm(view(A, :, j)) for j in 1:n]
  live = findall(>(0), s)
  As = A[:, live] ./ s[live]'
  bs = b ./ nb
  k = length(live)
  x = zeros(k)
  passive = falses(k)
  rejected = falses(k)   # entering indices refused since x last changed
  ever_rejected = stats === nothing ? nothing : falses(k)
  factor = _NNLSQR(m, k)
  P = factor.P
  tol = tol === nothing ? 10 * eps() * max(m, k) : tol
  r = copy(bs)
  w = As' * r
  converged = false
  guarded = false
  fallback = false
  for _ in 1:maxiter
    j = 0
    best = tol
    for i in 1:k
      !passive[i] && !rejected[i] && w[i] > best && ((j, best) = (i, w[i]))
    end
    if j == 0
      if all(i -> passive[i] || w[i] <= tol, 1:k)
        converged = true
        break
      end
      if fallback
        # Negative tolerances test the coefficient guard, not a negative KKT bound.
        guarded = tol < 0 && all(i -> passive[i] || w[i] <= 10 * eps() * max(m, k), 1:k)
        guarded && break
        continue
      end
      # Re-solve the passive set, then reconsider rejected candidates with the
      # same rank-revealing least-squares solves used before QR updating.
      fallback = true
      stats !== nothing && (stats[:fallbacks] += 1)
      fill!(rejected, false)
    else
      stats !== nothing && ever_rejected[j] && (stats[:reconsiderations] += 1)
      if fallback
        push!(P, j)
      elseif !_nnls_qr_insert!(factor, As, bs, j; positive = true)
        rejected[j] = true
        if stats !== nothing
          stats[:rejections] += 1
          ever_rejected[j] = true
        end
        continue
      end
      passive[j] = true
    end
    z = zeros(k)
    _nnls_passive_solve!(z, factor, As, bs, fallback)
    if fallback && j != 0 && z[j] <= 0
      # The updated path already checked this coefficient before committing.
      # In fallback mode there is no QR factorization to change or restore.
      pop!(P)
      passive[j] = false
      rejected[j] = true
      if stats !== nothing
        stats[:rejections] += 1
        ever_rejected[j] = true
      end
      continue
    end
    for _ in 1:k + 1
      if all(i -> z[i] > 0, P)
        x = z
        break
      end
      # the largest step toward z that keeps x ≥ 0; the indices that block it leave
      α, blocking = Inf, 0
      for i in P
        z[i] <= 0 || continue
        αi = x[i] / (x[i] - z[i])
        αi < α && ((α, blocking) = (αi, i))
      end
      x .+= α .* (z .- x)
      x[blocking] = 0.0
      blockers = 0
      for position in length(P):-1:1
        i = P[position]
        if x[i] <= 0
          passive[i] = false
          x[i] = 0.0
          if fallback
            deleteat!(P, position)
          else
            _nnls_qr_delete!(factor, As, bs, position)
          end
          blockers += 1
        end
      end
      if stats !== nothing
        stats[:deletions] += blockers
        stats[:max_blockers] = max(stats[:max_blockers], blockers)
      end
      _nnls_passive_solve!(z, factor, As, bs, fallback)
    end
    fill!(rejected, false)
    # x is zero off P, also after fallback solves. Only inactive dual entries
    # are read by the entering and stopping tests; include rejected indices.
    copyto!(r, bs)
    for i in P
      LinearAlgebra.axpy!(-x[i], view(As, :, i), r)
    end
    for i in 1:k
      w[i] = passive[i] ? 0.0 : LinearAlgebra.dot(view(As, :, i), r)
    end
  end
  converged || guarded || @warn "the nonnegative least-squares solver reached its iteration limit; the projection may be inexact"
  if stats !== nothing
    stats[:updates] = factor.updates
    stats[:rebuilds] = factor.rebuilds
  end
  out = zeros(n)
  out[live] = x .* nb ./ s[live]
  return out
end

"""
    ShapeConstrained

The projection of the estimates of a fit onto a cone of shapes, see
[`constrain`](@ref); `coef` returns the constrained estimates.
"""
struct ShapeConstrained{M}
  model::M
  shape::Symbol
  points::Vector{Float64}
  estimate::Vector{Float64}
end

"""
    constrain(m, shape; weights = nothing)

The estimates of a fit `m` at its points projected onto the sequences of the given
`shape` by [`shape_projection`](@ref): smoothing followed by an isotonic or convex
regression of the smooth estimate, as Mammen (1991) proposed for monotone
regression. The projection never increases the distance to a function of that
shape, measured over the points in the weighted norm, and removes the wiggles that
violate the shape; `weights` (equal by default) weigh the points, and the inverse
squared standard errors are a natural choice. Repeated points share one value.
`coef(c)` returns the constrained estimates. For [`LocalLikelihood`](@ref) these
are the estimates of the linear predictor, so that a convex shape is convexity of
the logit or the logarithm of the mean. For intervals that respect a monotone
shape, see [`rearrange`](@ref).
"""
function constrain(m, shape::Symbol; weights = nothing)
  pts, est = _pointestimates(m)
  any(isnan, est) && throw(ArgumentError("the fit has unidentified estimates (NaN), which cannot be constrained"))
  v = weights === nothing ? ones(length(pts)) : Vector{Float64}(weights)
  length(v) == length(pts) || throw(DimensionMismatch("$(length(v)) weights for $(length(pts)) points"))
  issorted(pts) || throw(ArgumentError("the points of the fit must be sorted"))
  # repeated points, as quantiles of data with ties: their weighted mean, with their
  # summed weight, enters the projection, and they share its value
  u = unique(pts)
  g = [searchsortedfirst(u, t) for t in pts]
  vu = zeros(length(u))
  eu = zeros(length(u))
  for i in eachindex(pts)
    vu[g[i]] += v[i]
    eu[g[i]] += v[i] * est[i]
  end
  θ = shape_projection(eu ./ vu, u; shape, weights = vu)
  return ShapeConstrained(m, shape, pts, θ[g])
end

function Base.show(io::IO, c::ShapeConstrained)
  print(io, "ShapeConstrained(:", c.shape, ", ", length(c.points), " points, ", nameof(typeof(c.model)), ")")
end

coef(c::ShapeConstrained) = copy(c.estimate)
