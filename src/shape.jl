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
`:increasing_concave`. The projection is exact: θ is written in the generators of
the cone, such as the steps `1{t ≥ tⱼ}` for increasing sequences and the hinges
`(t - tⱼ)₊` for convex ones, plus a linear part, and the nonnegative coefficients
of the generators solve a nonnegative least-squares problem by the active-set
method of Lawson and Hanson (1974). The cones do not change under increasing affine
maps of the points, and the projection is equivariant under the scale of the
values. Its cost grows with the number of generators
in the solution: a few milliseconds for a smooth estimate at 1000 points, but for
values that already have the shape, every generator enters, and the projection
takes about 0.1 s at 200 points, 1 s at 400 and 15 s at 800.
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
  # the cones are the same for points mapped onto [0, 1], which keeps the generators
  # well scaled whatever the units of the points
  z = m >= 2 ? (t .- t[1]) ./ (t[end] - t[1]) : zeros(m)
  F, E = _conegenerators(shape, z)
  return _coneproject(Vector{Float64}(values), F, E, v)
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
  m = length(y)
  size(E, 2) == 0 && return F * (F \ y)
  sw = sqrt.(v)
  Fw, Ew, yw = sw .* F, sw .* E, sw .* y
  Q = Matrix(LinearAlgebra.qr(Fw).Q)[:, 1:size(F, 2)]
  # the generators and the values with the linear part projected out
  Er = Ew .- Q * (Q' * Ew)
  yr = yw .- Q * (Q' * yw)
  c = _nnls(Er, yr)
  a = Fw \ (yw .- Ew * c)
  return F * a .+ E * c
end

# min ‖A x - b‖ over x ≥ 0 by the active-set method of Lawson and Hanson (1974,
# Chapter 23), on the columns of A and on b scaled to unit norm, so that its
# tolerance does not depend on their units: an index enters the passive set when its
# gradient is positive, but not if the least-squares solution on the new passive set
# gives it a nonpositive coefficient, which rounding can cause; an infeasible solution
# is moved toward until a coefficient reaches zero, and those that do leave. (A
# negative `tol` lets indices with a negative gradient enter, which the guard refuses,
# for the tests.)
function _nnls(A, b; maxiter = 3 * size(A, 2) + 50, tol = nothing)
  m, n = size(A)
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
  tol = tol === nothing ? 10 * eps() * max(m, k) : tol
  w = As' * bs
  converged = false
  for _ in 1:maxiter
    j = 0
    best = tol
    for i in 1:k
      !passive[i] && !rejected[i] && w[i] > best && ((j, best) = (i, w[i]))
    end
    if j == 0
      converged = true
      break
    end
    passive[j] = true
    P = findall(passive)
    z = zeros(k)
    z[P] = As[:, P] \ bs
    if z[j] <= 0
      passive[j] = false
      rejected[j] = true
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
      for i in P
        x[i] <= 0 && (passive[i] = false; x[i] = 0.0)
      end
      P = findall(passive)
      z = zeros(k)
      isempty(P) || (z[P] = As[:, P] \ bs)
    end
    fill!(rejected, false)
    w = As' * (bs .- As * x)
  end
  converged || @warn "the nonnegative least-squares solver reached its iteration limit; the projection may be inexact"
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
