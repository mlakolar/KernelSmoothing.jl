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
original one, in any ``L^p`` distance over the points, and it improves the
estimate wherever that is not monotone. The rearrangement is with respect to the
points, each counting equally, so that for a rearrangement over an interval the
points should be equally spaced, as the default points of the fits are.

`confint(r; kwargs...)` and `conf_band(r; kwargs...)` rearrange the lower and upper
ends of the intervals and bands of `m` separately. A band that covers a monotone
function still covers it after the rearrangement (Chernozhukov, Fernández-Val and
Galichon 2009, Proposition 2), and it is never wider; pointwise intervals keep no
such guarantee. Works for any fit with `points` and `coef`, such as
[`LocalPolynomialRegression`](@ref), [`PointwiseDensity`](@ref),
[`LocalPolynomialDensity`](@ref) and [`LocalLikelihood`](@ref).
"""
function rearrange(m; decreasing::Bool = false)
  pts = Vector{Float64}(m.points)
  issorted(pts) || throw(ArgumentError("rearrangement needs sorted points"))
  est = Vector{Float64}(coef(m))
  any(isnan, est) && throw(ArgumentError("the fit has unidentified estimates (NaN), which cannot be rearranged"))
  return Rearranged(m, decreasing, pts, sort(est; rev = decreasing))
end

function Base.show(io::IO, r::Rearranged)
  print(io, "Rearranged(", r.decreasing ? "decreasing" : "increasing", ", ", length(r.points), " points, ",
        nameof(typeof(r.model)), ")")
end

coef(r::Rearranged) = copy(r.estimate)

function confint(r::Rearranged; kwargs...)
  c = confint(r.model; kwargs...)
  return hcat(sort(c[:, 1]; rev = r.decreasing), sort(c[:, 2]; rev = r.decreasing))
end

function conf_band(r::Rearranged; kwargs...)
  b = conf_band(r.model; kwargs...)
  return merge(b, (lower = sort(b.lower; rev = r.decreasing), upper = sort(b.upper; rev = r.decreasing)))
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
method of Lawson and Hanson (1974).
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
  F, E = _conegenerators(shape, t)
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

# min ‖A x - b‖ over x ≥ 0 by the active-set method of Lawson and Hanson
function _nnls(A, b; maxiter = 30 * size(A, 2) + 50)
  n = size(A, 2)
  x = zeros(n)
  passive = falses(n)
  tol = 10 * eps() * max(1.0, LinearAlgebra.norm(A, Inf)) * max(1.0, LinearAlgebra.norm(b, Inf)) * size(A, 1)
  w = A' * b
  iter = 0
  while iter < maxiter
    iter += 1
    j = 0
    best = tol
    for i in 1:n
      !passive[i] && w[i] > best && ((j, best) = (i, w[i]))
    end
    j == 0 && break
    passive[j] = true
    while true
      P = findall(passive)
      z = zeros(n)
      z[P] = A[:, P] \ b
      if all(z[P] .> 0)
        x = z
        break
      end
      α = minimum(x[i] / (x[i] - z[i]) for i in P if z[i] <= 0)
      x .+= α .* (z .- x)
      for i in P
        x[i] <= tol && (passive[i] = false; x[i] = 0.0)
      end
    end
    w = A' * (b .- A * x)
  end
  return x
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
squared standard errors are a natural choice. `coef(c)` returns the constrained
estimates. For intervals that respect a monotone shape, see [`rearrange`](@ref).
"""
function constrain(m, shape::Symbol; weights = nothing)
  pts = Vector{Float64}(m.points)
  est = Vector{Float64}(coef(m))
  any(isnan, est) && throw(ArgumentError("the fit has unidentified estimates (NaN), which cannot be constrained"))
  return ShapeConstrained(m, shape, pts, shape_projection(est, pts; shape, weights))
end

function Base.show(io::IO, c::ShapeConstrained)
  print(io, "ShapeConstrained(:", c.shape, ", ", length(c.points), " points, ", nameof(typeof(c.model)), ")")
end

coef(c::ShapeConstrained) = copy(c.estimate)
