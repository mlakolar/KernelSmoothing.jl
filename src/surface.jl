## Bivariate local polynomial smoothing

# The observations of a bivariate smoother, sorted by the first coordinate: x1, x2,
# the responses, the weights, the groups, and the input order perm
struct _Bivariate{Y<:Union{Nothing,Vector{Float64}},W<:Union{Nothing,Vector{Float64}}}
  x1::Vector{Float64}
  x2::Vector{Float64}
  y::Y
  w::W
  groups::Vector{Int}
  perm::Vector{Int}
end

function _bivariate(x, y, weights, groups)
  x1, x2 = x
  n = length(x1)
  length(x2) == n || throw(DimensionMismatch("the two coordinates have $(length(x1)) and $(length(x2)) values"))
  n > 0 || throw(ArgumentError("there must be at least one observation"))
  _check_values(x1, "x[1]")
  _check_values(x2, "x[2]")
  if y !== nothing
    length(y) == n || throw(DimensionMismatch("y has $(length(y)) values for $n observations"))
    _check_values(y, "y")
  end
  if weights !== nothing
    length(weights) == n || throw(DimensionMismatch("weights has $(length(weights)) values for $n observations"))
    _check_values(weights, "weights")
    all(>=(0), weights) && any(>(0), weights) || throw(ArgumentError("weights must be nonnegative with a positive sum"))
  end
  groups === nothing || length(groups) == n ||
    throw(DimensionMismatch("groups has $(length(groups)) values for $n observations"))
  perm = sortperm(x1)
  codes = groups === nothing ? zeros(Int, n) : _clustercodes(groups, 1:n, perm)
  return _Bivariate(Float64[x1[i] for i in perm], Float64[x2[i] for i in perm],
                    y === nothing ? nothing : Float64[y[i] for i in perm],
                    weights === nothing ? nothing : Float64[weights[i] for i in perm], codes, perm)
end

_surfacebandwidths(h::Real) = _surfacebandwidths((h, h))
function _surfacebandwidths(h)
  length(h) == 2 && all(v -> v isa Real && 0 < v < Inf, h) ||
    throw(ArgumentError("bandwidth must be a positive number or a pair of positive numbers, got $(repr(h))"))
  return (Float64(h[1]), Float64(h[2]))
end

# the points of a bivariate smoother as two vectors of coordinates, and whether they
# form a grid
function _surfacepoints(points)
  if points isa Tuple{AbstractVector,AbstractVector}
    g1, g2 = _checkpoints(points[1]), _checkpoints(points[2])
    return vec([s for s in g1, t in g2]), vec([t for s in g1, t in g2]), (length(g1), length(g2))
  elseif points isa AbstractMatrix
    size(points, 2) == 2 || throw(ArgumentError("a matrix of points needs two columns, got $(size(points, 2))"))
    return _checkpoints(points[:, 1]), _checkpoints(points[:, 2]), nothing
  elseif points isa AbstractVector && all(p -> p isa Tuple && length(p) == 2, points)
    return _checkpoints(first.(points)), _checkpoints(last.(points)), nothing
  end
  throw(ArgumentError("points must be a pair of grids, a matrix with two columns or a vector of pairs"))
end

"""
    localpolynomial((x1, x2), y, points; bandwidth, degree = 1, kernel = EpanechnikovKernel(),
                    weights = nothing)

Bivariate local polynomial estimates of a regression function of two covariates:
at a point `(s, t)`, the fit of order `degree`, 0 (local constant) or 1 (local
linear), minimizes

    ∑ᵢ wᵢ k((x1ᵢ - s)/h1) k((x2ᵢ - t)/h2) (yᵢ - β₀ - β₁ (x1ᵢ - s) - β₂ (x2ᵢ - t))²

with the product kernel and the bandwidths `bandwidth = (h1, h2)`, or one number for
both, and `β̂₀` estimates the regression function there. `points` is a pair of
grids `(g1, g2)`, which gives a matrix of estimates, or the points as a matrix with
two columns or a vector of pairs, which gives a vector. Estimates whose window
does not identify the fit are `NaN`. This is the smoother of covariance surfaces in
functional data analysis; [`selectbandwidth`](@ref) with [`CrossValidation`](@ref)
chooses its bandwidth.
"""
function localpolynomial(x::Tuple{AbstractVector,AbstractVector}, y::AbstractVector, points; bandwidth,
                         degree::Integer = 1, kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing)
  degree in (0, 1) || throw(ArgumentError("bivariate fits have degree 0 or 1, got $degree"))
  b = _bivariate(x, y, weights, nothing)
  h = _surfacebandwidths(bandwidth)
  s, t, shape = _surfacepoints(points)
  est = _pointwise(j -> _surfacefit(b, kernel, s[j], t[j], h, degree, 0).estimate, eachindex(s))
  return shape === nothing ? est : reshape(est, shape)
end

"""
    smoothermatrix((x1, x2), points; bandwidth, degree = 1, kernel = EpanechnikovKernel(),
                   weights = nothing)

The weights of the bivariate estimates of [`localpolynomial`](@ref): row `j` holds
the weights of the estimate at point `j` (for a pair of grids, the points in the
column-major order of the matrix of estimates), with the columns in the input order
of the observations.
"""
function smoothermatrix(x::Tuple{AbstractVector,AbstractVector}, points; bandwidth, degree::Integer = 1,
                        kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing)
  degree in (0, 1) || throw(ArgumentError("bivariate fits have degree 0 or 1, got $degree"))
  b = _bivariate(x, nothing, weights, nothing)
  h = _surfacebandwidths(bandwidth)
  s, t, _ = _surfacepoints(points)
  rows = _pointwise(j -> _surfacefit(b, kernel, s[j], t[j], h, degree, 0; weights = true), eachindex(s))
  L = zeros(length(s), length(b.x1))
  for j in eachindex(rows)
    r, ℓ = rows[j].range, rows[j].ℓ
    for (l, i) in enumerate(r)
      L[j, b.perm[i]] = ℓ[l]
    end
  end
  return L
end

# The window of a kernel in the sorted first coordinate: the observations within the
# support of the kernel, or all for a kernel without a bounded support
function _surfacewindow(b, k, s, h1)
  lo, hi = support(k)
  a = isfinite(lo) ? searchsortedfirst(b.x1, s + lo * h1) : 1
  e = isfinite(hi) ? searchsortedlast(b.x1, s + hi * h1) : length(b.x1)
  return a:e
end

# The fit of order p at (s, t): the estimate, and with `weights = true` the weights of
# the observations in the window (in sorted order); observations of group `skip`,
# when positive, are left out, and with `self`, the weight of observation self in its
# own fit, for leave-one-out
function _surfacefit(b, k, s, t, h, p, skip; weights::Bool = false, self::Int = 0)
  r = _surfacewindow(b, k, s, h[1])
  P = p == 0 ? 1 : 3   # the intercept, and two slopes for a local linear fit
  A, c = zeros(P, P), zeros(P)
  for i in r
    skip > 0 && b.groups[i] == skip && continue
    u, v = (b.x1[i] - s) / h[1], (b.x2[i] - t) / h[2]
    kw = _weight(b.w, i) * k(u) * k(v)
    kw == 0 && continue
    z = p == 0 ? (1.0,) : (1.0, u, v)
    yi = b.y === nothing ? 0.0 : b.y[i]
    for a in 1:P
      c[a] += kw * z[a] * yi
      for e in a:P
        A[a, e] += kw * z[a] * z[e]
      end
    end
  end
  for a in 1:P, e in 1:a-1
    A[a, e] = A[e, a]
  end
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
  ok = LinearAlgebra.issuccess(C) && minimum(j -> C.factors[j, j]^2 / A[j, j], 1:P) > 1e-8
  ok || return (estimate = NaN, range = r, ℓ = fill(NaN, weights ? length(r) : 0), self = NaN)
  g = C \ [a == 1 ? 1.0 : 0.0 for a in 1:P]
  est = g' * c
  ℓself = self > 0 ? _weight(b.w, self) * k(0.0)^2 * g[1] : NaN
  ℓ = weights ? [skip > 0 && b.groups[i] == skip ? 0.0 :
                 _weight(b.w, i) * k((b.x1[i] - s) / h[1]) * k((b.x2[i] - t) / h[2]) *
                 (p == 0 ? g[1] : g[1] + g[2] * (b.x1[i] - s) / h[1] + g[3] * (b.x2[i] - t) / h[2]) for i in r] : Float64[]
  return (estimate = est, range = r, ℓ = ℓ, self = ℓself)
end

"""
    selectbandwidth(rule::CrossValidation, (x1, x2), y; degree = 1, kernel = EpanechnikovKernel(),
                    weights = nothing, groups = nothing, aspect = 1, bandwidths = nothing)

The bandwidths `(h, aspect h)` of a bivariate local polynomial fit that minimize the
cross-validation criterion `∑ᵢ wᵢ (yᵢ - m̂₋ᵢ(xᵢ))² / ∑ᵢ wᵢ`. Without `groups`,
`m̂₋ᵢ` is the fit without observation `i`, from the fit with it by the deletion
formula; with `groups`, as the curves of functional data, it is the fit without the
whole group of observation `i`, so that dependence within groups does not lead to
too small bandwidths. The criterion is evaluated over `bandwidths`, or over
`rule.gridsize` values of `h` spaced evenly on the log scale from `2r/√n` to `r`,
with `r` the larger of the range of `x1` and the range of `x2` divided by
`aspect`, and the minimum is refined between its neighbours on the grid. Fits that
are not identified make the criterion infinite. The details record the candidate
bandwidths and the criterion.
"""
function selectbandwidth(rule::CrossValidation, x::Tuple{AbstractVector,AbstractVector}, y::AbstractVector;
                         degree::Integer = 1, kernel::AbstractKernel = EpanechnikovKernel(), weights = nothing,
                         groups = nothing, aspect::Real = 1, bandwidths = nothing)
  degree in (0, 1) || throw(ArgumentError("bivariate fits have degree 0 or 1, got $degree"))
  0 < aspect < Inf || throw(ArgumentError("aspect must be positive and finite, got $aspect"))
  b = _bivariate(x, y, weights, groups)
  n = length(b.x1)
  crit(h) = _surfacecv(b, kernel, (h, aspect * h), degree, groups !== nothing)
  if bandwidths === nothing
    r = max(b.x1[end] - b.x1[1], (maximum(b.x2) - minimum(b.x2)) / aspect)
    r > 0 || throw(ArgumentError("the covariates have no spread"))
    grid = exp.(range(log(min(2r / sqrt(n), r / 2)), log(r); length = rule.gridsize))
    h, details = _gridminimize(crit, grid; largest = false, rtol = 1e-3)
  else
    cands = sort(collect(Float64, bandwidths))
    all(v -> 0 < v < Inf, cands) || throw(ArgumentError("bandwidths must be positive and finite"))
    vals = crit.(cands)
    h = cands[argmin(vals)]
    details = (grid = cands, criterion = vals)
  end
  isfinite(h) && isfinite(crit(h)) || throw(ArgumentError("cross-validation failed at every bandwidth; " *
    "the fits are not identified, so pass larger bandwidths"))
  return BandwidthSelection([h, aspect * h], rule, (grid = details.grid, criterion = details.criterion,
                                                    groups = groups !== nothing))
end

# the cross-validation criterion of bandwidths h, leaving out observations or groups
function _surfacecv(b, k, h, p, bygroup)
  n = length(b.x1)
  parts = _pointwise(i -> begin
    wi = _weight(b.w, i)
    wi == 0 && return (0.0, 0.0)
    if bygroup
      f = _surfacefit(b, k, b.x1[i], b.x2[i], h, p, b.groups[i])
      pred = f.estimate
    else
      f = _surfacefit(b, k, b.x1[i], b.x2[i], h, p, 0; self = i)
      pred = (f.estimate - f.self * b.y[i]) / (1 - f.self)
    end
    (wi * (b.y[i] - pred)^2, wi)
  end, 1:n)
  s = sum(first, parts)
  return isfinite(s) ? s / sum(last, parts) : Inf
end
