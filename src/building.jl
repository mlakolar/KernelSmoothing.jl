## Building blocks for other packages
#
# The pieces of the estimators that other packages, such as the varying-coefficient
# models of CoordinateDescent.jl, build on: the kernel weights of a window, default
# points and bandwidth grids, the search over a grid, sandwich variances of a local
# weighted least squares fit, and the critical value of uniform bands.

"""
    kernelweights(d::SmoothingData, k::AbstractKernel, x0, h)

The observations that the nonnegative kernel `k` weights at the point `x0` with the
bandwidth `h`, and their weights `wᵢ k((xᵢ - x0)/h)`, where `wᵢ` are the observation
weights of `d` (one without). Returns a named tuple of `window`, the range of sorted
observations from the first to the last with positive weight, `index`, their
indices in the order of the input (`d.perm[window]`), and `weights`. The weights
omit the factor `1/h` of `Kₕ(t) = k(t/h)/h`, which cancels in a local fit. The
window is empty when no observation has positive weight. Takes `O(log n)` time and
the length of the window.

# Examples
```julia
d = SmoothingData(x, y)
r = kernelweights(d, EpanechnikovKernel(), 0.5, 0.2)
Z = [ones(length(r.window)) (d.x[r.window] .- 0.5) ./ 0.2]   # a local linear design
θ = (Z' * (r.weights .* Z)) \\ (Z' * (r.weights .* d.y[r.window]))
```
"""
function kernelweights(d::SmoothingData, k::AbstractKernel, x0::Real, h::Real)
  _check_point(x0)
  0 < h < Inf || throw(ArgumentError("bandwidth h must be positive and finite, got $h"))
  _isnonnegative(k) || throw(ArgumentError("$k takes negative values; kernelweights needs a nonnegative kernel"))
  x0f, hf = Float64(x0), Float64(h)
  r = _poswindow(d, k, x0f, hf)
  w = Float64[_weight(d.w, i) * k((d.x[i] - x0f) / hf) for i in r]
  return (window = r, index = d.perm[r], weights = w)
end

"""
    regressionpoints(d::SmoothingData; compat = false)

The default evaluation points of the regression estimators: 30 equally spaced points
between the quantiles of order 0.025 and 0.975 of `x`, which keeps them out of sparse
tails, or between the extremes of `x` with `compat = true`, as nprobust, and where
these quantiles coincide.
"""
regressionpoints(d::SmoothingData; compat::Bool = false) = _regressionpoints(d, compat)

"""
    bandwidthgrid(d::SmoothingData; minobs = 2, mindistinct = 2, gridsize = 30)

`gridsize` bandwidths spaced evenly on the log scale, from the smallest one at which
the window of every observation holds at least `minobs` observations (itself
included, at most all of them) and `mindistinct` distinct values of `x`, increased by
one percent, to the range of `x` (or from half the range, where the smallest one
exceeds it). Below the lower end, the local fit at some observation has too few
observations or distinct values to be identified, as for a fit of order `p` with
`mindistinct ≤ p`, so that a cross-validation criterion is not finite there.
"""
function bandwidthgrid(d::SmoothingData; minobs::Integer = 2, mindistinct::Integer = 2, gridsize::Integer = 30)
  gridsize >= 2 || throw(ArgumentError("gridsize must be at least 2, got $gridsize"))
  minobs >= 1 || throw(ArgumentError("minobs must be at least 1, got $minobs"))
  1 <= mindistinct <= nunique(d) || throw(ArgumentError("mindistinct must be between 1 and the " *
    "number of distinct values, $(nunique(d)), got $mindistinct"))
  hmax = d.x[end] - d.x[1]
  hmax > 0 || throw(ArgumentError("the data have no spread"))
  m = min(Int(minobs), nobs(d))
  hmin = maximum(i -> max(knn_distance(d, d.x[i], m), knn_distance(d, d.x[i], mindistinct; distinct = true)),
                 1:nobs(d)) * 1.01
  hmin < hmax || (hmin = hmax / 2)
  hmin > 0 || throw(ArgumentError("the smallest bandwidth is zero; ask for at least 2 observations " *
    "or distinct values in each window"))
  return exp.(range(log(hmin), log(hmax); length = gridsize))
end

"""
    gridminimize(f, grid; largest = false, rtol = 1e-3)

The minimizer of the criterion `f` over the increasing `grid`, refined by a
golden-section search between the neighbours of the minimizer on the grid, to the
relative tolerance `rtol`, where both neighbours have finite values and the refined
point has a value no larger. With `largest = true`, the largest interior local
minimizer whose value is within half of the lowest interior local minimum replaces
the global one, as in Sheather's rule for least-squares cross-validation. Values that
are not finite count as `Inf`, and an error is thrown when none is. Returns
`(h, details)`, where `details` holds the `grid`, the `criterion` on it, the interior
local `minimizers`, and `atbound`, whether the minimizer is an end of the grid, beyond
which the minimum may lie.
"""
function gridminimize(f, grid::AbstractVector{<:Real}; largest::Bool = false, rtol::Real = 1e-3)
  isempty(grid) && throw(ArgumentError("the grid is empty"))
  all(isfinite, grid) || throw(ArgumentError("the grid must be finite"))
  all(i -> grid[i] < grid[i + 1], 1:length(grid)-1) || throw(ArgumentError("the grid must be increasing"))
  0 < rtol < 1 || throw(ArgumentError("rtol must be between 0 and 1, got $rtol"))
  return _gridminimize(f, collect(Float64, grid); largest, rtol = Float64(rtol))
end

"""
    sandwich(Z, v, e; vce = HCVariance(0), cluster = nothing)

The robust covariance matrix of the weighted least squares coefficients
`θ̂ = (ZᵀVZ)⁻¹ZᵀVy` of a local fit, from its design `Z`, one row per observation, its
weights `v` (the kernel weights times the observation weights, nonnegative) and its
residuals `e`,

    c (ZᵀVZ)⁻¹ (∑_g s_g s_gᵀ) (ZᵀVZ)⁻¹,   s_g = ∑_{i ∈ g} vᵢ ε̃ᵢ zᵢ,

where `g` runs over the clusters, or over the observations without them, and only
observations with positive weight enter. With `vce = HCVariance(type)`, the residuals
`ε̃ᵢ` and the factor `c` are those of [`HCVariance`](@ref), with the leverages
`hᵢᵢ = vᵢ zᵢᵀ(ZᵀVZ)⁻¹zᵢ` of this fit and `n` its number of observations with positive
weight. With `vce = ClusterVariance(type)` and the labels `cluster`, one per row, they
are those of [`ClusterVariance`](@ref), with the blocks `H_gg` of the hat matrix
`Z(ZᵀVZ)⁻¹ZᵀV`. When `Z` holds the powers of `(xᵢ - x₀)/h`, the first entry is the
squared standard error of the conventional estimate of [`LocalPolynomialRegression`](@ref)
at `x₀` with the same variance estimator.

The weights enter the middle of the sandwich squared, the convention for sampling
weights; for frequency weights, which stand for replicated observations, the middle
holds `wᵢ (kᵢ ε̃ᵢ)²` instead. The matrix is `NaN` when `ZᵀVZ` is singular, and its
entries are `NaN` where the variance estimator cannot be computed, as at a leverage
of one or with fewer than two clusters.
"""
function sandwich(Z::AbstractMatrix{<:Real}, v::AbstractVector{<:Real}, e::AbstractVector{<:Real};
                  vce::Union{HCVariance,ClusterVariance} = HCVariance(0), cluster = nothing)
  m, D = size(Z)
  length(v) == m || throw(DimensionMismatch("v has $(length(v)) weights for $m rows of Z"))
  length(e) == m || throw(DimensionMismatch("e has $(length(e)) residuals for $m rows of Z"))
  all(t -> isfinite(t) && t >= 0, v) || throw(ArgumentError("the weights v must be nonnegative and finite"))
  if vce isa ClusterVariance
    cluster === nothing && throw(ArgumentError("ClusterVariance needs the cluster of each row"))
    length(cluster) == m || throw(DimensionMismatch("cluster has $(length(cluster)) labels for $m rows of Z"))
  else
    cluster === nothing || throw(ArgumentError("with clusters, use ClusterVariance(type)"))
  end
  keep = findall(>(0), v)
  n = length(keep)
  Zk = Float64[Z[i, j] for i in keep, j in 1:D]
  vk, ek = Float64[v[i] for i in keep], Float64[e[i] for i in keep]
  G = Zk' * (vk .* Zk)
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(G); check = false)
  ok = n > 0 && LinearAlgebra.issuccess(C) && minimum(j -> C.factors[j, j]^2 / G[j, j], 1:D) > 1e-10
  ok || return fill(NaN, D, D)
  # with G = UᵀU, column i of A is U⁻ᵀzᵢ and of L the weights vᵢ G⁻¹zᵢ of θ̂
  A = C.U' \ permutedims(Zk)
  L = C.U \ (A .* permutedims(vk))
  V = if vce isa HCVariance
    t = vce.type
    f = map(eachindex(vk)) do i
      t == 0 && return 1.0
      t == 1 && return n > D ? sqrt(n / (n - D)) : NaN
      hii = vk[i] * sum(abs2, view(A, :, i))
      1 - hii > sqrt(eps()) || return NaN
      t == 2 ? 1 / sqrt(1 - hii) : 1 / (1 - hii)
    end
    M = L .* permutedims(ek .* f)
    M * M'
  else
    labels = [cluster[i] for i in keep]
    codes = Dict{eltype(labels),Int}()
    cl = [get!(codes, l, length(codes) + 1) for l in labels]
    ord = _clusterorder(cl)
    B = A .* permutedims(vk)   # Aᵀ_g B_g is the block H_gg of the hat matrix
    S = zeros(length(codes), D)
    for j in 1:D
      _, sums = _clustersums(cl, ord, view(L, j, :), ek, A, B, vce.type, :default)
      S[:, j] .= sums
    end
    _crscale(vce.type, n, length(codes), D, :default) .* (S' * S)
  end
  return (V + V') ./ 2
end

"""
    supt_critical(C; level = 0.95, draws = 10_000, rng = Random.default_rng())

The critical value of uniform bands for estimates with the covariance matrix `C`: the
`level` quantile of `maxⱼ |Zⱼ| / σⱼ` for `Z ~ N(0, C)` and `σⱼ² = Cⱼⱼ`, by simulation
from `draws` draws, kept between the pointwise and the Bonferroni critical values,
between which the exact one lies. Estimates with zero or `NaN` variance are left out,
and with one estimate the pointwise value is exact. This is the critical value of
[`conf_band`](@ref).
"""
function supt_critical(C::AbstractMatrix{<:Real}; level::Real = 0.95, draws::Integer = 10_000,
                       rng::Random.AbstractRNG = Random.default_rng())
  LinearAlgebra.checksquare(C)
  0 < level < 1 || throw(ArgumentError("level must be between 0 and 1, got $level"))
  draws >= 100 || throw(ArgumentError("draws must be at least 100, got $draws"))
  se = [C[j, j] > 0 ? sqrt(Float64(C[j, j])) : 0.0 for j in axes(C, 1)]
  return _supt(Matrix{Float64}(C), se, Float64(level), Int(draws), rng)
end
