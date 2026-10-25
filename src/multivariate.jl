## Multivariate kernel density estimation with a bandwidth matrix

# A sample in d dimensions: the observations as the columns of x, the weights and
# their cumulative sums, their sum and Kish's effective sample size
struct _MvData{W<:Union{Nothing,Vector{Float64}}}
  x::Matrix{Float64}
  w::W
  cumw::W
  sumw::Float64
  neff::Float64
end

nobs(d::_MvData) = size(d.x, 2)
_dimension(d::_MvData) = size(d.x, 1)

function _mvdata(x::AbstractMatrix, weights)
  Base.require_one_based_indexing(x)
  d, n = size(x)
  d >= 1 && n >= 1 || throw(ArgumentError("the sample needs at least one observation and one coordinate, got a $d × $n matrix"))
  _check_values(vec(x), "x")
  return _mvdata(Matrix{Float64}(x), weights, n)
end

function _mvdata(x::Tuple{Vararg{AbstractVector}}, weights)
  isempty(x) && throw(ArgumentError("the sample needs at least one coordinate"))
  n = length(x[1])
  X = Matrix{Float64}(undef, length(x), n)
  for (j, v) in enumerate(x)
    length(v) == n || throw(DimensionMismatch("the coordinates have $n and $(length(v)) values"))
    _check_values(v, "x[$j]")
    X[j, :] .= v
  end
  n >= 1 || throw(ArgumentError("the sample needs at least one observation"))
  return _mvdata(X, weights, n)
end

# As for SmoothingData, the weights are divided by a power of two where their
# products would overflow or underflow, only their ratios mattering, and the
# observations whose weight vanishes are dropped
function _mvdata(X::Matrix{Float64}, weights, n)
  size(X, 1) > n && @warn "the sample has more coordinates ($(size(X, 1))) than observations ($n); " *
    "the columns of x must be the observations"
  weights === nothing && return _MvData(X, nothing, nothing, Float64(n), Float64(n))
  length(weights) == n || throw(DimensionMismatch("weights has $(length(weights)) values for $n observations"))
  _check_values(weights, "weights")
  all(>=(0), weights) && any(>(0), weights) || throw(ArgumentError("weights must be nonnegative with a positive sum"))
  wf, _ = _scaledweights(weights)
  keep = findall(>(0), wf)
  w = wf[keep]
  cumw = cumsum(w)   # in order, so that a draw below the total never passes the last observation
  return _MvData(X[:, keep], w, cumw, cumw[end], cumw[end]^2 / sum(abs2, w))
end

# The weighted mean and the covariance matrix of the empirical distribution,
# ∑ wᵢ (xᵢ - x̄)(xᵢ - x̄)ᵀ / ∑ wᵢ, summed about the first observation so that a large
# offset of the data does not enter the rounding
function _mvmoments(d::_MvData)
  X, n, p = d.x, nobs(d), _dimension(d)
  x0 = X[:, 1]
  s1, s2 = zeros(p), zeros(p, p)
  for i in 1:n
    w = _weight(d.w, i)
    for a in 1:p
      ua = X[a, i] - x0[a]
      s1[a] += w * ua
      for b in 1:a
        s2[a, b] += w * ua * (X[b, i] - x0[b])
      end
    end
  end
  m = s1 ./ d.sumw
  S = [(s2[max(a, b), min(a, b)] - s1[a] * s1[b] / d.sumw) / d.sumw for a in 1:p, b in 1:p]
  return x0 .+ m, S
end

# the sample covariance matrix, corrected with Kish's effective sample size: divisor
# n - 1 with equal weights
function _mvcov(d::_MvData)
  n = d.neff
  n > 1 || throw(ArgumentError("a covariance matrix needs more than one observation"))
  return _mvmoments(d)[2] .* (n / (n - 1))
end

"""
    MultivariateKDE <: ContinuousMultivariateDistribution

A kernel density estimate in `d` dimensions with a bandwidth matrix,

    f̂(x) = ∑ᵢ wᵢ φ_H(x - xᵢ) / ∑ᵢ wᵢ,   φ_H(u) = (2π)^(-d/2) |H|^(-1/2) exp(-uᵀ H⁻¹ u / 2),

fitted with [`fit`](@ref)`(MultivariateKDE, x)`. The kernel is the normal density
with covariance matrix `H`, the bandwidth matrix, which can orient the smoothing
along any direction, not only along the axes. Observations of zero weight are
dropped. A `MultivariateKDE` is a continuous
multivariate distribution of Distributions.jl, so `pdf`, `logpdf`, `rand`, `mean`,
`cov`, `var`, `length` and `insupport` work as for any other distribution; `pdf(f, X)`
for a matrix `X` evaluates the estimate at its columns, in parallel when Julia runs
with several threads. `pdf(f, (g1, …, gd))` evaluates it on the grid of the vectors
`g1, …, gd` and returns an array of size `(length(g1), …, length(gd))`.
[`bandwidth`](@ref)`(f)` returns `H` and [`marginal`](@ref)`(f, j)` the estimate of
the density of coordinate `j`, a one-dimensional [`KDE`](@ref).
"""
struct MultivariateKDE{W<:Union{Nothing,Vector{Float64}},S<:BandwidthSelection} <:
       Distributions.ContinuousMultivariateDistribution
  data::_MvData{W}
  H::Matrix{Float64}
  L::LinearAlgebra.LowerTriangular{Float64,Matrix{Float64}}   # H = L Lᵀ
  z::Matrix{Float64}   # L⁻¹ xᵢ, the observations where the kernel is the standard normal density
  logc::Float64        # log((2π)^(-d/2) |H|^(-1/2) / ∑ wᵢ)
  selection::S
end

"""
    fit(MultivariateKDE, x; bandwidth = PlugInMatrix(), weights = nothing)

The kernel density estimate of the sample `x`, a `d × n` matrix whose columns are
the `n` observations, as for `fit(MvNormal, x)` and `pdf(d, x)` in Distributions.jl,
or a tuple of vectors `(x1, …, xd)`, one for each coordinate; `weights` are optional.
Note that elsewhere in this package, as in `local_polynomial`, a matrix of points has
one row per point; a distribution of Distributions.jl takes them as columns. A
sample with more coordinates than observations is probably transposed and gives a
warning. The
`bandwidth` is a bandwidth matrix `H`, the covariance matrix of the normal kernel,
which must be positive definite; a vector or tuple `(h1, …, hd)`, which gives the diagonal
matrix with entries `h1², …, hd²`; a positive number `h`, which gives `h² I`; or a
rule that chooses a matrix from the data: [`PlugInMatrix`](@ref), the default,
[`SCVMatrix`](@ref), [`LSCVMatrix`](@ref) or [`NormalScaleMatrix`](@ref). A number
or a vector are standard deviations, as the bandwidth of the Gaussian kernel in one
dimension is.

# Examples
```julia
julia> x = [0.0 1.0 0.5 2.0; 0.0 0.5 1.5 1.0];   # four observations in two dimensions

julia> f = fit(MultivariateKDE, x; bandwidth = [0.5, 0.8]);

julia> pdf(f, [1.0, 1.0]) ≈ sum(pdf(MvNormal(x[:, i], [0.25 0; 0 0.64]), [1.0, 1.0]) for i in 1:4) / 4
true
```
"""
function fit(::Type{MultivariateKDE}, x::Union{AbstractMatrix,Tuple{Vararg{AbstractVector}}};
             bandwidth = PlugInMatrix(), weights = nothing)
  d = _mvdata(x, weights)
  bandwidth === nothing && (bandwidth = PlugInMatrix())
  sel = _mvselection(bandwidth, d)
  return _mvkde(d, sel)
end

function _mvkde(d::_MvData, sel::BandwidthSelection)
  H = sel.h
  C = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H))
  L = LinearAlgebra.LowerTriangular(Matrix(C.L))
  p = _dimension(d)
  logc = -p / 2 * log(2π) - LinearAlgebra.logdet(C) / 2 - log(d.sumw)
  return MultivariateKDE(d, H, L, L \ d.x, logc, sel)
end

_mvselection(rule::BandwidthSelector, d::_MvData) = _selectmatrix(rule, d)
_mvselection(h, d::_MvData) = BandwidthSelection(_bandwidthmatrix(h, _dimension(d)), nothing, NamedTuple())

# The bandwidth matrix of a number, a vector of standard deviations or a matrix
function _bandwidthmatrix(h, p)
  if h isa Real
    0 < h < Inf || throw(ArgumentError("a bandwidth must be a positive number, got $h"))
    return Matrix{Float64}(LinearAlgebra.I, p, p) .* Float64(h)^2
  elseif h isa AbstractVector{<:Real} || h isa Tuple{Vararg{Real}}
    length(h) == p || throw(DimensionMismatch("$(length(h)) bandwidths for $p coordinates"))
    all(v -> 0 < v < Inf, h) || throw(ArgumentError("bandwidths must be positive numbers, got $(repr(h))"))
    return Matrix(LinearAlgebra.Diagonal(Float64[v^2 for v in h]))
  elseif h isa AbstractMatrix{<:Real}
    size(h) == (p, p) || throw(DimensionMismatch("a bandwidth matrix for $p coordinates must be $p × $p, got $(size(h))"))
    H = Matrix{Float64}(h)
    all(isfinite, H) || throw(ArgumentError("the bandwidth matrix must have finite entries"))
    isapprox(H, H'; rtol = 1e-10) || throw(ArgumentError("the bandwidth matrix must be symmetric"))
    H = (H .+ H') ./ 2
    LinearAlgebra.isposdef(H) || throw(ArgumentError("the bandwidth matrix must be positive definite"))
    return H
  end
  throw(ArgumentError("bandwidth must be a positive number, a vector of them, a positive definite " *
                      "matrix or a bandwidth matrix selector, got $(repr(h))"))
end

function Base.show(io::IO, f::MultivariateKDE)
  print(io, "MultivariateKDE(", length(f), " dimensions, ", nobs(f.data), " observations",
        f.data.w === nothing ? "" : ", weighted", ")")
end

bandwidth(f::MultivariateKDE) = f.H
nobs(f::MultivariateKDE) = nobs(f.data)
Base.length(f::MultivariateKDE) = size(f.H, 1)
Base.eltype(::Type{<:MultivariateKDE}) = Float64

# log ∑ᵢ wᵢ exp(-|y - zᵢ|² / 2), summed in one pass about the largest term so far
# and rescaled when a larger one appears, so that it stays finite where the sum
# underflows; -Inf where every distance overflows
function _logkernelsum(f::MultivariateKDE, y::AbstractVector)
  z, w = f.z, f.data.w
  p, n = size(z)
  T = float(eltype(y))
  rmin, s = T(Inf), zero(T)
  @inbounds for i in 1:n
    r = zero(T)
    for a in 1:p
      r += (y[a] - z[a, i])^2
    end
    if r == Inf
      continue
    elseif r < rmin
      s = s * exp(-(rmin - r) / 2) + _weight(w, i)
      rmin = r
    else
      s += _weight(w, i) * exp(-(r - rmin) / 2)
    end
  end
  return isfinite(rmin) ? log(s) - rmin / 2 : T(-Inf)
end

# NaN at a point with a NaN coordinate, -Inf at one with an infinite coordinate
function Distributions._logpdf(f::MultivariateKDE, x::AbstractVector{<:Real})
  any(isnan, x) && return float(eltype(x))(NaN)
  all(isfinite, x) || return float(eltype(x))(-Inf)
  return f.logc + _logkernelsum(f, f.L \ x)
end

# the estimate at the columns of x, in tasks that run in parallel with several
# threads; Distributions evaluates a matrix of points column by column, so pdf and
# logpdf of a matrix come here
function Distributions._logpdf!(out::AbstractArray{<:Real}, f::MultivariateKDE, x::AbstractMatrix{<:Real})
  if !(eltype(x) <: Union{AbstractFloat,Integer})   # dual numbers, for example
    for j in axes(x, 2)
      out[j] = Distributions._logpdf(f, view(x, :, j))
    end
    return out
  end
  # columns with infinite or NaN entries are evaluated by the scalar method
  finite = [all(isfinite, view(x, :, j)) for j in axes(x, 2)]
  y = f.L \ Matrix{Float64}(ifelse.(isfinite.(x), x, 0.0))
  _foreachchunk(collect(axes(y, 2))) do chunk
    for j in chunk
      out[j] = finite[j] ? f.logc + _logkernelsum(f, view(y, :, j)) : Distributions._logpdf(f, view(x, :, j))
    end
  end
  return out
end

function _checkmvpoints(f::MultivariateKDE, x::AbstractMatrix)
  size(x, 1) == length(f) || throw(DimensionMismatch("points for a $(length(f))-dimensional estimate need " *
    "$(length(f)) rows, got $(size(x, 1)); the columns are the points"))
end

logpdf(f::MultivariateKDE, x::AbstractMatrix{<:Real}) =
  (_checkmvpoints(f, x); Distributions._logpdf!(Vector{float(eltype(x))}(undef, size(x, 2)), f, x))
pdf(f::MultivariateKDE, x::AbstractMatrix{<:Real}) = exp.(logpdf(f, x))

"""
    pdf(f::MultivariateKDE, grid::Tuple)

The estimate on the grid of the vectors `grid = (g1, …, gd)`: an array of size
`(length(g1), …, length(gd))` with the estimate at `(g1[i1], …, gd[id])` in entry
`(i1, …, id)`.
"""
function pdf(f::MultivariateKDE, grid::Tuple{Vararg{AbstractVector{<:Real}}})
  p = length(f)
  length(grid) == p || throw(DimensionMismatch("a grid for $p coordinates needs $p vectors, got $(length(grid))"))
  dims = map(length, grid)
  pts = Matrix{Float64}(undef, p, prod(dims))
  for (j, I) in enumerate(CartesianIndices(dims))
    for a in 1:p
      pts[a, j] = grid[a][I[a]]
    end
  end
  return reshape(pdf(f, pts), dims)
end

Distributions.insupport(f::MultivariateKDE, x::AbstractVector{<:Real}) = length(x) == length(f) && all(isfinite, x)

mean(f::MultivariateKDE) = _mvmoments(f.data)[1]
cov(f::MultivariateKDE) = _mvmoments(f.data)[2] .+ f.H
var(f::MultivariateKDE) = LinearAlgebra.diag(cov(f))

# the smoothed bootstrap: an observation drawn by weight plus a draw from the kernel
function Distributions._rand!(rng::AbstractRNG, f::MultivariateKDE, x::AbstractVector{<:Real})
  d = f.data
  n = nobs(d)
  i = d.w === nothing ? rand(rng, 1:n) : min(searchsortedlast(d.cumw, rand(rng) * d.sumw) + 1, n)
  x .= view(d.x, :, i) .+ f.L * randn(rng, length(f))
  return x
end

"""
    marginal(f::MultivariateKDE, j)

The estimate of the density of coordinate `j`: the one-dimensional [`KDE`](@ref)
of that coordinate with the Gaussian kernel and bandwidth `√H[j, j]`, which is the
marginal density of `f` exactly.
"""
function marginal(f::MultivariateKDE, j::Integer)
  1 <= j <= length(f) || throw(ArgumentError("coordinate $j of a $(length(f))-dimensional estimate"))
  return fit(KDE, f.data.x[j, :]; kernel = GaussianKernel(), bandwidth = sqrt(f.H[j, j]), weights = f.data.w)
end
