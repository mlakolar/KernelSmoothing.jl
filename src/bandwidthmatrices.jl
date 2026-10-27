## Bandwidth matrices for multivariate kernel density estimation

"""
    MatrixSelector <: BandwidthSelector

Supertype of the rules that choose the bandwidth matrix of a
[`MultivariateKDE`](@ref): [`NormalScaleMatrix`](@ref), [`PlugInMatrix`](@ref),
[`SCVMatrix`](@ref) and [`LSCVMatrix`](@ref). [`select_bandwidth`](@ref)`(rule, x;
weights)` applies a rule to a sample `x`, a `d × n` matrix whose columns are the
observations or a tuple of vectors, and returns a [`BandwidthSelection`](@ref) whose
bandwidth is the matrix.

With `diagonal = true`, a rule chooses a diagonal matrix, which smooths along the
axes only. The plug-in and cross-validation rules work on the data transformed by
`S^(-1/2)`, where `S` is the sample covariance matrix (`pre = :sphere`), or by the
inverse standard deviations (`pre = :scale`, the choice for diagonal matrices),
and transform the matrix back. The plug-in and smoothed cross-validation rules
depend on this choice through their pilot bandwidths, which are multiples of the
identity in the transformed coordinates.

The criteria are sums over pairs of observations. In two and three dimensions,
the plug-in and cross-validation rules can approximate these sums by linear
binning on one grid in the transformed coordinates. This is the default above
500 observations in two dimensions and, for plug-in and smoothed cross-validation,
1000 in three, counting only positive weights. Least-squares cross-validation
uses exact sums by default in three dimensions (see [`LSCVMatrix`](@ref)).
`binned = false` uses exact sums, which take a time of order `n²`. Both
forms of the sums run in parallel when Julia runs with several threads.
"""
abstract type MatrixSelector <: BandwidthSelector end

_binning_dimensions(::MatrixSelector) = (2, 3)
_matrixbins(rule::MatrixSelector, y, d::_MvData; loo = false) =
  _matrixbins(y, d, rule.binned, rule.gridsize; loo, default_dims = _binning_dimensions(rule))

select_bandwidth(rule::MatrixSelector, x::Union{AbstractMatrix,Tuple{Vararg{AbstractVector}}}; weights = nothing) =
  _selectmatrix(rule, _mvdata(x, weights))

_selectmatrix(rule::BandwidthSelector, d) = throw(ArgumentError(
  "$rule chooses a bandwidth in one dimension; a multivariate estimate needs a MatrixSelector, such as PlugInMatrix()"))
select_bandwidth(rule::MatrixSelector, ::SmoothingData, ::AbstractKernel; kwargs...) = throw(ArgumentError(
  "$rule chooses a bandwidth matrix for MultivariateKDE; in one dimension use a rule such as SheatherJones()"))

# The number of tasks of the sums over pairs: it depends on n only, not on the number
# of threads, so that the order of the summation, and with it the result to the last
# bit, does not depend on the threads
_pairtasks(n) = clamp(n ÷ 64, 1, 256)

function _checkpre(pre::Symbol, diagonal::Bool)
  pre in (:sphere, :scale) || throw(ArgumentError("pre must be :sphere or :scale, got :$pre"))
  diagonal && pre === :sphere &&
    throw(ArgumentError("a diagonal bandwidth matrix needs pre = :scale, which keeps the axes"))
  return pre
end

function _checkstages(stages)
  stages in (1, 2) || throw(ArgumentError("stages must be 1 or 2, got $stages"))
  return Int(stages)
end

# the sample covariance matrix must be well conditioned: its smallest eigenvalue
# above 1e-12 of its largest
function _checkcov(S)
  λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(S))
  λ[1] > 1e-12 * λ[end] || throw(ArgumentError("the sample covariance matrix is singular or nearly so, so the " *
    "data do not span all coordinates; the columns of x must be the observations"))
end

# The pre-transformation: the data are y = A x, with A = S^(-1/2), the symmetric
# square root, or the inverse standard deviations; a bandwidth matrix Hy for y is
# H = A⁻¹ Hy A⁻ᵀ for x. Returns A, y and the covariance matrix of y.
function _pretransform(d::_MvData, pre::Symbol)
  S = _mvcov(d)
  _checkcov(S)
  A = if pre === :sphere
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(S))
    E.vectors * LinearAlgebra.Diagonal(1 ./ sqrt.(E.values)) * E.vectors'
  else
    Matrix(LinearAlgebra.Diagonal(1 ./ sqrt.(LinearAlgebra.diag(S))))
  end
  return A, A * d.x, A * S * A'
end

_backtransform(A, Hy) = (B = inv(A); H = B * Hy * B'; (H .+ H') ./ 2)

# the normal scale matrix (4 / ((d + 2) n))^(2 / (d + 4)) Σ, or its diagonal
_normalscale(Σ, n, diagonal) =
  (4 / ((size(Σ, 1) + 2) * n))^(2 / (size(Σ, 1) + 4)) .* (diagonal ? Matrix(LinearAlgebra.Diagonal(Σ)) : Σ)

## Normal densities and their derivatives

# the multi-indices r with |r| = j in p dimensions, as the columns of a matrix
function _multiindices(p::Int, j::Int)
  p == 1 && return reshape([j], 1, 1)
  cols = Vector{Int}[]
  for r1 in j:-1:0, rest in eachcol(_multiindices(p - 1, j - r1))
    push!(cols, vcat(r1, rest))
  end
  return reduce(hcat, cols)
end

# the probabilists' Hermite polynomials He₀(u), …, He_J(u)
function _hermite!(h, u, J)
  h[1] = 1.0
  J >= 1 && (h[2] = u)
  for m in 2:J
    h[m + 1] = u * h[m] - (m - 1) * h[m - 1]
  end
  return h
end

# He_m(0): zero for odd m, (-1)^(m/2) (m - 1)!! for even m
_hermite0(m) = isodd(m) ? 0.0 : (-1)^(m ÷ 2) * prod(1:2:m-1; init = 1.0)

# The derivative ∂ʳφ_Σ(0) of the normal density with covariance Σ at zero, by
# Isserlis' theorem: (-1)^(|r|/2) φ_Σ(0) times the sum over the perfect matchings of
# the indices of r of the products of the entries of Σ⁻¹ that the pairs pick
function _normalderivative0(Σ::AbstractMatrix, r)
  j = sum(r)
  isodd(j) && return 0.0
  P = inv(Σ)
  idx = reduce(vcat, [fill(a, r[a]) for a in eachindex(r)]; init = Int[])
  φ0 = (2π)^(-size(Σ, 1) / 2) / sqrt(LinearAlgebra.det(Σ))
  return (-1)^(j ÷ 2) * φ0 * _matchings(P, idx)
end

function _matchings(P, idx)
  isempty(idx) && return 1.0
  a = idx[1]
  s = 0.0
  for k in 2:length(idx)
    rest = [idx[m] for m in 2:length(idx) if m != k]
    s += P[a, idx[k]] * _matchings(P, rest)
  end
  return s
end

## Estimates of the functionals ψ_r = ∫ ∂ʳf(x) f(x) dx

# The estimates ψ̂_r = ∑ᵢ ∑ⱼ wᵢ wⱼ ∂ʳφ_{g²I}(yᵢ - yⱼ) / (∑ wᵢ)² for the multi-indices r
# (the columns of R, all of the same even order j), the pairs i = j included. With
# u = (yᵢ - yⱼ) / g, ∂ʳφ_{g²I} = (2π)^(-p/2) g^(-p-j) ∏ₐ He_{rₐ}(uₐ) exp(-|u|²/2) for
# even j. Returns a dictionary from the multi-indices to the estimates.
function _psihat(y::Matrix{Float64}, w, sumw, neff, g::Float64, R::Matrix{Int}, bins = nothing)
  p, n = size(y)
  j = sum(view(R, :, 1))
  m = size(R, 2)
  # the terms i = j: ∑ wᵢ² ∂ʳφ(0) / (∑ wᵢ)² = ∂ʳφ(0) / neff
  diag0 = [prod(_hermite0(R[a, k]) for a in 1:p) for k in 1:m]
  s = if bins === nothing
    ntasks = _pairtasks(n)
    parts = _fetchall([Threads.@spawn _psipairs(y, w, g, R, j, t, ntasks) for t in 1:ntasks], Vector{Float64})
    reduce(+, parts)
  else
    _psibins(bins, g, R, j)
  end
  c = (2π)^(-p / 2) * g^(-p - j)
  v = c .* (diag0 ./ neff .+ 2 .* s ./ sumw^2)
  return Dict(R[:, k] => v[k] for k in 1:m)
end

# ∑ over the pairs i < j with i ≡ t (mod ntasks) of wᵢ wⱼ ∏ₐ He_{rₐ}(uₐ) exp(-|u|²/2);
# the striped split balances the triangle of pairs between the tasks
function _psipairs(y, w, g, R, j, t, ntasks)
  p, n = size(y)
  m = size(R, 2)
  s = zeros(m)
  h = zeros(j + 1, p)
  u = zeros(p)
  @inbounds for i in t:ntasks:n
    wi = _weight(w, i)
    wi == 0 && continue
    for k in i+1:n
      r2 = 0.0
      for a in 1:p
        u[a] = (y[a, i] - y[a, k]) / g
        r2 += u[a]^2
      end
      e = wi * _weight(w, k) * exp(-r2 / 2)
      e == 0 && continue
      for a in 1:p
        _hermite!(view(h, :, a), u[a], j)
      end
      for c in 1:m
        v = e
        for a in 1:p
          v *= h[R[a, c] + 1, a]
        end
        s[c] += v
      end
    end
  end
  return s
end

# The normal-reference values of the functionals of order j for the covariance Σ,
# ψ_r = ∂ʳφ_{2Σ}(0)
_psinormal(Σ, R) = Dict(R[:, k] => _normalderivative0(2 .* Σ, R[:, k]) for k in axes(R, 2))

# The SAMSE pilot of Duong and Hazelton (2003) for the functionals of order j (the
# columns of R) from the functionals of order j + 2: with G = g² I, ψ̂_r has the
# asymptotic bias A_r g^(-p-j) / n + g² B_r / 2, with A_r = ∂ʳφ(0) for the standard
# normal density and B_r = ∑ₐ ψ_{r+2eₐ}, and g minimizes the sum of the squared biases
# over the r whose entries are all even: with t = n g^(p+j+2), the root of
# C t² + (2 - p - j) B t - 2 (p + j) A = 0, where A = ∑ A_r², B = ∑ A_r B_r and C = ∑ B_r².
function _samse(R::Matrix{Int}, psinext::Dict{Vector{Int},Float64}, n)
  p = size(R, 1)
  j = sum(view(R, :, 1))
  A = B = C = 0.0
  for k in axes(R, 2)
    r = R[:, k]
    all(iseven, r) || continue
    Ar = (2π)^(-p / 2) * prod(_hermite0(v) for v in r)
    Br = 0.0
    for a in 1:p
      r[a] += 2
      Br += psinext[r]
      r[a] -= 2
    end
    A += Ar^2
    B += Ar * Br
    C += Br^2
  end
  b = (2 - p - j) * B
  t = (-b + sqrt(b^2 + 8 * (p + j) * A * C)) / (2C)
  return (t / n)^(1 / (p + j + 2))
end

# ks passes the estimates of order 6 to the pilot of order 4 as the whole tensor of
# derivatives, in the order of vec(D^⊗6), where the pilot reads them in the order of
# the distinct multi-indices: the u-th of these gets the u-th entry of the tensor
function _ksorder(psi::Dict{Vector{Int},Float64}, R::Matrix{Int})
  p, j = size(R, 1), sum(view(R, :, 1))
  function counts(k)
    c, q = zeros(Int, p), k - 1
    for _ in 1:j
      c[q % p + 1] += 1
      q ÷= p
    end
    return c
  end
  return Dict(R[:, u] => psi[counts(u)] for u in axes(R, 2))
end

# The pilots and estimates of the plug-in chain: the functionals of order
# 2(stages + 2) come from the normal reference for Σ, and each stage chooses the
# SAMSE pilot of the next lower order from the functionals above it, estimates the
# functionals of that order with it and passes them on. Returns the estimates of
# each order and the pilot of each order.
function _psichain(y, d::_MvData, Σ, stages::Int, compat::Bool, bins = nothing)
  p = size(y, 1)
  top = 4 + 2 * stages
  psi = Dict{Int,Dict{Vector{Int},Float64}}(top => _psinormal(Σ, _multiindices(p, top)))
  pilots = Dict{Int,Float64}()
  for j in top-2:-2:4
    R = _multiindices(p, j)
    above = psi[j + 2]
    compat && j + 2 < top && (above = _ksorder(above, _multiindices(p, j + 2)))
    pilots[j] = _samse(R, above, d.neff)
    psi[j] = _psihat(y, d.w, d.sumw, d.neff, pilots[j], R, bins)
  end
  return psi, pilots
end

# the tensor ψ_{abce} of the functionals of order 4 from the distinct ones
function _psitensor(psi4::Dict{Vector{Int},Float64}, p)
  T = zeros(p, p, p, p)
  for I in CartesianIndices(T)
    r = zeros(Int, p)
    for a in Tuple(I)
      r[a] += 1
    end
    T[I] = psi4[r]
  end
  return T
end

## Minimizing a criterion over bandwidth matrices

# The bandwidth matrix of the parameters θ: H = L Lᵀ, with L lower triangular, the
# logarithms of its diagonal and its other entries column by column in θ; or, for a
# diagonal matrix, H = diag(exp.(2θ))
function _hfromtheta(θ, p, diagonal)
  diagonal && return Matrix(LinearAlgebra.Diagonal(exp.(2 .* θ))), nothing
  L = zeros(p, p)
  k = 0
  for b in 1:p, a in b:p
    k += 1
    L[a, b] = a == b ? exp(θ[k]) : θ[k]
  end
  return L * L', L
end

function _thetafromh(H, diagonal)
  diagonal && return log.(LinearAlgebra.diag(H)) ./ 2
  L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H)).L
  p = size(H, 1)
  return [a == b ? log(L[a, b]) : L[a, b] for b in 1:p for a in b:p]
end

# the gradient in θ from the gradient G in the entries of H (symmetric)
function _thetagradient(G, H, L, diagonal)
  diagonal && return 2 .* LinearAlgebra.diag(G) .* LinearAlgebra.diag(H)
  GL = 2 .* (G * L)
  p = size(H, 1)
  return [a == b ? GL[a, b] * L[a, b] : GL[a, b] for b in 1:p for a in b:p]
end

# Minimizes a criterion over bandwidth matrices from H0 by BFGS in θ with a
# backtracking line search; fg(H) returns the criterion and its gradient in the
# entries of H. Stops when neither the quasi-Newton step nor, after a restart, the
# gradient step lowers the criterion by more than reltol of its value, which near
# the minimum locates θ to about √reltol.
function _minimizematrix(fg, H0, diagonal; maxiter = 500, reltol = 1e-12)
  p = size(H0, 1)
  # a trial matrix that is singular to working precision or puts the criterion
  # outside its domain is rejected as a step to an infinite criterion
  function fgθ(t)
    local Ht, Lt = _hfromtheta(t, p, diagonal)
    local ft, Gt = try
      fg(Ht)
    catch e
      e isa Union{LinearAlgebra.PosDefException,LinearAlgebra.SingularException,DomainError} || rethrow()
      Inf, fill(NaN, p, p)
    end
    local gt = _thetagradient(Gt, Ht, Lt, diagonal)
    return all(isfinite, gt) ? ft : oftype(ft, Inf), gt
  end
  θ = _thetafromh(H0, diagonal)
  f, g = fgθ(θ)
  isfinite(f) || throw(ArgumentError("the criterion or its gradient is not finite at the starting bandwidth matrix"))
  m = length(θ)
  B = Matrix{Float64}(LinearAlgebra.I, m, m)   # the inverse Hessian
  iter, restarted, converged = 0, false, false
  while iter < maxiter
    iter += 1
    dir = -(B * g)
    slope = LinearAlgebra.dot(g, dir)
    if !(slope < 0)   # not a descent direction: restart from the gradient
      B = Matrix{Float64}(LinearAlgebra.I, m, m)
      dir, slope = -g, -LinearAlgebra.dot(g, g)
      slope < 0 || (converged = true; break)
    end
    # steps of the logarithms of the scales of at most 2
    α = min(1.0, 2.0 / maximum(abs, dir))
    θnew, fnew, gnew = θ, f, g
    for _ in 1:60
      θnew = θ .+ α .* dir
      fnew, gnew = fgθ(θnew)
      isfinite(fnew) && fnew <= f + 1e-4 * α * slope && break
      α /= 2
    end
    if !(isfinite(fnew) && fnew < f - reltol * abs(f))
      # no progress: once more from the gradient, then stop
      (restarted || B == LinearAlgebra.I) && (converged = true; break)
      B, restarted = Matrix{Float64}(LinearAlgebra.I, m, m), true
      continue
    end
    restarted = false
    s, yv = θnew .- θ, gnew .- g
    sy = LinearAlgebra.dot(s, yv)
    if sy > 1e-12 * LinearAlgebra.norm(s) * LinearAlgebra.norm(yv)
      ρ = 1 / sy
      V = LinearAlgebra.I - ρ .* (s * yv')
      B = V * B * V' .+ ρ .* (s * s')
    end
    θ, f, g = θnew, fnew, gnew
  end
  converged || @warn "the minimization of the bandwidth criterion stopped after $iter iterations without converging"
  H, _ = _hfromtheta(θ, p, diagonal)
  return (H .+ H') ./ 2, f, iter, converged
end

## The normal scale rule

"""
    NormalScaleMatrix(; diagonal = false)

The bandwidth matrix that minimizes the asymptotic mean integrated squared error
when the density is normal,

    H = (4 / ((d + 2) n))^(2 / (d + 4)) S,

where `S` is the sample covariance matrix and `n` the effective number of
observations. With `diagonal = true`, the diagonal matrix that minimizes the same
error, which is smaller than the diagonal of the matrix above when the coordinates
are correlated. The rule oversmooths densities that are far from normal; it starts
the other rules. In one dimension it is [`NormalReference`](@ref)`(robust = false)`
for the Gaussian kernel.
"""
struct NormalScaleMatrix <: MatrixSelector
  diagonal::Bool
end

NormalScaleMatrix(; diagonal::Bool = false) = NormalScaleMatrix(diagonal)

function _selectmatrix(rule::NormalScaleMatrix, d::_MvData)
  S = _mvcov(d)
  _checkcov(S)
  p, n = _dimension(d), d.neff
  H = _normalscale(S, n, rule.diagonal)
  if rule.diagonal && p > 1
    # in the coordinates scaled to unit variances, where the criterion is well scaled
    A, _, R = _pretransform(d, :scale)
    T = _psitensor(_psinormal(R, _multiindices(p, 4)), p)
    Hy = _minimizematrix(Hd -> _picriterion(Hd, T, (4π)^(-p / 2) / n), _normalscale(R, n, true), true)[1]
    H = _backtransform(A, Hy)
  end
  return BandwidthSelection(H, rule, (cov = S, n = n))
end

## The plug-in rule

# PI(H) = c0 |H|^(-1/2) + ∑ H_ab H_ce T_abce / 4 and its gradient in the entries of H
function _picriterion(H, T, c0)
  p = size(H, 1)
  D = exp(-LinearAlgebra.logdet(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(H))) / 2)
  TH = [sum(T[a, b, c, e] * H[c, e] for c in 1:p, e in 1:p) for a in 1:p, b in 1:p]
  return c0 * D + sum(H .* TH) / 4, -(c0 * D / 2) .* inv(H) .+ TH ./ 2
end

"""
    PlugInMatrix(; diagonal = false, stages = 2, pre = diagonal ? :scale : :sphere,
                   compat = false, binned = nothing, gridsize = nothing)

The plug-in bandwidth matrix of Wand and Jones (1994) and Duong and Hazelton
(2003): the minimizer of the estimated asymptotic mean integrated squared error,

    PI(H) = (4π)^(-d/2) |H|^(-1/2) / n + ∑ₐ ∑_b ∑_c ∑_e H_ab H_ce ψ̂_abce / 4,

where `ψ̂_abce` estimates the functional `∫ ∂⁴f/∂xₐ∂x_b∂x_c∂x_e (x) f(x) dx` of the
density by

    ψ̂_r = ∑ᵢ ∑ⱼ wᵢ wⱼ ∂ʳφ_G(xᵢ - xⱼ) / (∑ᵢ wᵢ)²,

the pairs `i = j` included, with the pilot bandwidth matrix `G = g² I` in the
transformed coordinates (see [`MatrixSelector`](@ref)). The pilot `g` minimizes the
sum of the squared asymptotic biases of the estimates whose indices are all even
(SAMSE, Duong and Hazelton 2003). These biases depend on the functionals of order
6: with `stages = 2` they are estimated in turn, with a pilot chosen in the same
way from normal-reference functionals of order 8, and with `stages = 1` they come
from the normal reference. `compat = true` reads the estimates of order 6 in the
order in which the R package ks reads them, which matches `ks::Hpi(pilot =
"samse", binned = FALSE)` to its minimization accuracy when `binned = false`
(see [Comparison with R](@ref)); it changes nothing for `stages = 1` and is
independent of binning.

`binned = true` approximates the pair sums, including those of every pilot, by
linear binning in two or three dimensions. With `binned = nothing`, it is used
above 500 observations with positive weight in two dimensions and 1000 in three.
`gridsize` is the number of nodes per axis, at least 3, defaulting to 151 in two
dimensions and 31 in three. `binned = false` uses exact sums and ignores
`gridsize`. The smeared self-pairs are removed and the exact diagonal terms kept.

The bandwidths converge to the optimal ones at the relative rate `n^(-2/(d+6))`.
The details record the pilots by order, the minimized criterion and the number of
iterations, and `binned`, `gridsize`, `spacing` and `resolution`. The resolution
gives the smallest selected kernel standard deviation and the smallest pilot in
units of the largest grid spacing in the transformed coordinates; a value below
1 warns to increase `gridsize` or use `binned = false`.
"""
struct PlugInMatrix <: MatrixSelector
  diagonal::Bool
  stages::Int
  pre::Symbol
  compat::Bool
  binned::Union{Nothing,Bool}
  gridsize::Union{Nothing,Int}

  PlugInMatrix(diagonal::Bool, stages::Integer, pre::Symbol, compat::Bool,
               binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
    new(diagonal, _checkstages(stages), _checkpre(pre, diagonal), compat, binned, _checkgridsize(gridsize))
end

PlugInMatrix(; diagonal::Bool = false, stages::Integer = 2, pre::Symbol = diagonal ? :scale : :sphere,
             compat::Bool = false, binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
  PlugInMatrix(diagonal, stages, pre, compat, binned, gridsize)

function _selectmatrix(rule::PlugInMatrix, d::_MvData)
  A, y, Σ = _pretransform(d, rule.pre)
  p, n = _dimension(d), d.neff
  bins = _matrixbins(rule, y, d)
  psi, pilots = _psichain(y, d, Σ, rule.stages, rule.compat, bins)
  T = _psitensor(psi[4], p)
  c0 = (4π)^(-p / 2) / n
  Hy, crit, iter, converged = _minimizematrix(H -> _picriterion(H, T, c0), _normalscale(Σ, n, rule.diagonal),
                                              rule.diagonal)
  # the criteria in the original coordinates: integrated squared errors scale with |A|
  details = (pilots = pilots, criterion = crit * abs(LinearAlgebra.det(A)), iterations = iter, converged = converged)
  return BandwidthSelection(_backtransform(A, Hy), rule, merge(details, _binningdetails(bins, Hy, minimum(values(pilots)))))
end

## Cross-validation

# ∑ over the pairs i < j of cᵢⱼ φ_Σ(yᵢ - yⱼ) and ∑ cᵢⱼ φ_Σ(yᵢ - yⱼ) (yᵢ - yⱼ)(yᵢ - yⱼ)ᵀ for
# each covariance matrix Σ of Ss, with cᵢⱼ = wᵢ wⱼ, or for loo = true the weights of
# the leave-one-out sums, wᵢ wⱼ (1 / (W - wᵢ) + 1 / (W - wⱼ)) / W
function _gausspairs(y, w, W, Ss::Vector{Matrix{Float64}}, loo::Vector{Bool}, grad::Bool, bins = nothing)
  bins === nothing || return _gaussbins(bins, Ss, loo, grad)
  p, n = size(y)
  K = length(Ss)
  Rinv = [inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S)).L) for S in Ss]
  c = [(2π)^(-p / 2) / sqrt(LinearAlgebra.det(S)) for S in Ss]
  ntasks = _pairtasks(n)
  parts = _fetchall([Threads.@spawn _gausspairstask(y, w, W, Rinv, loo, grad, t, ntasks) for t in 1:ntasks],
                    Tuple{Vector{Float64},Vector{Matrix{Float64}}})
  s = zeros(K)
  M = [zeros(p, p) for _ in 1:K]
  for (sk, Mk) in parts
    s .+= sk
    grad && (M .+= Mk)
  end
  # the tasks fill the lower triangles
  return c .* s, [c[k] .* (M[k] .+ M[k]' .- LinearAlgebra.Diagonal(M[k])) for k in 1:K]
end

function _gausspairstask(y, w, W, Rinv, loo, grad, t, ntasks)
  p, n = size(y)
  K = length(Rinv)
  s = zeros(K)
  M = [zeros(p, p) for _ in 1:K]
  Δ = zeros(p)
  @inbounds for i in t:ntasks:n
    wi = _weight(w, i)
    wi == 0 && continue
    for j in i+1:n
      wj = _weight(w, j)
      wj == 0 && continue
      for a in 1:p
        Δ[a] = y[a, i] - y[a, j]
      end
      for k in 1:K
        R = Rinv[k]
        r2 = 0.0
        for a in 1:p
          va = 0.0
          for b in 1:a
            va += R[a, b] * Δ[b]
          end
          r2 += va^2
        end
        cij = loo[k] ? wi * wj * (1 / (W - wi) + 1 / (W - wj)) / W : wi * wj
        e = cij * exp(-r2 / 2)
        s[k] += e
        if grad
          Mk = M[k]
          for b in 1:p, a in b:p
            Mk[a, b] += e * Δ[a] * Δ[b]
          end
        end
      end
    end
  end
  return s, M
end

# ∂φ_Σ(Δ)/∂Σ summed: (Σ⁻¹ M Σ⁻¹ - s Σ⁻¹) / 2 for the sums s and M of _gausspairs
function _gaussgradient(S, s, M)
  P = inv(S)
  return (P * M * P .- s .* P) ./ 2
end

"""
    LSCVMatrix(; diagonal = false, binned = nothing, gridsize = nothing)

The least-squares cross-validation bandwidth matrix (Rudemo 1982, Bowman 1984, and
Duong and Hazelton 2005 for matrices): the minimizer of an unbiased estimate of the
integrated squared error up to a constant,

    LSCV(H) = ∫ f̂² - 2 ∑ᵢ wᵢ f̂₋ᵢ(xᵢ) / ∑ᵢ wᵢ
            = ∑ᵢ ∑ⱼ wᵢ wⱼ φ_{2H}(xᵢ - xⱼ) / W² - 2 ∑ᵢ ∑_{j≠i} wᵢ wⱼ φ_H(xᵢ - xⱼ) / (W (W - wᵢ)),

where `W = ∑ wᵢ` and `f̂₋ᵢ` leaves observation `i` out; with equal weights,
`∑ᵢ ∑ⱼ φ_{2H}(xᵢ - xⱼ) / n² - 2 ∑ᵢ ∑_{j≠i} φ_H(xᵢ - xⱼ) / (n (n - 1))`. It needs no pilot
and adapts to densities far from normal, but its bandwidths vary much more from
sample to sample than plug-in ones, and the criterion can have several local
minima: the search starts from the normal scale matrix and finds a minimum near
it. The exact criterion does not depend on the pre-transformation.

`binned = true` approximates the pair sums by linear binning in two or three
dimensions, with the smeared self-pairs removed and the exact diagonal kept.
With `binned = nothing`, it is used only in two dimensions, above 500 observations
with positive weight; three dimensions use exact sums by default. A binned LSCV
criterion can develop a spurious minimum as `H → 0` when observations share grid
cells. In three dimensions the LSCV bandwidth is often below one default grid
spacing; a hybrid fix is tracked in issue #101.

`gridsize` is the number of nodes per axis, at least 3, defaulting to 151 in two
dimensions and 31 in three. `binned = false`
uses exact sums and ignores `gridsize`. The details record `binned`, `gridsize`,
`spacing` and `resolution`, whose `selected` value is the smallest kernel standard
deviation in units of the largest grid spacing in the transformed coordinates.
A value below 1 warns to increase `gridsize` or use `binned = false`.
"""
struct LSCVMatrix <: MatrixSelector
  diagonal::Bool
  binned::Union{Nothing,Bool}
  gridsize::Union{Nothing,Int}

  LSCVMatrix(diagonal::Bool, binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
    new(diagonal, binned, _checkgridsize(gridsize))
end

LSCVMatrix(; diagonal::Bool = false, binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
  LSCVMatrix(diagonal, binned, gridsize)

_binning_dimensions(::LSCVMatrix) = (2,)

function _selectmatrix(rule::LSCVMatrix, d::_MvData)
  count(i -> _weight(d.w, i) > 0, 1:nobs(d)) >= 2 ||
    throw(ArgumentError("least-squares cross-validation needs at least two observations"))
  A, y, Σ = _pretransform(d, rule.diagonal ? :scale : :sphere)
  p, n, W = _dimension(d), d.neff, d.sumw
  bins = _matrixbins(rule, y, d; loo = true)
  Hy, crit, iter, converged = _minimizematrix(H -> _lscvcriterion(y, d.w, W, n, H, bins), _normalscale(Σ, n, rule.diagonal),
                                              rule.diagonal)
  details = (criterion = crit * abs(LinearAlgebra.det(A)), iterations = iter, converged = converged)
  return BandwidthSelection(_backtransform(A, Hy), rule, merge(details, _binningdetails(bins, Hy)))
end

# the criterion of LSCVMatrix and its gradient in the entries of H
function _lscvcriterion(y, w, W, n, H, bins = nothing)
  p = size(H, 1)
  s, M = _gausspairs(y, w, W, [2 .* H, H], [false, true], true, bins)
  D = 1 / sqrt(LinearAlgebra.det(2 .* H))
  f = (2π)^(-p / 2) * D / n + 2 * s[1] / W^2 - 2 * s[2]
  G = -((2π)^(-p / 2) * D / (2n)) .* inv(H) .+ (2 / W^2) .* 2 .* _gaussgradient(2 .* H, s[1], M[1]) .-
      2 .* _gaussgradient(H, s[2], M[2])
  return f, G
end

"""
    SCVMatrix(; diagonal = false, stages = 2, pre = diagonal ? :scale : :sphere,
                binned = nothing, gridsize = nothing)

The smoothed cross-validation bandwidth matrix (Hall, Marron and Park 1992, and
Duong and Hazelton 2005 for matrices): the minimizer of

    SCV(H) = (4π)^(-d/2) |H|^(-1/2) / n
             + ∑ᵢ ∑ⱼ wᵢ wⱼ (φ_{2H+2G} - 2 φ_{H+2G} + φ_{2G})(xᵢ - xⱼ) / (∑ᵢ wᵢ)²,

an estimate of the mean integrated squared error whose bias part smooths the data
with the pilot bandwidth matrix `G = g² I` in the transformed coordinates. The
pilot minimizes the squared asymptotic bias of the selected matrix (Duong 2004,
Theorem 4),

    g = (2(d + 4) c₂ᵀc₂ / ((-(d + 2) c₂ᵀc₁ + √((d + 2)² (c₂ᵀc₁)² + 8(d + 4) c₁ᵀc₁ c₂ᵀc₂)) n))^(1/(d + 6)),

with `c₁ = Dᵀ vec(Θ̂₆ Ĥ) / 2`, `c₂ = (4π)^(-d/2) Dᵀ (2 vec Ĥ + tr(Ĥ) vec I) / 8`, where `D`
is the duplication matrix, `Ĥ` the plug-in matrix of [`PlugInMatrix`](@ref) with the
same `stages`, and `Θ̂₆` the matrix with entries `∑_k ∑_l ψ̂_{eₐ+e_b+2e_k+2e_l}` estimated
with the plug-in pilot of order 6. The criterion is less variable than that of
[`LSCVMatrix`](@ref) and does not need the asymptotic approximation of the bias
that the plug-in rule makes.

`binned = true` approximates all the pair sums, including the internal plug-in
selection and pilots, by linear binning in two or three dimensions. With
`binned = nothing`, it is used above 500 observations with positive weight in
two dimensions and 1000 in three. `gridsize` is the number of nodes per axis, at
least 3, defaulting to 151 in two dimensions and 31 in three. `binned = false`
uses exact sums and ignores `gridsize`. The smeared self-pairs are removed and
the exact diagonal terms kept. The details record `binned`, `gridsize`, `spacing`
and `resolution`, the smallest selected kernel standard deviation and the pilot
`g` in units of the largest grid spacing in the transformed coordinates. A value
below 1 warns to increase `gridsize` or use `binned = false`.
"""
struct SCVMatrix <: MatrixSelector
  diagonal::Bool
  stages::Int
  pre::Symbol
  binned::Union{Nothing,Bool}
  gridsize::Union{Nothing,Int}

  SCVMatrix(diagonal::Bool, stages::Integer, pre::Symbol,
            binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
    new(diagonal, _checkstages(stages), _checkpre(pre, diagonal), binned, _checkgridsize(gridsize))
end

SCVMatrix(; diagonal::Bool = false, stages::Integer = 2, pre::Symbol = diagonal ? :scale : :sphere,
          binned::Union{Nothing,Bool} = nothing, gridsize::Union{Nothing,Integer} = nothing) =
  SCVMatrix(diagonal, stages, pre, binned, gridsize)

function _selectmatrix(rule::SCVMatrix, d::_MvData)
  A, y, Σ = _pretransform(d, rule.pre)
  p, n, W = _dimension(d), d.neff, d.sumw
  bins = _matrixbins(rule, y, d)
  # Θ̂₆ needs the functionals of order 6 estimated, with a pilot from the normal
  # reference of order 8; for one stage, the functionals of order 4 come from the
  # normal reference of order 6
  psi, pilots = if rule.stages == 2
    _psichain(y, d, Σ, 2, false, bins)
  else
    R4, R6 = _multiindices(p, 4), _multiindices(p, 6)
    g6 = _samse(R6, _psinormal(Σ, _multiindices(p, 8)), n)
    g4 = _samse(R4, _psinormal(Σ, R6), n)
    Dict(6 => _psihat(y, d.w, W, n, g6, R6, bins), 4 => _psihat(y, d.w, W, n, g4, R4, bins)), Dict(6 => g6, 4 => g4)
  end
  T = _psitensor(psi[4], p)
  Hpi = _minimizematrix(H -> _picriterion(H, T, (4π)^(-p / 2) / n), _normalscale(Σ, n, rule.diagonal), rule.diagonal)[1]
  g = _scvpilot(_theta6(psi[6], p), Hpi, n)
  G = g^2 .* Matrix{Float64}(LinearAlgebra.I, p, p)
  s0, _ = _gausspairs(y, d.w, W, [2 .* G], [false], false, bins)
  const0 = (2π)^(-p / 2) / sqrt(LinearAlgebra.det(2 .* G)) / n + 2 * s0[1] / W^2
  Hy, crit, iter, converged = _minimizematrix(H -> _scvcriterion(y, d.w, W, n, H, G, const0, bins),
                                              _normalscale(Σ, n, rule.diagonal), rule.diagonal)
  details = (pilot = g, criterion = crit * abs(LinearAlgebra.det(A)), iterations = iter, converged = converged)
  return BandwidthSelection(_backtransform(A, Hy), rule, merge(details, _binningdetails(bins, Hy, g)))
end

# Θ₆ with entries ∑_k ∑_l ψ_{eₐ+e_b+2e_k+2e_l}
function _theta6(psi6, p)
  Θ = zeros(p, p)
  r = zeros(Int, p)
  for a in 1:p, b in 1:p, k in 1:p, l in 1:p
    fill!(r, 0)
    r[a] += 1
    r[b] += 1
    r[k] += 2
    r[l] += 2
    Θ[a, b] += psi6[r]
  end
  return Θ
end

# Dᵀ vec(M) for the duplication matrix D: the entries of vech, the off-diagonal
# ones from both triangles
_dtvec(M) = (p = size(M, 1); [a == b ? M[a, a] : M[a, b] + M[b, a] for b in 1:p for a in b:p])

# the SCV pilot of Duong (2004), Theorem 4
function _scvpilot(Θ, H, n)
  p = size(H, 1)
  c1 = _dtvec(Θ * H) ./ 2
  c2 = (4π)^(-p / 2) .* (2 .* _dtvec(H) .+ LinearAlgebra.tr(H) .* _dtvec(Matrix{Float64}(LinearAlgebra.I, p, p))) ./ 8
  a, b, c = LinearAlgebra.dot(c2, c2), LinearAlgebra.dot(c2, c1), LinearAlgebra.dot(c1, c1)
  return (2 * (p + 4) * a / ((-(p + 2) * b + sqrt((p + 2)^2 * b^2 + 8 * (p + 4) * c * a)) * n))^(1 / (p + 6))
end

# the criterion of SCVMatrix for the pilot G and its gradient in the entries of H;
# const0 is the part that does not depend on H
function _scvcriterion(y, w, W, n, H, G, const0, bins = nothing)
  p = size(H, 1)
  S1, S2 = 2 .* H .+ 2 .* G, H .+ 2 .* G
  s, M = _gausspairs(y, w, W, [S1, S2], [false, false], true, bins)
  D = 1 / sqrt(LinearAlgebra.det(H))
  # the terms i = j: (φ_{2H+2G} - 2 φ_{H+2G})(0) / neff
  d1 = (2π)^(-p / 2) / sqrt(LinearAlgebra.det(S1))
  d2 = (2π)^(-p / 2) / sqrt(LinearAlgebra.det(S2))
  f = (4π)^(-p / 2) * D / n + (d1 - 2 * d2) / n + 2 * (s[1] - 2 * s[2]) / W^2 + const0
  Gr = -((4π)^(-p / 2) * D / (2n)) .* inv(H) .- (d1 / n) .* inv(S1) .+ (d2 / n) .* inv(S2) .+
       (2 / W^2) .* (2 .* _gaussgradient(S1, s[1], M[1]) .- 2 .* _gaussgradient(S2, s[2], M[2]))
  return f, Gr
end
