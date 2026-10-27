## Binned pair sums for bandwidth matrices

function _checkgridsize(gridsize)
  gridsize === nothing && return nothing
  gridsize >= 3 || throw(ArgumentError("gridsize must be at least 3, got $gridsize"))
  return Int(gridsize)
end

function _usebinning(binned, p, n)
  binned === true && !(p in (2, 3)) &&
    throw(ArgumentError("binned bandwidth matrix selection needs 2 or 3 coordinates, got $p"))
  return binned === nothing ? (p == 2 && n > 500 || p == 3 && n > 1000) : binned
end

# One grid on the transformed observations, shared by all the pilots and criteria.
# The products are unordered sums, with each observation's smeared self-pairs
# removed. Only one of each pair of opposite lags is needed for the even kernels.
struct _MatrixBins{D,L<:Union{Nothing,Vector{Float64}}}
  gridsize::Int
  spacing::NTuple{D,Float64}
  lags::Matrix{Float64}
  ww::Vector{Float64}
  loo::L
end

function _matrixbins(y, d::_MvData, binned, gridsize; loo = false)
  p = size(y, 1)
  _usebinning(binned, p, nobs(d)) || return nothing
  M = gridsize === nothing ? (p == 2 ? 151 : 31) : gridsize
  return _MatrixBins(y, d.w, d.sumw, M; loo)
end

# Serial multilinear binning and the self-correlations on {-1, 0, 1}^d. For LSCV
# the second set of counts has weights aᵢ = wᵢ / (W - wᵢ).
_linearbinning(y, w, W, M; loo = false) = _linearbinning(y, w, W, M, loo, Val(size(y, 1)))

function _linearbinning(y, w, W, M, loo, ::Val{D}) where {D}
  origin = ntuple(a -> minimum(view(y, a, :)), D)
  spacing = ntuple(a -> (maximum(view(y, a, :)) - origin[a]) / (M - 1), D)
  all(>(0), spacing) || throw(ArgumentError("the binning grid needs positive spread in every coordinate"))
  counts = zeros(ntuple(_ -> M, D))
  loo_counts = loo ? zeros(size(counts)) : nothing
  self_ww = zeros(ntuple(_ -> 3, D))
  self_ab = loo ? zeros(size(self_ww)) : nothing
  corners = CartesianIndices(ntuple(_ -> 2, D))
  @inbounds for i in axes(y, 2)
    wi = _weight(w, i)
    wi == 0 && continue
    ai = loo ? wi / (W - wi) : 0.0
    t = ntuple(a -> clamp((y[a, i] - origin[a]) / spacing[a], 0.0, M - 1.0), D)
    pos = ntuple(a -> min(floor(Int, t[a]), M - 2), D)
    frac = ntuple(a -> t[a] - pos[a], D)
    for I in corners
      b = prod(a -> I[a] == 1 ? 1 - frac[a] : frac[a], 1:D)
      J = CartesianIndex(ntuple(a -> pos[a] + I[a], D))
      counts[J] += wi * b
      loo && (loo_counts[J] += ai * b)
    end
    for I in CartesianIndices(self_ww)
      b = prod(a -> I[a] == 2 ? (1 - frac[a])^2 + frac[a]^2 : frac[a] * (1 - frac[a]), 1:D)
      self_ww[I] += wi^2 * b
      loo && (self_ab[I] += ai * wi * b)
    end
  end
  return (; counts, loo_counts, self_ww, self_ab, origin, spacing)
end

# The package's radix-2 FFT along each axis, using one serial workspace per axis.
function _fft_nd!(z::Array{ComplexF64,D}) where {D}
  stride = 1
  for a in 1:D
    N = size(z, a)
    line = Vector{ComplexF64}(undef, N)
    block = stride * N
    @inbounds for start in 0:block:length(z)-1, offset in 1:stride
      for j in 1:N
        line[j] = z[start + offset + (j - 1) * stride]
      end
      _fft!(line)
      for j in 1:N
        z[start + offset + (j - 1) * stride] = line[j]
      end
    end
    stride = block
  end
  return z
end

# B(ℓ) = ∑ₘ cₘ eₘ₊ℓ at every signed lag, indexed by ℓₐ + size(c, a).
# Padding to at least 2M - 1 prevents circular correlations.
function _lagcorrelations(c::Array{Float64,D}, e::Array{Float64,D} = c) where {D}
  size(c) == size(e) || throw(DimensionMismatch("the bin arrays must have the same size"))
  M = size(c)
  N = ntuple(a -> nextpow(2, 2M[a] - 1), D)
  z = zeros(ComplexF64, N)
  z[CartesianIndices(c)] .= c
  _fft_nd!(z)
  if e === c
    z .= abs2.(z)
  else
    v = zeros(ComplexF64, N)
    v[CartesianIndices(e)] .= e
    _fft_nd!(v)
    z .= conj.(z) .* v
  end
  z .= conj.(z)
  _fft_nd!(z)
  B = Array{Float64}(undef, ntuple(a -> 2M[a] - 1, D))
  @inbounds for I in CartesianIndices(B)
    J = CartesianIndex(ntuple(a -> mod(I[a] - M[a], N[a]) + 1, D))
    B[I] = real(z[J]) / length(z)
  end
  return B
end

function _MatrixBins(y, w, W, M; loo = false)
  b = _linearbinning(y, w, W, M; loo)
  return _binproducts(b, M, W, Val(size(y, 1)))
end

function _binproducts(b, M, W, ::Val{D}) where {D}
  B = _lagcorrelations(b.counts)
  C = b.loo_counts === nothing ? nothing : _lagcorrelations(b.loo_counts, b.counts)
  middle = (length(B) + 1) ÷ 2
  lags = Matrix{Float64}(undef, D, middle)
  ww = zeros(middle)
  loo = C === nothing ? nothing : zeros(middle)
  indices = CartesianIndices(B)
  @inbounds for k in middle:length(B)
    lag = Tuple(indices[k]) .- M
    j = k - middle + 1
    for a in 1:D
      lags[a, j] = lag[a] * b.spacing[a]
    end
    near = all(a -> abs(lag[a]) <= 1, 1:D)
    I = CartesianIndex(ntuple(a -> near ? lag[a] + 2 : 2, D))
    s = near ? b.self_ww[I] : 0.0
    # Average opposite lags before removing the self-pairs; tiny negative products
    # from FFT rounding are zero. The zero lag counts half an unordered sum.
    ww[j] = max(0.0, (B[k] + B[end + 1 - k]) / 2 - s) / (k == middle ? 2 : 1)
    if loo !== nothing
      s = near ? b.self_ab[I] : 0.0
      loo[j] = max(0.0, C[k] + C[end + 1 - k] - 2s) / (W * (k == middle ? 2 : 1))
    end
  end
  return _MatrixBins(M, b.spacing, lags, ww, loo)
end

# Tasks depend on the grid only; fetching and adding them in order keeps the
# result independent of the number of threads.
_bintasks(b::_MatrixBins{D}) where {D} = _pairtasks((2b.gridsize - 1)^D)

function _psibins(b::_MatrixBins, g, R, j)
  ntasks = _bintasks(b)
  parts = _fetchall([Threads.@spawn _psibinstask(b, g, R, j, t, ntasks) for t in 1:ntasks], Vector{Float64})
  return reduce(+, parts)
end

function _psibinstask(b::_MatrixBins{D}, g, R, j, t, ntasks) where {D}
  s = zeros(size(R, 2))
  h = zeros(j + 1, D)
  @inbounds for i in t:ntasks:length(b.ww)
    b.ww[i] == 0 && continue
    r2 = 0.0
    for a in 1:D
      u = b.lags[a, i] / g
      r2 += u^2
      _hermite!(view(h, :, a), u, j)
    end
    e = b.ww[i] * exp(-r2 / 2)
    e == 0 && continue
    for k in axes(R, 2)
      v = e
      for a in 1:D
        v *= h[R[a, k] + 1, a]
      end
      s[k] += v
    end
  end
  return s
end

function _gaussbins(b::_MatrixBins{D}, Ss, loo, grad) where {D}
  any(loo) && b.loo === nothing && throw(ArgumentError("the binned cache needs leave-one-out products"))
  Rinv = [inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(S)).L) for S in Ss]
  c = [(2π)^(-D / 2) / sqrt(LinearAlgebra.det(S)) for S in Ss]
  ntasks = _bintasks(b)
  parts = _fetchall([Threads.@spawn _gaussbinstask(b, Rinv, loo, grad, t, ntasks) for t in 1:ntasks],
                    Tuple{Vector{Float64},Vector{Matrix{Float64}}})
  s = zeros(length(Ss))
  M = [zeros(D, D) for _ in Ss]
  for (sk, Mk) in parts
    s .+= sk
    grad && (M .+= Mk)
  end
  return c .* s, [c[k] .* (M[k] .+ M[k]' .- LinearAlgebra.Diagonal(M[k])) for k in eachindex(Ss)]
end

function _gaussbinstask(b::_MatrixBins{D}, Rinv, loo, grad, t, ntasks) where {D}
  s = zeros(length(Rinv))
  M = [zeros(D, D) for _ in Rinv]
  @inbounds for i in t:ntasks:length(b.ww)
    for k in eachindex(Rinv)
      cij = loo[k] ? b.loo[i] : b.ww[i]
      cij == 0 && continue
      R = Rinv[k]
      r2 = 0.0
      for a in 1:D
        u = 0.0
        for c in 1:a
          u += R[a, c] * b.lags[c, i]
        end
        r2 += u^2
      end
      e = cij * exp(-r2 / 2)
      s[k] += e
      if grad
        for c in 1:D, a in c:D
          M[k][a, c] += e * b.lags[a, i] * b.lags[c, i]
        end
      end
    end
  end
  return s, M
end

_binningdetails(::Nothing, H, pilot = nothing) =
  (binned = false, gridsize = nothing, spacing = nothing, resolution = nothing)

function _binningdetails(b::_MatrixBins, H, pilot = nothing)
  δ = maximum(b.spacing)
  resolution = (selected = sqrt(LinearAlgebra.eigmin(LinearAlgebra.Symmetric(H))) / δ,
                pilot = pilot === nothing ? nothing : pilot / δ)
  if resolution.selected < 1 || (resolution.pilot !== nothing && resolution.pilot < 1)
    @warn "the binning grid is coarse relative to the bandwidth or pilot; increase gridsize or use binned = false" resolution
  end
  return (binned = true, gridsize = b.gridsize, spacing = b.spacing, resolution = resolution)
end
