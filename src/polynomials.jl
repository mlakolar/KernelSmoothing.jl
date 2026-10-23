# Polynomials are coefficient collections in ascending powers,
# c[1] + c[2] t + c[3] t^2 + ...; tuples (named kernels) and vectors (derived
# kernels) both work.

# ∫_0^t s^j p(s) ds, by Horner's rule on the antiderivative
function _antideriv(c, t, j::Integer = 0)
  n = length(c)
  s = c[n] / (n + j)
  for i in n-1:-1:1
    s = muladd(t, s, c[i] / (i + j))
  end
  return t^(j + 1) * s
end

# ∫_a^b t^j p(t) dt
_integrate(c, a, b, j::Integer = 0) = _antideriv(c, b, j) - _antideriv(c, a, j)

# coefficients of the r-th derivative; d^r/dt^r t^(i-1) = (i-1)!/(i-1-r)! t^(i-1-r)
function _derivcoefs(c, r::Integer)
  length(c) <= r && return [0.0]
  return [c[i] * prod(i-r:i-1; init = 1) for i in r+1:length(c)]
end

function _polymul(a, b)
  c = zeros(promote_type(eltype(a), eltype(b)), length(a) + length(b) - 1)
  for i in eachindex(a), j in eachindex(b)
    c[i + j - 1] += a[i] * b[j]
  end
  return c
end

# coefficients of p(t + m) in the number type T; p(u) = q(u - m) for the returned q
_shifted(T, c, m) = _shift!(T[c[i] for i in eachindex(c)], length(c), T(m))

# g[1:len] ← the coefficients of g(t + m), by repeated synthetic division
function _shift!(g, len, m)
  for i in 1:len-1, k in len-1:-1:i
    g[k] += m * g[k + 1]
  end
  return g
end

# g[1:len+1] ← the coefficients of g(t) (ρ t + δ); returns the new length
function _mulaffine!(g, len, ρ, δ)
  g[len + 1] = ρ * g[len]
  for q in len:-1:2
    g[q] = muladd(δ, g[q], ρ * g[q - 1])
  end
  g[1] *= δ
  return len + 1
end

# g[1:len] ← the coefficients of g(ρ t)
function _scale!(g, len, ρ)
  f = one(ρ)
  for q in 1:len
    g[q] *= f
    f *= ρ
  end
  return g
end

# coefficients of (α + β t)^n
_affinepow(α, β, n::Integer) = [binomial(n, t) * α^(n - t) * β^t for t in 0:n]

# acc += scale * c, growing acc as needed
function _addto!(acc::Vector, c, scale = one(eltype(acc)))
  length(acc) < length(c) && append!(acc, zeros(eltype(acc), length(c) - length(acc)))
  for i in eachindex(c)
    acc[i] += scale * c[i]
  end
  return acc
end

# nodes and weights of the m-point Gauss–Legendre rule on [-1, 1] (Golub and Welsch)
function _gausslegendre(m)
  β = [l / sqrt(4l^2 - 1) for l in 1:m-1]
  E = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(zeros(m), β))
  return E.values, 2 .* E.vectors[1, :] .^ 2
end

const _GL24 = _gausslegendre(24)

# ∫ f over the support (lo, hi) of a kernel, by the 24-point Gauss–Legendre rule on
# 16 equal pieces of a bounded support, or on 32 pieces of (-1, 1) after the
# substitution u = t / (1 - t²) for an unbounded one; exact for polynomials of degree
# up to 47 on each piece, and accurate to rounding for smooth integrands that decay
# faster than a power of u
function _quadrature(f, (lo, hi))
  xs, ws = _GL24
  bounded = isfinite(lo) && isfinite(hi)
  # u = t / (1 - t²) increases from -Inf to Inf on (-1, 1), and t = 2u / (1 + √(1 + 4u²))
  tof(u) = isfinite(u) ? 2u / (1 + sqrt(1 + 4u^2)) : sign(u)
  a, b = bounded ? (lo, hi) : (tof(lo), tof(hi))
  m = bounded ? 16 : 32
  s = 0.0
  for c in 1:m
    l, r = a + (c - 1) * (b - a) / m, a + c * (b - a) / m
    mid, half = (l + r) / 2, (r - l) / 2
    for q in eachindex(xs)
      t = mid + half * xs[q]
      s += half * ws[q] * (bounded ? f(t) : f(t / (1 - t^2)) * (1 + t^2) / (1 - t^2)^2)
    end
  end
  return s
end
