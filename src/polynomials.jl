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

# g[1:len+1] ← the coefficients of g(t) (t + δ); returns the new length
function _mulaffine!(g, len, δ)
  g[len + 1] = g[len]
  for q in len:-1:2
    g[q] = muladd(δ, g[q], g[q - 1])
  end
  g[1] *= δ
  return len + 1
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
