"""
    AbstractKernel

Supertype of smoothing kernels. A kernel `k` is a function of a scaled distance:
an estimator with bandwidth `h` gives an observation at distance `d` the weight
`k(d / h) / h`.

The named kernels are symmetric probability densities with a finite, positive
second moment, that is, second-order kernels. The compactly supported ones live
on `[-1, 1]`, so `h` is the half-width of the smoothing window.
[`GaussianKernel`](@ref) is the standard normal density, for which `h` is a
standard deviation. A bandwidth `h_sd` in the standard-deviation convention of
KernelDensity.jl, KDEpy and R's `density` corresponds to the half-width
`h = h_sd / sqrt(moment(k, 2))`.

| Kernel                       | `k(u)` on its support    | `moment(k, 2)` | `roughness(k)`    | `smoothness(k)` |
|:-----------------------------|:-------------------------|:---------------|:------------------|:----------------|
| [`UniformKernel`](@ref)      | `1/2`                    | `1/3`          | `1/2`             | `0`             |
| [`TriangularKernel`](@ref)   | `1 - abs(u)`             | `1/6`          | `2/3`             | `1`             |
| [`EpanechnikovKernel`](@ref) | `3/4 (1 - u^2)`          | `1/5`          | `3/5`             | `1`             |
| [`BiweightKernel`](@ref)     | `15/16 (1 - u^2)^2`      | `1/7`          | `5/7`             | `2`             |
| [`TriweightKernel`](@ref)    | `35/32 (1 - u^2)^3`      | `1/9`          | `350/429`         | `3`             |
| [`GaussianKernel`](@ref)     | `exp(-u^2/2) / sqrt(2π)` | `1`            | `1 / (2 sqrt(π))` | `typemax(Int)`  |

Every kernel can be evaluated, `k(u)`, and supports [`support`](@ref),
[`cdf`](@ref), [`ccdf`](@ref), [`moment`](@ref), [`roughness`](@ref),
[`smoothness`](@ref), [`derivative`](@ref) and [`convolve`](@ref). Derivatives and convolutions are
kernels too, although a derivative integrates to zero rather than one.
"""
abstract type AbstractKernel end

Base.broadcastable(k::AbstractKernel) = Ref(k)

"""
    smoothness(k)

The number of derivatives of the kernel `k` that are functions. For
`r ≤ smoothness(k)`, the derivative of order `r` exists almost everywhere and is
the derivative in the sense of distributions, so [`derivative`](@ref)`(k, r)`
estimates the `r`-th derivative of a density. The next derivative has point
masses where the kernel or one of its derivatives jumps.
"""
function smoothness end

"""
    roughness(k)

The integral `∫ k(u)^2 du`. A kernel density estimate with bandwidth `h` has
variance close to `roughness(k) f(x) / (n h)`. Plug-in bandwidth selectors also
need the roughness of a derivative, `roughness(derivative(k, r))`.
"""
function roughness end

"""
    derivative(k, r = 1)

The derivative of order `r` of the kernel `k`, as a kernel. The estimate
`∑_i derivative(k, r)((x - X_i) / h) / (n h^(r + 1))` is the `r`-th derivative of
the kernel density estimate.

Throws an `ArgumentError` when `r` exceeds [`smoothness`](@ref)`(k)`. The
Epanechnikov kernel, for example, has a first derivative but no second, because
its first derivative jumps at `±1`.
"""
function derivative end

_check_order(r) = r >= 0 || throw(ArgumentError("derivative order must be nonnegative, got $r"))


## Polynomial kernels

# Compactly supported kernels that are polynomials between breakpoints. Subtypes
# implement `pieces(k)`, returning breakpoints `b` and coefficients `c` with
# k(u) = c[j][1] + c[j][2] t + c[j][3] t^2 + ..., t = u - _mid(b, j), on
# [b[j], b[j+1]] and k(u) = 0 outside [b[1], b[end]], and `smoothness(k)`.
# Expanding each piece about its midpoint keeps evaluation accurate where a
# high-degree piece is nearly zero.
abstract type PolynomialKernel <: AbstractKernel end

_mid(b, j) = (b[j] + b[j + 1]) / 2

"""
    UniformKernel()

The uniform kernel `k(u) = 1/2` on `[-1, 1]`. It jumps at `±1`, so it has no
derivatives.
"""
struct UniformKernel <: PolynomialKernel end

"""
    TriangularKernel()

The triangular kernel `k(u) = 1 - abs(u)` on `[-1, 1]`, the optimal kernel for
local polynomial estimation at a boundary point.
"""
struct TriangularKernel <: PolynomialKernel end

"""
    EpanechnikovKernel()

The Epanechnikov kernel `k(u) = 3/4 (1 - u^2)` on `[-1, 1]`, which minimizes the
asymptotic mean integrated squared error among second-order kernels.

# Examples
```julia
julia> k = EpanechnikovKernel();

julia> k(0.5)
0.5625

julia> moment(k, 2), roughness(k)
(0.2, 0.6)
```
"""
struct EpanechnikovKernel <: PolynomialKernel end

"""
    BiweightKernel()

The biweight, or quartic, kernel `k(u) = 15/16 (1 - u^2)^2` on `[-1, 1]`.
"""
struct BiweightKernel <: PolynomialKernel end

"""
    TriweightKernel()

The triweight kernel `k(u) = 35/32 (1 - u^2)^3` on `[-1, 1]`.
"""
struct TriweightKernel <: PolynomialKernel end

pieces(::UniformKernel) = ((-1.0, 1.0), ((0.5,),))
pieces(::TriangularKernel) = ((-1.0, 0.0, 1.0), ((0.5, 1.0), (0.5, -1.0)))   # about ∓1/2
pieces(::EpanechnikovKernel) = ((-1.0, 1.0), ((0.75, 0.0, -0.75),))
pieces(::BiweightKernel) = ((-1.0, 1.0), ((15/16, 0.0, -15/8, 0.0, 15/16),))
pieces(::TriweightKernel) =
  ((-1.0, 1.0), ((35/32, 0.0, -105/32, 0.0, 105/32, 0.0, -35/32),))

smoothness(::UniformKernel) = 0
smoothness(::TriangularKernel) = 1
smoothness(::EpanechnikovKernel) = 1
smoothness(::BiweightKernel) = 2
smoothness(::TriweightKernel) = 3

const NamedPolynomialKernel =
  Union{UniformKernel,TriangularKernel,EpanechnikovKernel,BiweightKernel,TriweightKernel}

# The Epanechnikov, biweight and triweight kernels are evaluated as powers of 1 - u²,
# which is never negative on [-1, 1] and, with a fused multiply-add, rounded once, so
# that it stays accurate near ±1, where the terms of the expanded pieces cancel and
# their sum can turn negative.
(::UniformKernel)(u::Real) = -1 <= u <= 1 ? 0.5 + zero(float(u)) : _outside(u)
(::TriangularKernel)(u::Real) = -1 <= u <= 1 ? 1.0 - abs(u) : _outside(u)
(::EpanechnikovKernel)(u::Real) = -1 <= u <= 1 ? 0.75 * muladd(-u, u, 1.0) : _outside(u)
(::BiweightKernel)(u::Real) = -1 <= u <= 1 ? 15 / 16 * muladd(-u, u, 1.0)^2 : _outside(u)
(::TriweightKernel)(u::Real) = -1 <= u <= 1 ? 35 / 32 * muladd(-u, u, 1.0)^3 : _outside(u)

# the value of a kernel outside its support: zero, and NaN for NaN
_outside(u) = isnan(u) ? float(u) : zero(float(u))

# The expanded pieces of their distribution functions cancel in the lower tail, which
# is evaluated as F(u) = (1 + u)^m P(u) / c instead, with the terms of P of one sign
# for u ≤ 0. Elsewhere F(u) is above 0.07, and the pieces are accurate and cheaper.
function cdf(k::NamedPolynomialKernel, u::Real)
  -1 < u <= -0.5 && return _lowertail(k, u)
  return min(_piececdf(k, u), 1.0)
end

_lowertail(::UniformKernel, u) = (1.0 + u) / 2
_lowertail(::TriangularKernel, u) = (1.0 + u)^2 / 2
_lowertail(::EpanechnikovKernel, u) = (1.0 + u)^2 * (2.0 - u) / 4
_lowertail(::BiweightKernel, u) = (1.0 + u)^3 * evalpoly(u, (8.0, -9.0, 3.0)) / 16
_lowertail(::TriweightKernel, u) = (1.0 + u)^4 * evalpoly(u, (16.0, -29.0, 20.0, -5.0)) / 32

# The cdf on each piece, a polynomial in u - _mid(b, j) whose constant term carries
# the mass to the left of the piece, and the total mass; _cdfcoefs computes the same
# from the pieces, which the tests check.
_cdfpieces(::UniformKernel) = ((-1.0, 1.0), ((0.5, 0.5),), 1.0)
_cdfpieces(::TriangularKernel) =
  ((-1.0, 0.0, 1.0), ((0.125, 0.5, 0.5), (0.875, 0.5, -0.5)), 1.0)
_cdfpieces(::EpanechnikovKernel) = ((-1.0, 1.0), ((0.5, 0.75, 0.0, -0.25),), 1.0)
_cdfpieces(::BiweightKernel) = ((-1.0, 1.0), ((0.5, 15/16, 0.0, -5/8, 0.0, 3/16),), 1.0)
_cdfpieces(::TriweightKernel) =
  ((-1.0, 1.0), ((0.5, 35/32, 0.0, -35/32, 0.0, 21/32, 0.0, -5/32),), 1.0)

# marks coefficients expanded about the midpoints of the pieces
struct Centered end

"""
    PiecewisePolynomialKernel(breaks, coefs)

A compactly supported kernel that is a polynomial between consecutive
breakpoints: on `[breaks[j], breaks[j+1]]` it equals
`coefs[j][1] + coefs[j][2] u + coefs[j][3] u^2 + ...`, and it is zero outside
`[breaks[1], breaks[end]]`. [`derivative`](@ref) and [`convolve`](@ref) return
this type for polynomial kernels. It also defines new kernels, for example the
fourth-order kernel `3/8 (3 - 5u^2)` on `[-1, 1]`:

```julia
julia> k = KernelSmoothing.PiecewisePolynomialKernel([-1, 1], [[9/8, 0, -15/8]]);

julia> moment(k, 0), moment(k, 2)
(1.0, 0.0)
```

The [`smoothness`](@ref) is found by checking which derivatives are continuous
at the breakpoints. The type is not exported, because KernelFunctions.jl exports
a different type of the same name.

The coefficients are converted in 256-bit precision, set with `setprecision`,
which is local to the task from Julia 1.11 on but global on Julia 1.10; there,
building kernels while another task changes the `BigFloat` precision is unsafe.
"""
struct PiecewisePolynomialKernel <: PolynomialKernel
  breaks::Vector{Float64}             # strictly increasing
  coefs::Vector{Vector{Float64}}      # coefs[j]: ascending powers of u - _mid(breaks, j)
  smoothness::Int
  cdfcoefs::Vector{Vector{Float64}}   # the cdf on each piece, see _cdfcoefs
  mass::Float64                       # the integral of the kernel
  lowercdf::Vector{Float64}           # the cdf on the first piece in powers of u - breaks[1]
  upperccdf::Vector{Float64}          # the upper tail on the last piece in powers of breaks[end] - u

  # Coefficients given in extended precision, the exact pieces of a convolution, a
  # product or a kernel given in powers of u, yield the distribution function and its
  # tails before they are rounded; others are taken as they are, in Float64. `tails`
  # passes tails that are already known, as for a reflection.
  function PiecewisePolynomialKernel(::Centered, breaks::AbstractVector{<:Real},
                                     coefs::AbstractVector{<:AbstractVector{<:Real}},
                                     smoothness::Integer; tails = nothing)
    isempty(coefs) && throw(ArgumentError("a kernel needs at least one piece"))
    length(breaks) == length(coefs) + 1 || throw(ArgumentError(
      "need one more breakpoint than pieces, got $(length(breaks)) and $(length(coefs))"))
    all(isfinite, breaks) && all(>(0), diff(breaks)) ||
      throw(ArgumentError("breakpoints must be finite and strictly increasing"))
    any(isempty, coefs) && throw(ArgumentError("every piece needs at least one coefficient"))
    all(cj -> all(isfinite, cj), coefs) || throw(ArgumentError("coefficients must be finite"))
    smoothness >= 0 || throw(ArgumentError("smoothness must be nonnegative, got $smoothness"))
    b, c = Vector{Float64}(breaks), [Vector{Float64}(cj) for cj in coefs]
    cc, mass, lower, upper = if eltype(first(coefs)) <: BigFloat
      setprecision(BigFloat, 256) do
        cb = [BigFloat.(cj) for cj in coefs]
        A, M = _cdfcoefs(b, cb)
        [Float64.(a) for a in A], Float64(M), (tails === nothing ? _cdftails(b, cb) : tails)...
      end
    else
      (_cdfcoefs(b, c)..., (tails === nothing ? _cdftails(b, c) : tails)...)
    end
    new(b, c, smoothness, cc, mass, Vector{Float64}(lower), Vector{Float64}(upper))
  end
end

function PiecewisePolynomialKernel(breaks::AbstractVector{<:Real},
                                   coefs::AbstractVector{<:AbstractVector{<:Real}})
  k = PiecewisePolynomialKernel(Centered(), breaks, coefs, 0)   # validates the arguments
  b = k.breaks
  centered = setprecision(BigFloat, 256) do
    [_shifted(BigFloat, coefs[j], _mid(b, j)) for j in eachindex(k.coefs)]
  end
  return PiecewisePolynomialKernel(Centered(), b, centered, _smoothness(b, [Float64.(c) for c in centered]))
end

pieces(k::PiecewisePolynomialKernel) = (k.breaks, k.coefs)
smoothness(k::PiecewisePolynomialKernel) = k.smoothness
_cdfpieces(k::PiecewisePolynomialKernel) = (k.breaks, k.cdfcoefs, k.mass)

# The cdf on piece j is M_j + A_j(u - m_j) - A_j(b_j - m_j), with A_j the
# antiderivative of the piece and M_j the mass to its left; folding the constants
# into A_j's constant term leaves one polynomial per piece. Returns those and the
# total mass.
function _cdfcoefs(b, c)
  T = float(eltype(first(c)))
  cc = Vector{Vector{T}}(undef, length(c))
  mass = zero(T)
  for j in eachindex(c)
    m = _mid(b, j)
    A = [zero(T); [c[j][i] / i for i in eachindex(c[j])]]
    A[1] = mass - evalpoly(b[j] - m, A)
    mass = evalpoly(b[j + 1] - m, A)
    cc[j] = A
  end
  return cc, mass
end

# The cdf on the first piece in powers of u - b[1] and the upper tail on the last
# piece in powers of b[end] - u, both zero at their end of the support. Near the ends
# the expansions about the midpoints cancel, while these keep their relative
# accuracy. Computed in the precision of c; coefficients at the level of its rounding
# are zeros of the exact polynomials, such as the vanishing derivatives of a
# convolution at the ends, and are set to zero.
function _cdftails(b, c)
  T = float(eltype(first(c)))
  lo = _shifted(T, c[1], b[1] - _mid(b, 1))                  # in powers of u - b[1]
  hi = _shifted(T, c[end], b[end] - _mid(b, length(c)))      # in powers of u - b[end]
  lower = [zero(T); [lo[i] / i for i in eachindex(lo)]]
  upper = [zero(T); [(-1)^(i - 1) * hi[i] / i for i in eachindex(hi)]]
  chop(a) = (m = maximum(abs, a); Float64[abs(t) <= 1e-60 * m ? 0.0 : Float64(t) for t in a])
  return chop(lower), chop(upper)
end

# In the outer halves of the end pieces, the distribution function and its upper tail
# come from the expansions about the ends of the support
function cdf(k::PiecewisePolynomialKernel, u::Real)
  b = k.breaks
  b[1] < u < _mid(b, 1) && return evalpoly(u - b[1], k.lowercdf)
  return _piececdf(k, u)
end

function ccdf(k::PiecewisePolynomialKernel, u::Real)
  b = k.breaks
  z = zero(u * b[1])
  isnan(u) && return z + u
  u <= b[1] && return z + k.mass
  u >= b[end] && return z
  _mid(b, length(b) - 1) < u && return evalpoly(b[end] - u, k.upperccdf)
  return k.mass - _piececdf(k, u)
end

# The order of the first derivative that jumps somewhere on the real line (the
# kernel is zero outside its support), from the one-sided limits at the breakpoints.
function _smoothness(breaks, coefs)
  n = length(breaks)
  for r in 0:maximum(length, coefs)
    d = [_derivcoefs(c, r) for c in coefs]
    # a jump counts when it exceeds the rounding of the derivative's values, so that
    # the test does not depend on the scale of u
    scale = maximum(_bound(d[j], (breaks[j + 1] - breaks[j]) / 2) for j in eachindex(d))
    for j in 1:n
      left = j == 1 ? 0.0 : evalpoly(breaks[j] - _mid(breaks, j - 1), d[j - 1])
      right = j == n ? 0.0 : evalpoly(breaks[j] - _mid(breaks, j), d[j])
      abs(left - right) > sqrt(eps()) * scale && return r
    end
  end
  return maximum(length, coefs)
end

# a bound on |p(t)| for |t| ≤ w
_bound(c, w) = sum(abs(c[i]) * w^(i - 1) for i in eachindex(c))

function Base.show(io::IO, k::PiecewisePolynomialKernel)
  n = length(k.coefs)
  print(io, "PiecewisePolynomialKernel(", n, n == 1 ? " piece" : " pieces", " on [",
        first(k.breaks), ", ", last(k.breaks), "], smoothness ", k.smoothness, ")")
end

# inlined into the sums over windows, where each term evaluates the kernel
@inline function (k::PolynomialKernel)(u::Real)
  b, c = pieces(k)
  isnan(u) && return u * b[1]
  (u < first(b) || u > last(b)) && return zero(u * b[1])
  j = 1
  @inbounds while j < length(c) && u >= b[j + 1]
    j += 1
  end
  return @inbounds _horner(c[j], u - _mid(b, j))
end

# evalpoly(t, c) for coefficients in a tuple or a vector, without bounds checks
@inline _horner(c::Tuple, t) = evalpoly(t, c)
@inline function _horner(c::AbstractVector, t)
  n = length(c)
  @inbounds s = c[n] + zero(t)
  @inbounds for i in n-1:-1:1
    s = muladd(t, s, c[i])
  end
  return s
end

"""
    support(k::AbstractKernel)

The interval `(lo, hi)` outside which the kernel `k` is zero: `(-1.0, 1.0)` for
the compactly supported named kernels and `(-Inf, Inf)` for the Gaussian. See
also [`insupport`](@ref).
"""
support(k::PolynomialKernel) = (b = first(pieces(k)); (first(b), last(b)))

"""
    insupport(k::AbstractKernel, u)

Whether `u` lies in the [`support`](@ref) `(lo, hi)` of the kernel `k`, that is,
`lo ≤ u ≤ hi`.
"""
insupport(k::AbstractKernel, u::Real) = (s = support(k); s[1] <= u <= s[2])

"""
    cdf(k::AbstractKernel, u)

The integral of the kernel `k` from `-Inf` to `u`. For a named kernel this is a
distribution function. For a derivative of order `r ≥ 1` it is the derivative of
order `r - 1`.
"""
cdf(k::PolynomialKernel, u::Real) = _piececdf(k, u)

"""
    ccdf(k::AbstractKernel, u)

The integral of the kernel `k` from `u` to `Inf`, `moment(k, 0) - cdf(k, u)`,
computed without the cancellation of that difference where the upper tail is small.
A new kernel type can define it to keep its upper tail accurate, which the upper
tail of a [`KDE`](@ref) with the kernel then inherits; otherwise the difference is
used.
"""
ccdf(k::AbstractKernel, u::Real) = moment(k, 0) - cdf(k, u)
ccdf(k::NamedPolynomialKernel, u::Real) = cdf(k, -u)   # the named kernels are symmetric

function _piececdf(k::PolynomialKernel, u::Real)
  b, cc, mass = _cdfpieces(k)
  z = zero(u * b[1])
  isnan(u) && return z + u
  u <= first(b) && return z
  u >= last(b) && return z + mass
  j = 1
  while j < length(cc) && u >= b[j + 1]
    j += 1
  end
  return evalpoly(u - _mid(b, j), cc[j])
end

"""
    moment(k::AbstractKernel, j)

The moment `∫ u^j k(u) du` of order `j ≥ 0`. A second-order kernel has
`moment(k, 0) == 1`, `moment(k, 1) == 0` and a positive `moment(k, 2)`, its
variance.
"""
function moment(k::PolynomialKernel, j::Integer)
  j >= 0 || throw(ArgumentError("moment order must be nonnegative, got $j"))
  b, c = pieces(k)
  return sum(eachindex(c)) do p
    m = _mid(b, p)
    # u^j = (t + m)^j with t = u - m
    sum(binomial(j, i) * m^(j - i) * _integrate(c[p], b[p] - m, b[p + 1] - m, i) for i in 0:j)
  end
end

function roughness(k::PolynomialKernel)
  b, c = pieces(k)
  return sum(eachindex(c)) do p
    m = _mid(b, p)
    _integrate(_polymul(c[p], c[p]), b[p] - m, b[p + 1] - m)
  end
end

function derivative(k::PolynomialKernel, r::Integer = 1)
  _check_order(r)
  r == 0 && return k
  s = smoothness(k)
  r <= s || throw(ArgumentError("$k has smoothness $s, so its derivative of order $r " *
    "is not a function: it has point masses where a lower derivative jumps"))
  b, c = pieces(k)
  return PiecewisePolynomialKernel(Centered(), collect(b), [_derivcoefs(cj, r) for cj in c], s - r)
end

"""
    convolve(k1::AbstractKernel, k2::AbstractKernel)

The convolution `u ↦ ∫ k1(u - v) k2(v) dv`, as a kernel. Least-squares
cross-validation needs `convolve(k, k)`, because `∫ f̂(x)^2 dx` is a double sum
of it over pairs of observations. Convolutions of polynomial kernels are
polynomial kernels, and convolutions of Gaussian kernels and their derivatives
are derivatives of scaled Gaussian densities. Mixing the two families throws an
`ArgumentError`. Polynomial convolutions are computed in 256-bit precision, with
the caveat for Julia 1.10 described under `PiecewisePolynomialKernel`.
"""
function convolve(k1::PolynomialKernel, k2::PolynomialKernel)
  b1, c1 = pieces(k1)
  b2, c2 = pieces(k2)
  # The pieces are expanded in powers of u, convolved and expanded about the new
  # midpoints in extended precision, where the cancellation is harmless.
  centered = setprecision(BigFloat, 256) do
    g1 = [_shifted(BigFloat, c1[i], -_mid(b1, i)) for i in eachindex(c1)]
    g2 = [_shifted(BigFloat, c2[j], -_mid(b2, j)) for j in eachindex(c2)]
    parts = Tuple{Float64,Float64,Vector{BigFloat}}[]   # (lo, hi, coefficients in powers of u)
    for i in eachindex(c1), j in eachindex(c2)
      _convolve_pieces!(parts, g1[i], b1[i], b1[i + 1], g2[j], b2[j], b2[j + 1])
    end
    breaks = sort!(unique!(vcat([p[1] for p in parts], [p[2] for p in parts])))
    coefs = [zeros(BigFloat, 1) for _ in 1:length(breaks) - 1]
    # a part covers the pieces between its ends, found by position, since the midpoint
    # of a piece one unit in the last place wide rounds onto one of its ends
    for (lo, hi, c) in parts
      for m in searchsortedfirst(breaks, lo):searchsortedfirst(breaks, hi)-1
        _addto!(coefs[m], c)
      end
    end
    breaks, [_shifted(BigFloat, coefs[m], _mid(breaks, m)) for m in eachindex(coefs)]
  end
  return PiecewisePolynomialKernel(Centered(), centered..., smoothness(k1) + smoothness(k2) + 1)
end

# The convolution of p on [a, b] with q on [c, d] (both zero elsewhere, and both
# in powers of u) is a polynomial between consecutive points of a + c, a + d,
# b + c and b + d. There it is ∫ p(u - v) q(v) dv over max(c, u - b) ≤ v ≤
# min(d, u - a), and both limits are affine in u.
function _convolve_pieces!(parts, p, a, b, q, c, d)
  T = promote_type(eltype(p), eltype(q))
  # A[m, l] is the coefficient of u^(m-1) v^(l-1) in p(u - v) q(v)
  A = zeros(T, length(p), length(p) + length(q) - 1)
  for i in eachindex(p), k in 0:i-1, j in eachindex(q)
    A[i - k, k + j] += p[i] * binomial(i - 1, k) * (-1)^k * q[j]
  end
  ts = sort!([a + c, a + d, b + c, b + d])
  for s in 1:3
    lo, hi = ts[s], ts[s + 1]
    lo < hi || continue
    # the lower limit is u - b from b + c on and the upper one u - a up to a + d;
    # comparing the ends of the part with these points also decides parts one unit in
    # the last place wide, whose midpoints round onto an end
    αL, βL = lo >= b + c ? (-b, 1) : (c, 0)   # lower limit αL + βL u
    αU, βU = hi <= a + d ? (-a, 1) : (d, 0)   # upper limit αU + βU u
    poly = zeros(T, 1)
    for m in axes(A, 1), l in axes(A, 2)
      iszero(A[m, l]) && continue
      # u^(m-1) ∫ v^(l-1) dv = u^(m-1) (U^l - L^l) / l
      upper = _affinepow(T(αU), T(βU), l) .- _affinepow(T(αL), T(βL), l)
      _addto!(poly, vcat(zeros(T, m - 1), upper), A[m, l] / l)
    end
    push!(parts, (lo, hi, poly))
  end
  return parts
end

"""
    k1 * k2

The product `u ↦ k1(u) k2(u)` of two kernels, as a kernel. Variances need
`k * k`, because the variance of a kernel estimate is a sum of `wᵢ² k(uᵢ)²`
terms. Products of polynomial kernels are polynomial kernels on the intersection
of the supports, computed in 256-bit precision; products of Gaussian kernels
without derivatives are scaled Gaussian kernels. Other products throw an
`ArgumentError`.
"""
function Base.:*(k1::PolynomialKernel, k2::PolynomialKernel)
  b1, c1 = pieces(k1)
  b2, c2 = pieces(k2)
  lo, hi = max(first(b1), first(b2)), min(last(b1), last(b2))
  lo < hi || throw(ArgumentError("the supports of $k1 and $k2 do not overlap"))
  breaks = sort!(unique!(filter(t -> lo <= t <= hi, vcat(collect(b1), collect(b2)))))
  coefs = setprecision(BigFloat, 256) do
    map(1:length(breaks) - 1) do j
      mid = _mid(breaks, j)
      # the pieces of the factors that hold the new piece, found by its left end
      i1, i2 = _pieceat(b1, breaks[j]), _pieceat(b2, breaks[j])
      # both factors expanded about the midpoint of the new piece
      g1 = _shifted(BigFloat, c1[i1], mid - _mid(b1, i1))
      g2 = _shifted(BigFloat, c2[i2], mid - _mid(b2, i2))
      _polymul(g1, g2)
    end
  end
  return PiecewisePolynomialKernel(Centered(), breaks, coefs, _smoothness(breaks, [Float64.(c) for c in coefs]))
end

# the piece j with b[j] ≤ u < b[j+1]
function _pieceat(b, u)
  j = 1
  while j < length(b) - 1 && u >= b[j + 1]
    j += 1
  end
  return j
end

Base.:*(k1::AbstractKernel, k2::AbstractKernel) =
  throw(ArgumentError("the product of $k1 and $k2 is not implemented"))


## Gaussian kernels

"""
    GaussianKernel()

The Gaussian kernel `k(u) = exp(-u^2/2) / sqrt(2π)`, the standard normal
density. It has unbounded support, so the bandwidth is its standard deviation,
and derivatives of every order.
"""
struct GaussianKernel <: AbstractKernel end

const INV_SQRT_2PI = 1 / sqrt(2π)

"""
    GaussianDerivativeKernel(σ, r, a = 1)

`a` times the derivative of order `r` of the normal density with mean zero and
standard deviation `σ`, `u ↦ a σ^-(r+1) φ^(r)(u / σ)`. [`derivative`](@ref),
[`convolve`](@ref) and products return this type for Gaussian kernels.
"""
struct GaussianDerivativeKernel <: AbstractKernel
  σ::Float64      # standard deviation
  r::Int          # order of the derivative
  a::Float64      # amplitude
  invσ::Float64   # 1 / σ
  c::Float64      # a (-1)^r / (sqrt(2π) σ^(r+1))

  function GaussianDerivativeKernel(σ::Real, r::Integer, a::Real = 1.0)
    isfinite(σ) && σ > 0 ||
      throw(ArgumentError("standard deviation σ must be positive and finite, got $σ"))
    _check_order(r)
    isfinite(a) || throw(ArgumentError("amplitude a must be finite, got $a"))
    new(σ, r, a, 1 / σ, a * (-1)^r * INV_SQRT_2PI / Float64(σ)^(r + 1))
  end
end

function Base.show(io::IO, k::GaussianDerivativeKernel)
  print(io, typeof(k), "(", k.σ, ", ", k.r)
  k.a == 1 || print(io, ", ", k.a)
  print(io, ")")
end

const GaussianFamily = Union{GaussianKernel, GaussianDerivativeKernel}
_scale_order(::GaussianKernel) = (1.0, 0)
_scale_order(k::GaussianDerivativeKernel) = (k.σ, k.r)
_amplitude(::GaussianKernel) = 1.0
_amplitude(k::GaussianDerivativeKernel) = k.a

(::GaussianKernel)(u::Real) = INV_SQRT_2PI * exp(-float(u)^2 / 2)

# φ^(r)(z) = (-1)^r He_r(z) φ(z), with He_r the probabilists' Hermite polynomials;
# beyond |z| = 38.6 the exponential is zero while the polynomial can overflow
function (k::GaussianDerivativeKernel)(u::Real)
  z = u * k.invσ
  e = exp(-z^2 / 2)
  return iszero(e) ? k.c * e : k.c * _hermite(k.r, z) * e
end

# He_r(z), from He_{n+1} = z He_n - n He_{n-1}
function _hermite(r::Integer, z)
  r == 0 && return one(z)
  h0, h1 = one(z), z
  for n in 1:r-1
    h0, h1 = h1, z * h1 - n * h0
  end
  return h1
end

support(::GaussianFamily) = (-Inf, Inf)
smoothness(::GaussianFamily) = typemax(Int)

cdf(::GaussianKernel, u::Real) = cdf(Distributions.Normal(), u)

function cdf(k::GaussianDerivativeKernel, u::Real)
  k.r == 0 && return k.a * cdf(Distributions.Normal(), u / k.σ)
  return GaussianDerivativeKernel(k.σ, k.r - 1, k.a)(u)
end

ccdf(::GaussianKernel, u::Real) = ccdf(Distributions.Normal(), u)

# a derivative of order r ≥ 1 integrates to zero, so that its upper tail is minus its
# distribution function
function ccdf(k::GaussianDerivativeKernel, u::Real)
  k.r == 0 && return k.a * ccdf(Distributions.Normal(), u / k.σ)
  return -GaussianDerivativeKernel(k.σ, k.r - 1, k.a)(u)
end

# ∫ u^j φ^(r)(u) du = (-1)^r j!/(j-r)! E[Z^(j-r)], and E[Z^m] = (m-1)!! for even m
function moment(k::GaussianFamily, j::Integer)
  j >= 0 || throw(ArgumentError("moment order must be nonnegative, got $j"))
  σ, r = _scale_order(k)
  m = j - r
  (m < 0 || isodd(m)) && return 0.0
  return _amplitude(k) * (-1)^r * prod(m+1:j; init = 1.0) * prod(1:2:m-1; init = 1.0) * σ^m
end

# ∫ (φ^(r))^2 = (2r)! / (2^(2r+1) r! sqrt(π))
function roughness(k::GaussianFamily)
  σ, r = _scale_order(k)
  return _amplitude(k)^2 * prod(r+1:2r; init = 1.0) / (2.0^(2r + 1) * sqrt(π) * σ^(2r + 1))
end

function derivative(k::GaussianFamily, r::Integer = 1)
  _check_order(r)
  r == 0 && return k
  σ, r0 = _scale_order(k)
  return GaussianDerivativeKernel(σ, r0 + r, _amplitude(k))
end

function convolve(k1::GaussianFamily, k2::GaussianFamily)
  σ1, r1 = _scale_order(k1)
  σ2, r2 = _scale_order(k2)
  return GaussianDerivativeKernel(hypot(σ1, σ2), r1 + r2, _amplitude(k1) * _amplitude(k2))
end

# φ_σ₁ φ_σ₂ = φ_σ σ / (sqrt(2π) σ₁ σ₂) with σ^-2 = σ₁^-2 + σ₂^-2
function Base.:*(k1::GaussianFamily, k2::GaussianFamily)
  (σ1, r1), (σ2, r2) = _scale_order(k1), _scale_order(k2)
  r1 == r2 == 0 || throw(ArgumentError(
    "the product of $k1 and $k2 is not implemented; only Gaussian kernels without derivatives multiply"))
  σ = 1 / sqrt(1 / σ1^2 + 1 / σ2^2)
  return GaussianDerivativeKernel(σ, 0, _amplitude(k1) * _amplitude(k2) * σ * INV_SQRT_2PI / (σ1 * σ2))
end

convolve(k1::AbstractKernel, k2::AbstractKernel) =
  throw(ArgumentError("the convolution of $k1 and $k2 is not implemented"))


## Helpers for estimators

# u ↦ k(-u), as a kernel. Kernel sums are taken over u = (xᵢ - x) / h, while a
# density estimate weights observations by k((x - xᵢ) / h).
_reflect(k::Union{NamedPolynomialKernel,GaussianKernel}) = k
_reflect(k::GaussianDerivativeKernel) = isodd(k.r) ? GaussianDerivativeKernel(k.σ, k.r, -k.a) : k

# piece j becomes piece n + 1 - j, and c(-u - m) = ∑ cᵢ (-1)^i (u + m)^i; the upper
# tail becomes the lower one, and a derived kernel passes on its tails, which came
# from coefficients before their rounding
function _reflect(k::PolynomialKernel)
  b, c = pieces(k)
  breaks = [-b[i] for i in length(b):-1:1]
  coefs = [[c[j][i] * (-1)^(i - 1) for i in eachindex(c[j])] for j in length(c):-1:1]
  return PiecewisePolynomialKernel(Centered(), breaks, coefs, smoothness(k); tails = _reflectedtails(k))
end

_reflectedtails(k::PiecewisePolynomialKernel) = (k.upperccdf, k.lowercdf)
_reflectedtails(::PolynomialKernel) = nothing

# other kernels: a lazy reflection
_reflect(k::AbstractKernel) = _Reflected(k)

struct _Reflected{K<:AbstractKernel} <: AbstractKernel
  k::K
end

(r::_Reflected)(u::Real) = r.k(-u)
support(r::_Reflected) = (s = support(r.k); (-s[2], -s[1]))
cdf(r::_Reflected, u::Real) = ccdf(r.k, -u)
ccdf(r::_Reflected, u::Real) = cdf(r.k, -u)

# k(u)², as a kernel: closed forms where they exist, a lazy product otherwise
_square(k::Union{PolynomialKernel,GaussianKernel}) = k * k
_square(k::GaussianDerivativeKernel) = k.r == 0 ? k * k : _Squared(k)
_square(k::AbstractKernel) = _Squared(k)

struct _Squared{K<:AbstractKernel} <: AbstractKernel
  k::K
end

(s::_Squared)(u::Real) = s.k(u)^2
support(s::_Squared) = support(s.k)

# Whether k(u) ≥ 0 for all u. A polynomial piece is checked at its ends and at the
# real zeros of its derivative inside it, with a tolerance for rounding.
_isnonnegative(::Union{NamedPolynomialKernel,GaussianKernel}) = true
_isnonnegative(k::GaussianDerivativeKernel) = k.r == 0 && k.a >= 0

# other kernels: checked on a grid
function _isnonnegative(k::AbstractKernel)
  lo, hi = support(k)
  us = range(isfinite(lo) ? lo : -40.0, isfinite(hi) ? hi : 40.0; length = 4001)
  vals = k.(us)
  return minimum(vals) >= -1e-12 * maximum(abs, vals)
end

function _isnonnegative(k::PolynomialKernel)
  b, c = pieces(k)
  # relative to a bound on the values rather than on the coefficients, which grow
  # with the scale of u
  tol = 1e-12 * maximum(_bound(c[j], (b[j + 1] - b[j]) / 2) for j in eachindex(c))
  for j in eachindex(c)
    w = (b[j + 1] - b[j]) / 2
    for t in _criticalpoints(c[j], w)
      evalpoly(t, c[j]) < -tol && return false
    end
  end
  return true
end

# -w, w and the real zeros of the derivative of the polynomial c in (-w, w). The
# zeros are found in s = t / w, where the sizes of the coefficients measure their
# terms on the piece, so that the leading ones are dropped by what they contribute
# there and not by a size that depends on the scale of u.
function _criticalpoints(c, w)
  ts = [-w, w]
  d = _derivcoefs([c[i] * w^(i - 1) for i in eachindex(c)], 1)
  while length(d) > 1 && abs(d[end]) <= 1e-14 * maximum(abs, d)
    pop!(d)
  end
  n = length(d) - 1   # degree of the derivative
  n >= 1 || return ts
  C = zeros(n, n)     # companion matrix of the monic derivative
  for i in 2:n
    C[i, i - 1] = 1
  end
  C[:, n] .= -d[1:n] ./ d[n + 1]
  for z in LinearAlgebra.eigvals(C)
    s = real(z)
    abs(imag(z)) <= 1e-7 * (1 + abs(s)) && -1 < s < 1 && push!(ts, s * w)
  end
  return ts
end

# u ↦ cdf(k, -u), restricted to the support of k reflected: the observations with
# (xᵢ - x) / h below it count fully towards the distribution function at x, and
# those above it not at all.
_cdfkernel(k::PolynomialKernel) = _cdfpolynomials(k)
_cdfkernel(k::NamedPolynomialKernel) = _NamedCDF(k, -1.0, _cdfpolynomials(k))
_cdfkernel(k::AbstractKernel) = _ReflectedCDF(k)

function _cdfpolynomials(k::PolynomialKernel)
  b, cc, _ = _cdfpieces(k)
  breaks = [-b[i] for i in length(b):-1:1]
  coefs = [[cc[j][i] * (-1)^(i - 1) for i in eachindex(cc[j])] for j in length(cc):-1:1]
  return PiecewisePolynomialKernel(Centered(), breaks, coefs, 0)
end

# u ↦ 1 - cdf(k, -u) = cdf(k reflected, u) on the support of k reflected, the
# mirror image for the upper tail: observations with (xᵢ - x) / h above the support
# count fully towards 1 - F̂(x), and those below it not at all
_ccdfkernel(k::AbstractKernel) = _reflect(_cdfkernel(_reflect(k)))

# u ↦ cdf(k, s u) on [-1, 1] for a named kernel and s = ±1, from its factored
# distribution function, with the pieces of the same polynomial for FastSums
struct _NamedCDF{K<:NamedPolynomialKernel} <: PolynomialKernel
  k::K
  s::Float64
  pk::PiecewisePolynomialKernel
end

pieces(g::_NamedCDF) = pieces(g.pk)
smoothness(::_NamedCDF) = 0
_cdfpieces(g::_NamedCDF) = _cdfpieces(g.pk)
(g::_NamedCDF)(u::Real) = -1 <= u <= 1 ? cdf(g.k, g.s * u) : _outside(u)
_reflect(g::_NamedCDF) = _NamedCDF(g.k, -g.s, _reflect(g.pk))

struct _ReflectedCDF{K<:AbstractKernel} <: AbstractKernel
  k::K
end

support(g::_ReflectedCDF) = (lo = support(g.k); (-lo[2], -lo[1]))

function (g::_ReflectedCDF)(u::Real)
  lo, hi = support(g)
  return lo <= u <= hi ? cdf(g.k, -u) : zero(float(u))
end

# The p-quantile of a kernel that is a probability density, for 0 < p < 1.
_kernelquantile(::GaussianKernel, p) = Distributions.quantile(Distributions.Normal(), p)
_kernelquantile(k::GaussianDerivativeKernel, p) = k.σ * Distributions.quantile(Distributions.Normal(), p)

function _kernelquantile(k::AbstractKernel, p)
  lo, hi = support(k)
  a = isfinite(lo) ? lo : -1.0
  b = isfinite(hi) ? hi : 1.0
  # an unbounded support is bracketed by doubling, a bounded one by its ends
  while !isfinite(lo) && isfinite(a) && cdf(k, a) > p
    a = 2a - 1
  end
  while !isfinite(hi) && isfinite(b) && cdf(k, b) < p
    b = 2b + 1
  end
  # a kernel whose mass is just below one has no quantile above it
  cdf(k, b) < p && return b
  return _invertcdf(u -> cdf(k, u), k, p, a, b)
end

# The smallest x in [a, b] with F(x) ≥ p, to a few units in the last place, for a
# nondecreasing F with derivative f, F(a) ≤ p ≤ F(b): Newton steps where they stay
# inside the bracket and at least halve the step before the last one, bisection
# otherwise, as in rtsafe of Numerical Recipes. In a convex tail, Newton steps stay
# inside the bracket but shrink slowly. Once Newton has converged, the bracket is
# closed around its limit. Where F is flat at the level p, bisection moves to the
# left end of the flat part.
function _invertcdf(F, f, p, a, b)
  x = (a + b) / 2
  step, before = b - a, b - a   # the last step and the one before it
  for _ in 1:2000
    Fx = F(x)
    Fx < p ? (a = x) : (b = x)
    b - a <= 4 * eps(max(abs(a), abs(b))) && break
    fx = f(x)
    xn = fx > 0 ? x - (Fx - p) / fx : (a + b) / 2
    if !(a < xn < b) || 2 * abs(xn - x) > abs(before)
      xn = (a + b) / 2
    elseif abs(xn - x) <= 4 * eps(abs(x))
      δ = 8 * eps(abs(xn)) + floatmin()
      lo, hi = max(a, xn - δ), min(b, xn + δ)
      F(lo) < p && (a = lo)
      F(hi) >= p && (b = hi)
      xn = (a + b) / 2
    end
    before, step = step, xn - x
    x = xn
  end
  return b
end

# A draw from a kernel that is a probability density. (1 + U) / 2 has the
# Beta(j + 1, j + 1) distribution when U has density proportional to (1 - u²)^j.
_randkernel(rng, ::UniformKernel) = 2 * rand(rng) - 1
_randkernel(rng, ::TriangularKernel) = rand(rng) - rand(rng)
_randkernel(rng, ::EpanechnikovKernel) = 2 * rand(rng, Distributions.Beta(2, 2)) - 1
_randkernel(rng, ::BiweightKernel) = 2 * rand(rng, Distributions.Beta(3, 3)) - 1
_randkernel(rng, ::TriweightKernel) = 2 * rand(rng, Distributions.Beta(4, 4)) - 1
_randkernel(rng, ::GaussianKernel) = randn(rng)
_randkernel(rng, k::GaussianDerivativeKernel) = k.σ * randn(rng)
_randkernel(rng, k::AbstractKernel) = _kernelquantile(k, rand(rng))
