module KernelsTest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, PolynomialKernel, GaussianDerivativeKernel,
                       pieces, _cdfpieces, _cdfcoefs
import Distributions
using ForwardDiff
using QuadGK

const COMPACT = [UniformKernel(), TriangularKernel(), EpanechnikovKernel(),
                 BiweightKernel(), TriweightKernel()]
const KERNELS = [COMPACT; GaussianKernel()]

formula(::UniformKernel, u) = abs(u) <= 1 ? 1/2 : 0.0
formula(::TriangularKernel, u) = abs(u) <= 1 ? 1 - abs(u) : 0.0
formula(::EpanechnikovKernel, u) = abs(u) <= 1 ? 3/4 * (1 - u^2) : 0.0
formula(::BiweightKernel, u) = abs(u) <= 1 ? 15/16 * (1 - u^2)^2 : 0.0
formula(::TriweightKernel, u) = abs(u) <= 1 ? 35/32 * (1 - u^2)^3 : 0.0
formula(::GaussianKernel, u) = exp(-u^2 / 2) / sqrt(2π)

# second moment, roughness and smoothness
const TABLE = Dict(
  UniformKernel() => (1/3, 1/2, 0),
  TriangularKernel() => (1/6, 2/3, 1),
  EpanechnikovKernel() => (1/5, 3/5, 1),
  BiweightKernel() => (1/7, 5/7, 2),
  TriweightKernel() => (1/9, 350/429, 3),
  GaussianKernel() => (1.0, 1 / (2 * sqrt(π)), typemax(Int)),
)

# points away from the breakpoints 0 and ±1
const POINTS = (-1.3, -0.83, -0.41, 0.17, 0.62, 0.95, 2.4)

breakpoints(k::PolynomialKernel) = collect(first(pieces(k)))
breakpoints(k) = Float64[]

# ∫ f over [lo, hi] ∩ support(k), split at the kernel's breakpoints and at `extra`
function integral(f, k, lo = -Inf, hi = Inf; extra = Float64[])
  a, b = support(k)
  lo, hi = max(lo, a), min(hi, b)
  lo < hi || return 0.0
  pts = sort!(unique!(filter(t -> lo < t < hi, [breakpoints(k); extra])))
  return quadgk(f, lo, pts..., hi; rtol = 1e-12, atol = 1e-18)[1]
end

# (k1 * k2)(u) = ∫ k1(u - v) k2(v) dv by quadrature
function numconv(k1, k2, u)
  a, b = support(k1)
  return integral(v -> k1(u - v) * k2(v), k2, u - b, u - a; extra = u .- breakpoints(k1))
end

@testset "named kernels" begin
  for k in KERNELS
    μ2, R, s = TABLE[k]
    @test support(k) == (k isa GaussianKernel ? (-Inf, Inf) : (-1.0, 1.0))
    for u in (-1.5, -1.0, -0.7, -0.2, 0.0, 0.3, 0.999, 1.0, 1.2)
      @test k(u) ≈ formula(k, u) atol = 1e-15
    end
    @test @inferred(k(0.3)) isa Float64
    @test k(0) == k(0.0)
    @test moment(k, 0) ≈ 1
    @test abs(moment(k, 1)) < 1e-15
    @test moment(k, 2) ≈ μ2
    @test abs(moment(k, 3)) < 1e-15
    @test moment(k, 4) ≈ integral(u -> u^4 * k(u), k)
    @test_throws ArgumentError moment(k, -1)
    @test roughness(k) ≈ R
    @test roughness(k) ≈ integral(u -> k(u)^2, k)
    @test smoothness(k) == s
  end
end

@testset "cdf" begin
  for k in KERNELS
    lo, hi = support(k)
    @test cdf(k, lo) == 0
    @test cdf(k, 0.0) ≈ 1/2
    @test cdf(k, hi) ≈ 1
    @test isnan(cdf(k, NaN))
    @test @inferred(cdf(k, 0.3)) isa Float64
    for u in POINTS
      @test cdf(k, u) ≈ integral(k, k, -Inf, u) atol = 1e-13
      @test ForwardDiff.derivative(t -> cdf(k, t), u) ≈ k(u) atol = 1e-13
    end
  end
end

@testset "derivatives" begin
  φ(u) = exp(-u^2 / 2) / sqrt(2π)
  g = GaussianKernel()
  for u in (-2.1, -0.4, 0.0, 0.9)
    @test derivative(g, 1)(u) ≈ -u * φ(u) atol = 1e-15
    @test derivative(g, 2)(u) ≈ (u^2 - 1) * φ(u) atol = 1e-15
    @test derivative(g, 4)(u) ≈ (u^4 - 6u^2 + 3) * φ(u) atol = 1e-15
  end
  for k in KERNELS
    @test derivative(k, 0) === k
    @test_throws ArgumentError derivative(k, -1)
    for r in 1:min(smoothness(k), 4)
      d, d0 = derivative(k, r), derivative(k, r - 1)
      @test smoothness(d) == (k isa GaussianKernel ? typemax(Int) : smoothness(k) - r)
      @test derivative(d0, 1)(0.37) ≈ d(0.37)
      for u in POINTS
        @test d(u) ≈ ForwardDiff.derivative(d0, u) atol = 1e-12
        @test cdf(d, u) ≈ d0(u) atol = 1e-12
      end
      @test abs(moment(d, 0)) < 1e-12
      # integration by parts: ∫ u^j k^(r)(u) du = (-1)^r j!/(j-r)! ∫ u^(j-r) k(u) du
      for j in r:r+2
        @test moment(d, j) ≈ (-1)^r * prod(j-r+1:j) * moment(k, j - r) atol = 1e-12
        @test moment(d, j) ≈ integral(u -> u^j * d(u), d) atol = 1e-11
      end
      @test roughness(d) ≈ integral(u -> d(u)^2, d)
    end
  end
  for k in COMPACT
    @test_throws ArgumentError derivative(k, smoothness(k) + 1)
  end
end

@testset "convolutions" begin
  epa2(u) = 3/160 * (2 - abs(u))^3 * (u^2 + 6abs(u) + 4)
  gauss2(u) = exp(-u^2 / 4) / (2 * sqrt(π))
  @test convolve(UniformKernel(), UniformKernel())(0.7) ≈ (2 - 0.7) / 4
  for u in (-1.7, -0.6, 0.0, 0.4, 1.9)
    @test convolve(EpanechnikovKernel(), EpanechnikovKernel())(u) ≈ epa2(u) atol = 1e-14
    @test convolve(GaussianKernel(), GaussianKernel())(u) ≈ gauss2(u) atol = 1e-15
  end
  for k in KERNELS
    c = convolve(k, k)
    lo, hi = support(k)
    @test support(c) == (2lo, 2hi)
    @test moment(c, 0) ≈ 1
    @test moment(c, 2) ≈ 2 * moment(k, 2)   # variances add
    @test c(0.0) ≈ roughness(k)             # ∫ k(-v) k(v) dv = ∫ k(v)^2 dv
    @test smoothness(c) == (k isa GaussianKernel ? typemax(Int) : 2 * smoothness(k) + 1)
    # relative accuracy also in the tails, where a high-degree piece is nearly zero
    for u in (-2.5, -1.97, -1.7, -0.6, 0.4, 1.3, 1.9, 1.99)
      @test c(u) ≈ numconv(k, k, u) atol = 1e-15 rtol = 1e-9
    end
  end
  # different kernels, including derivatives
  for (k1, k2) in [(EpanechnikovKernel(), UniformKernel()),
                   (TriangularKernel(), BiweightKernel()),
                   (derivative(BiweightKernel(), 1), TriweightKernel()),
                   (derivative(GaussianKernel(), 1), derivative(GaussianKernel(), 2))]
    c = convolve(k1, k2)
    @test moment(c, 0) ≈ moment(k1, 0) * moment(k2, 0) atol = 1e-12
    for u in (-2.5, -1.95, -1.2, -0.3, 0.5, 1.6, 1.95)
      @test c(u) ≈ numconv(k1, k2, u) atol = 1e-15 rtol = 1e-9
    end
  end
  @test moment(convolve(EpanechnikovKernel(), UniformKernel()), 2) ≈ 1/5 + 1/3
  b = BiweightKernel()
  @test convolve(derivative(b, 1), b)(0.8) ≈ derivative(convolve(b, b), 1)(0.8)
  @test_throws ArgumentError convolve(EpanechnikovKernel(), GaussianKernel())
end

@testset "PiecewisePolynomialKernel" begin
  b, c = pieces(EpanechnikovKernel())
  k = PiecewisePolynomialKernel(collect(b), [collect(c[1])])
  @test smoothness(k) == 1
  @test k(0.3) ≈ EpanechnikovKernel()(0.3)
  tri = PiecewisePolynomialKernel([-1, 0, 1], [[1, 1], [1, -1]])   # coefficients in powers of u
  @test smoothness(tri) == 1
  @test all(tri(u) ≈ TriangularKernel()(u) for u in (-0.8, -0.3, 0.0, 0.45, 0.9))
  @test smoothness(PiecewisePolynomialKernel([-1, 1], [[35/32, 0, -105/32, 0, 105/32, 0, -35/32]])) == 3
  k4 = PiecewisePolynomialKernel([-1, 1], [[9/8, 0, -15/8]])   # a fourth-order kernel
  @test smoothness(k4) == 0
  @test moment(k4, 0) ≈ 1
  @test abs(moment(k4, 2)) < 1e-15
  @test moment(k4, 4) ≈ -3/35
  @test_throws ArgumentError derivative(k4, 1)
  @test_throws ArgumentError PiecewisePolynomialKernel([1, -1], [[1.0]])
  @test_throws ArgumentError PiecewisePolynomialKernel([-1, 0, 1], [[1.0]])
  @test_throws ArgumentError PiecewisePolynomialKernel([-1, 1], [Float64[]])
  @test_throws ArgumentError PiecewisePolynomialKernel([-1, 1], [[NaN]])
  @test_throws ArgumentError PiecewisePolynomialKernel([-1, 1], [[1.0, Inf]])
  # a constant piece still propagates NaN
  @test isnan(PiecewisePolynomialKernel([-1, 1], [[0.5]])(NaN))
  # the smoothness test does not depend on the scale of u: the biweight on [-10³, 10³]
  @test smoothness(PiecewisePolynomialKernel([-1e3, 1e3], [[15 / 16e3, 0, -15 / 8e9, 0, 15 / 16e15]])) == 2
  @test smoothness(PiecewisePolynomialKernel([-1e-3, 1e-3], [[15 / 16e-3, 0, -15 / 8e-9, 0, 15 / 16e-15]])) == 2
  @test sprint(show, convolve(UniformKernel(), UniformKernel())) ==
        "PiecewisePolynomialKernel(2 pieces on [-2.0, 2.0], smoothness 1)"
end

@testset "cdf tables" begin
  # the written-out cdf pieces of the named kernels are what _cdfcoefs computes
  for k in COMPACT
    b, cc, mass = _cdfpieces(k)
    gcc, gmass = _cdfcoefs(pieces(k)...)
    @test all(collect(cc[j]) == gcc[j] for j in eachindex(gcc))
    @test mass == gmass == 1
    @test cdf(k, last(b)) == 1 && cdf(k, 3.0) == 1
  end
  d = derivative(BiweightKernel(), 1)
  @test abs(_cdfpieces(d)[3]) < 1e-15   # a derivative integrates to zero
end

@testset "Gaussian amplitudes" begin
  g = GaussianKernel()
  @test endswith(sprint(show, derivative(g, 2)), "GaussianDerivativeKernel(1.0, 2)")
  @test endswith(sprint(show, g * g), "GaussianDerivativeKernel(0.7071067811865475, 0, 0.28209479177387814)")
  s = GaussianDerivativeKernel(1.5, 2, 3.0)   # 3 φ''_1.5
  base = GaussianDerivativeKernel(1.5, 2)
  for u in (-2.0, 0.3, 1.7)
    @test s(u) ≈ 3 * base(u)
    @test cdf(s, u) ≈ 3 * cdf(base, u)
  end
  @test moment(s, 4) ≈ 3 * moment(base, 4)
  @test roughness(s) ≈ 9 * roughness(base)
  @test convolve(s, g)(0.4) ≈ 3 * convolve(base, g)(0.4)
  @test derivative(s, 1)(0.4) ≈ 3 * derivative(base, 1)(0.4)
  @test_throws ArgumentError GaussianDerivativeKernel(1.0, 0, Inf)
  @test_throws ArgumentError GaussianDerivativeKernel(Inf, 0)
end

@testset "products" begin
  for k in KERNELS
    kk = k * k
    @test moment(kk, 0) ≈ roughness(k)   # ∫ k(u)^2 du
    for u in POINTS
      @test kk(u) ≈ k(u)^2 atol = 1e-15
    end
  end
  @test smoothness(EpanechnikovKernel() * EpanechnikovKernel()) == 2   # (1 - u^2)^2, like the biweight
  for (k1, k2) in [(TriangularKernel(), BiweightKernel()),
                   (convolve(EpanechnikovKernel(), EpanechnikovKernel()), TriangularKernel()),
                   (derivative(TriweightKernel(), 1), UniformKernel()),
                   (PiecewisePolynomialKernel([-1.0, 0.25, 2.0], [[0.5, 0.2], [0.1, -0.05, 0.3]]),
                    convolve(UniformKernel(), UniformKernel()))]
    p = k1 * k2
    @test support(p) == (max(support(k1)[1], support(k2)[1]), min(support(k1)[2], support(k2)[2]))
    for u in range(-2.2, 2.2; length = 23)
      @test p(u) ≈ k1(u) * k2(u) atol = 1e-14
    end
    @test moment(p, 1) ≈ integral(u -> u * k1(u) * k2(u), p) atol = 1e-12
  end
  g = GaussianKernel()
  for u in (-1.2, 0.0, 0.7)
    @test (g * g)(u) ≈ g(u)^2
    @test (g * convolve(g, g))(u) ≈ g(u) * convolve(g, g)(u)
  end
  @test_throws ArgumentError derivative(g, 1) * g
  @test_throws ArgumentError EpanechnikovKernel() * g
  @test_throws ArgumentError UniformKernel() * PiecewisePolynomialKernel([2.0, 3.0], [[1.0]])
end

@testset "accuracy near the ends of the support" begin
  # the named kernels and their distribution functions in factored form keep their
  # relative accuracy near ±1, where the terms of the expanded pieces cancel; the
  # reference evaluates the expanded pieces in 256-bit precision
  for k in COMPACT
    b, cc, _ = _cdfpieces(k)
    function F(t)
      j = something(findlast(i -> b[i] <= t, 1:length(b)-1), 1)
      return evalpoly(t - (big(b[j]) + b[j + 1]) / 2, big.(cc[j]))
    end
    for j in 1:15, u in (-1 + 10.0^-j, 1 - 10.0^-j)
      @test k(u) >= 0
      @test k(u) ≈ formula(k, big(u)) rtol = 1e-13
      @test cdf(k, u) ≈ F(big(u)) rtol = 1e-13
    end
    @test cdf(k, -1.0) == 0 && cdf(k, 1.0) == 1 && cdf(k, 0.0) == 0.5
    @test isnan(k(NaN)) && k(Inf) == 0 && k(-Inf) == 0
    @test k(1) === k(1.0) && cdf(k, 0) === 0.5
  end
  @test isnan(UniformKernel()(NaN))
  @test insupport(EpanechnikovKernel(), 0.5) && insupport(EpanechnikovKernel(), -1.0)
  @test !insupport(EpanechnikovKernel(), 1.5) && insupport(GaussianKernel(), 1e10)
end

@testset "Gaussian kernels far out" begin
  g = GaussianKernel()
  @test g(2^32) == 0 && g(Inf) == 0 && isnan(g(NaN))
  for r in (1, 2, 4, 7)
    d = derivative(g, r)
    @test d(Inf) == 0 && d(-Inf) == 0 && d(40.0) == 0 && isnan(d(NaN))
    @test isfinite(d(38.0))
  end
end

@testset "convolutions and products on narrow pieces" begin
  # breakpoints that coincide only up to rounding, -0.7 + 1 and 0.3, make a piece one
  # unit in the last place wide; the convolution must not drop it
  box = PiecewisePolynomialKernel([-0.7, 0.3], [[1.0]])
  c = convolve(TriangularKernel(), box)
  a, b = -0.7, 0.3
  T(t) = t <= -1 ? 0.0 : t >= 1 ? 1.0 : t <= 0 ? (1 + t)^2 / 2 : 1 - (1 - t)^2 / 2
  exact(u) = max(T(min(1.0, u - a)) - T(max(-1.0, u - b)), 0.0)   # ∫_{u-b}^{u-a} triangular
  @test c(0.3) ≈ 0.5 atol = 1e-15
  @test all(abs(c(u) - exact(u)) < 1e-14 for u in range(-1.7, 1.3; length = 3001))
  @test all(abs(c(u) - exact(u)) < 1e-14 for u in (0.3, nextfloat(0.3), prevfloat(0.3), -0.7, nextfloat(-0.7)))
  # products: the factors of a piece one unit in the last place wide
  k1 = PiecewisePolynomialKernel([-1, 0.1, 1], [[1.0], [2.0]])
  k2 = PiecewisePolynomialKernel([-1, nextfloat(0.1), 1], [[3.0], [5.0]])
  p = k1 * k2
  for u in (0.0, 0.1, nextfloat(0.1), 0.5, prevfloat(0.1))
    @test p(u) == k1(u) * k2(u)
  end
end

# a polynomial kernel defined outside the package, with its pieces in tuples
struct TupleKernel <: PolynomialKernel end
KernelSmoothing.pieces(::TupleKernel) = ((-1.0, 0.0, 1.0), ((0.5, 1.0), (0.5, -1.0)))   # the triangular kernel
KernelSmoothing.smoothness(::TupleKernel) = 1

@testset "polynomial kernels defined elsewhere" begin
  k = TupleKernel()
  for u in (-1.5, -0.7, -0.2, 0.0, 0.3, 1.0)
    @test k(u) ≈ TriangularKernel()(u) atol = 1e-15
  end
  @test moment(k, 2) ≈ 1 / 6
  # the distribution function of a named kernel, as a kernel for the sums, has the
  # pieces of its polynomial and no derivative
  g = KernelSmoothing._cdfkernel(EpanechnikovKernel())
  @test smoothness(g) == 0
  @test KernelSmoothing._cdfpieces(g)[3] ≈ KernelSmoothing._cdfpieces(g.pk)[3]
end

@testset "names shared with Distributions" begin
  @test cdf === Distributions.cdf
  @test support === Distributions.support
  @test convolve === Distributions.convolve
  @test moment === Distributions.moment
  @test insupport === Distributions.insupport
  @test cdf.(EpanechnikovKernel(), [-2.0, 0.0, 2.0]) ≈ [0, 0.5, 1]
end

@testset "upper tails and the tails of derived kernels" begin
  # ccdf without the cancellation of 1 - cdf
  for k in COMPACT
    for u in (-0.3, 0.5, 0.9, 1 - 1e-6)
      @test ccdf(k, u) ≈ quadgk(k, u, 1; rtol = 1e-13)[1] rtol = 1e-10
    end
    @test all(abs(cdf(k, u) + ccdf(k, u) - 1) <= 4eps() for u in range(-1.2, 1.2; length = 241))
  end
  @test ccdf(GaussianKernel(), 30.0) ≈ 4.906713927148187e-198 rtol = 1e-12
  g = derivative(GaussianKernel(), 2)
  @test ccdf(g, 1.3) == -cdf(g, 1.3)   # a derivative integrates to zero
  # The convolution of two Epanechnikov kernels near the ends of its support, where
  # its cdf is 3/32 s⁴ - 3/80 s⁵ + O(s⁶) in s = u + 2: the expansions about the
  # midpoints of its pieces turned negative below s = 1e-4.
  E = EpanechnikovKernel()
  EE = convolve(E, E)
  F(s) = 3 / 32 * s^4 - 3 / 80 * s^5
  for j in 6:15
    u = -2 + 10.0^-j
    s = u + 2   # exactly
    @test cdf(EE, u) ≈ F(s) rtol = 1e-10
    @test ccdf(EE, -u) ≈ F(s) rtol = 1e-10
    @test cdf(KernelSmoothing._reflect(EE), u) ≈ F(s) rtol = 1e-10
  end
  # the product (9/16)(1 - u²)², whose cdf is 3/4 s³ - 9/16 s⁴ + 9/80 s⁵ in s = u + 1
  P = E * E
  for j in 1:15
    u = -1 + 10.0^-j
    s = u + 1
    @test cdf(P, u) ≈ 3 / 4 * s^3 - 9 / 16 * s^4 + 9 / 80 * s^5 rtol = 1e-13
    @test ccdf(P, -u) ≈ 3 / 4 * s^3 - 9 / 16 * s^4 + 9 / 80 * s^5 rtol = 1e-13
  end
  # a kernel from coefficients in powers of u, the biweight
  B = PiecewisePolynomialKernel([-1, 1], [[15 / 16, 0, -15 / 8, 0, 15 / 16]])
  for j in 1:15
    @test cdf(B, -1 + 10.0^-j) ≈ cdf(BiweightKernel(), -1 + 10.0^-j) rtol = 1e-13
    @test ccdf(B, 1 - 10.0^-j) ≈ ccdf(BiweightKernel(), 1 - 10.0^-j) rtol = 1e-13
  end
  @test all(abs(cdf(EE, u) + ccdf(EE, u) - 1) <= 4eps() for u in range(-2.5, 2.5; length = 501))
end

@testset "the test for negative kernels does not depend on the scale of u" begin
  # a sixth-order kernel, negative in places, rescaled to [-s, s]
  c = [315 / 512 * a for a in (3.0, 0, -20, 0, 42, 0, -36, 0, 11)]
  for s in (1.0, 100.0, 1000.0, 1e6)
    k = PiecewisePolynomialKernel([-s, s], [[c[i] / s^i for i in eachindex(c)]])
    @test !KernelSmoothing._isnonnegative(k)
    @test KernelSmoothing._isnonnegative(PiecewisePolynomialKernel([-s, s], [[15 / 16 / s, 0, -15 / 8 / s^3, 0, 15 / 16 / s^5]]))
  end
end

# a kernel without closed forms for the moments of its square
struct CosKernel <: AbstractKernel end
(::CosKernel)(u::Real) = abs(u) <= 1 ? π / 4 * cos(π * u / 2) : 0.0
KernelSmoothing.support(::CosKernel) = (-1.0, 1.0)
KernelSmoothing.moment(::CosKernel, j::Integer) = quadgk(u -> u^j * CosKernel()(u), -1, 1)[1]
struct LapKernel <: AbstractKernel end
(::LapKernel)(u::Real) = exp(-abs(u)) / 2
KernelSmoothing.support(::LapKernel) = (-Inf, Inf)

@testset "moments of squared kernels by quadrature, and of high order" begin
  for j in 0:6
    @test moment(KernelSmoothing._square(CosKernel()), j) ≈ quadgk(u -> u^j * CosKernel()(u)^2, -1, 1)[1] rtol = 1e-12 atol = 1e-15
    # ∫ u^j e^(-2|u|) / 4 du = j! / (4 2^j) for even j
    @test moment(KernelSmoothing._square(LapKernel()), j) ≈ (iseven(j) ? factorial(j) / (4 * 2^j) : 0.0) atol = 1e-12
  end
  @test roughness(CosKernel()) ≈ π^2 / 16 rtol = 1e-12
  @test roughness(LapKernel()) ≈ 1 / 4 rtol = 1e-12
  # the binomial coefficients of high orders overflowed integers
  @test moment(EpanechnikovKernel(), 70) ≈ 0.75 * (2 / 71 - 2 / 73) rtol = 1e-12
  @test moment(TriangularKernel(), 80) ≈ 2 / (81 * 82) rtol = 1e-10
end

end
