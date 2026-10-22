module KernelsTest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, PolynomialKernel, pieces
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
  @test k(0.3) == EpanechnikovKernel()(0.3)
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
  @test sprint(show, convolve(UniformKernel(), UniformKernel())) ==
        "PiecewisePolynomialKernel(2 pieces on [-2.0, 2.0], smoothness 1)"
end

@testset "names shared with Distributions" begin
  @test cdf === Distributions.cdf
  @test support === Distributions.support
  @test convolve === Distributions.convolve
  @test moment === Distributions.moment
  @test cdf.(EpanechnikovKernel(), [-2.0, 0.0, 2.0]) ≈ [0, 0.5, 1]
end

end
