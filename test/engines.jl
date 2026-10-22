module EnginesTest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, _sumvalues

# grid data with many ties and observations exactly at window edges
const XS = round.(7 .* sin.(1:500) .+ cos.(3 .* (1:500)); digits = 1)
const YS = cos.(1:500)
const WS = Float64.(mod.(1:500, 4))
# smooth data
const XR = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:2000]
const YR = cos.(1:2000)
const WR = 1.0 .+ mod.(1:2000, 3)
# data far from zero: 1000 + U(0, 1)
const XF = 1000 .+ mod.((1:5000) .* 0.6180339887498949, 1.0)

const KERNELS = [UniformKernel(), TriangularKernel(), EpanechnikovKernel(), BiweightKernel(),
                 TriweightKernel(), derivative(TriweightKernel(), 2), derivative(TriangularKernel(), 1),
                 convolve(EpanechnikovKernel(), EpanechnikovKernel()),
                 convolve(TriangularKernel(), TriangularKernel()),
                 PiecewisePolynomialKernel([-1.0, 0.25, 2.0], [[0.5, 0.2], [0.1, -0.05, 0.3]])]

const POINTS = collect(range(-10, 10; length = 81))

# The scale of the rounding error of fast sums: ∑ |wᵢ zᵢ| over the observations
# within three kernel radii of the point, times sup |k| and the largest |u|^r there.
# (Relative to the sum itself the error can be larger, where the sum is nearly zero.)
function localscale(d, k, h, points, kmax; z = nothing)
  v = _sumvalues(d, z)
  a, b = support(k)
  R = 3 * max(abs(a), abs(b))
  supk = maximum(abs ∘ k, range(a, b; length = 2001))
  M = zeros(kmax + 1, length(points))
  for j in eachindex(points), i in eachindex(d.x)
    abs((d.x[i] - points[j]) / h) <= R || continue
    for r in 1:kmax+1
      M[r, j] += abs(v[i]) * supk * R^(r - 1)
    end
  end
  return M
end

# the exact engines against DirectSums: WindowSums adds the same terms in the same
# order, and FastSums agrees to rounding
function check_exact(d, k, h, points, kmax; z = nothing)
  D = kernelsums(d, k, h, points; kmax, z, engine = DirectSums())
  W = kernelsums(d, k, h, points; kmax, z, engine = WindowSums())
  F = kernelsums(d, k, h, points; kmax, z, engine = FastSums())
  M = localscale(d, k, h, points, kmax; z)
  return W == D && all(abs.(F .- D) .<= 1e-13 .* M .+ 1e-300)
end

relerr(B, D) = maximum(abs, B .- D) / maximum(abs, D)

@testset "exact engines agree" begin
  for d in (SmoothingData(XS, YS; weights = WS), SmoothingData(XR, YR; weights = WR)),
      k in KERNELS, h in (0.1, 0.35, 1.5)
    @test check_exact(d, k, h, POINTS, 3)
    @test check_exact(d, k, h, POINTS, 3; z = d.y)
    @test check_exact(d, k, h, d.x[1:4:end], 2)   # at observations: many exact edge hits
  end
  dg = SmoothingData(XS)
  @test check_exact(dg, EpanechnikovKernel(), 0.1, reverse(POINTS), 2)   # unsorted points
  @test check_exact(SmoothingData([2.0]), EpanechnikovKernel(), 0.5, [1.5, 2.0, 3.0], 2)
  @test check_exact(SmoothingData(fill(3.0, 50)), TriangularKernel(), 0.2, [2.9, 3.0, 3.1, 3.2], 3)
end

@testset "FastSums on data far from zero" begin
  d = SmoothingData(XF)
  points = collect(range(1000, 1001; length = 201))
  for h in (0.005, 0.0005), k in (EpanechnikovKernel(), TriweightKernel())
    @test check_exact(d, k, h, points, 6)   # power sums about one centre would fail here
  end
  # a bandwidth below the resolution of the data falls back to windowed sums
  t = SmoothingData([1e6, 1e6 + 1.0])
  @test kernelsums(t, EpanechnikovKernel(), 1e-12, [1e6]; engine = FastSums()) ==
        kernelsums(t, EpanechnikovKernel(), 1e-12, [1e6]; engine = DirectSums())
end

@testset "Gaussian kernels" begin
  d = SmoothingData(XR; weights = WR)
  @test_throws ArgumentError kernelsums(d, GaussianKernel(), 0.3, POINTS; engine = WindowSums())
  @test_throws ArgumentError kernelsums(d, GaussianKernel(), 0.3, POINTS; engine = FastSums())
  for k in (GaussianKernel(), derivative(GaussianKernel(), 4))
    D = kernelsums(d, k, 0.3, POINTS; kmax = 2)   # DirectSums by default
    @test D == kernelsums(d, k, 0.3, POINTS; kmax = 2, engine = DirectSums())
    @test relerr(kernelsums(d, k, 0.3, POINTS; kmax = 2, engine = BinnedSums()), D) < 1e-3
  end
end

@testset "BinnedSums" begin
  d = SmoothingData(XR)
  errors(k, h) = [relerr(kernelsums(d, k, h, POINTS; engine = BinnedSums(; binsperh = m)),
                         kernelsums(d, k, h, POINTS; engine = DirectSums())) for m in (25, 50, 100)]
  # second order for continuous kernels (1/16 over two doublings of binsperh), much
  # slower for kernels that jump
  for k in (TriweightKernel(), EpanechnikovKernel(), GaussianKernel())
    e = errors(k, 0.4)
    @test e[2] / e[1] < 0.4 && e[3] / e[2] < 0.4 && e[3] / e[1] < 0.16
    @test e[3] < 1e-3
  end
  for k in (UniformKernel(), derivative(TriangularKernel(), 1))
    e = errors(k, 0.4)
    @test e[3] < e[2] < e[1] && e[3] / e[1] > 0.2
  end
  @test !isexact(BinnedSums()) && isexact(FastSums()) && isexact(WindowSums()) && isexact(DirectSums())
  @test BinnedSums().binsperh == 50 && BinnedSums(; binsperh = 80) == BinnedSums(80)
  @test_throws ArgumentError BinnedSums(; binsperh = 0)
end

@testset "density estimate" begin
  # f̂(x) = S[1] / (h ∑w) integrates to one (up to the Riemann sum's error)
  d = SmoothingData(XR; weights = WR)
  for k in (EpanechnikovKernel(), GaussianKernel())
    h = 0.4
    grid = range(minimum(XR) - 6h, maximum(XR) + 6h; length = 4001)
    f = vec(kernelsums(d, k, h, grid)) ./ (h * d.sumw)
    @test sum(f) * step(grid) ≈ 1 atol = 1e-4
  end
end

@testset "interface" begin
  d = SmoothingData(XS, YS)
  k = EpanechnikovKernel()
  @test kernelsums(d, k, 0.3) == kernelsums(d, k, 0.3, d.x)
  @test size(kernelsums(d, k, 0.3, POINTS; kmax = 4)) == (5, 81)
  @test kernelsums(d, k, 0.3, reverse(POINTS)) == reverse(kernelsums(d, k, 0.3, POINTS); dims = 2)
  out = zeros(3, 81)
  @test kernelsums!(out, d, k, 0.3, POINTS) === out
  @test out == kernelsums(d, k, 0.3, POINTS; kmax = 2)
  @test kernelsums(d, k, 1, 0:2) == kernelsums(d, k, 1.0, [0.0, 1.0, 2.0])
  @test @inferred(kernelsums(d, k, 0.3, POINTS)) isa Matrix{Float64}
  @test_throws ArgumentError kernelsums(d, k, 0.0, POINTS)
  @test_throws ArgumentError kernelsums(d, k, -1.0, POINTS)
  @test_throws ArgumentError kernelsums(d, k, Inf, POINTS)
  @test_throws ArgumentError kernelsums(d, k, NaN, POINTS)
  @test_throws ArgumentError kernelsums(d, k, 0.3, [0.0, NaN])
  @test_throws ArgumentError kernelsums(d, k, 0.3, POINTS; kmax = -1)
  @test_throws DimensionMismatch kernelsums(d, k, 0.3, POINTS; z = YS[1:10])
  @test_throws ArgumentError kernelsums(d, k, 0.3, POINTS; z = fill(NaN, 500))
  @test_throws DimensionMismatch kernelsums!(zeros(3, 80), d, k, 0.3, POINTS)
end

end
