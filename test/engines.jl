module EnginesTest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, _sumvalues!, _choose_engine

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
  v = _sumvalues!(SumsWorkspace(), d, z)
  a, b = support(k)
  R = 3 * max(abs(a), abs(b))
  supk = maximum(abs ∘ k, range(a, b; length = 2001))
  M = zeros(kmax + 1, length(points))
  for j in eachindex(points), i in eachindex(d.x)
    abs((d.x[i] - points[j]) / (h isa Real ? h : h[j])) <= R || continue
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
  dr = SmoothingData(XR, YR; weights = WR)
  @test check_exact(dr, TriangularKernel(), 0.35, dr.x, 2)   # 2000 points: several chunks
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

@testset "FastSums for high powers" begin
  # relative to ∑ᵢ |wᵢ k(uᵢ)|, blocks one window wide lost up to 7.8e-7 for the
  # convolution of two triweight kernels with kmax = 6; narrower blocks keep 1e-13
  x = [mod(0.6180339887498949i, 1.0) for i in 1:20_000]
  d = SmoothingData(x)
  pts = collect(range(0.05, 0.95; length = 151))
  h = 0.01
  tw = TriweightKernel()
  for (k, kmax) in ((EpanechnikovKernel(), 6), (tw, 2), (tw, 6), (convolve(tw, tw), 0), (convolve(tw, tw), 6),
                    (convolve(BiweightKernel(), BiweightKernel()), 2))
    F = kernelsums(d, k, h, pts; kmax, engine = FastSums())
    D = kernelsums(d, k, h, pts; kmax, engine = DirectSums())
    scale = [sum(abs(k((xi - p) / h)) for xi in x) for p in pts]
    @test maximum(abs.(F .- D) ./ scale') < 1e-13
  end
  @test KernelSmoothing._blockfactor(4) == 1 && KernelSmoothing._blockfactor(8) == 2
  @test KernelSmoothing._blockfactor(19) == 4
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

@testset "bandwidth per point" begin
  d = SmoothingData(XR, YR; weights = WR)
  smooth = [0.05 + 0.6 * abs(sin(3p)) for p in POINTS]
  jumpy = [isodd(j) ? 0.05 : 2.0 for j in eachindex(POINTS)]   # boundaries move both ways
  for hs in (smooth, jumpy), k in (EpanechnikovKernel(), TriangularKernel(),
                                   convolve(EpanechnikovKernel(), EpanechnikovKernel()))
    D = kernelsums(d, k, hs, POINTS; kmax = 2, z = d.y, engine = DirectSums())
    # each column is the sum with that point's bandwidth alone
    @test D == reduce(hcat, [kernelsums(d, k, hs[j], [POINTS[j]]; kmax = 2, z = d.y,
                                        engine = DirectSums()) for j in eachindex(POINTS)])
    @test check_exact(d, k, hs, POINTS, 2; z = d.y)
    @test check_exact(d, k, reverse(hs), reverse(POINTS), 2)
  end
  g = GaussianKernel()
  @test relerr(kernelsums(d, g, smooth, POINTS; engine = BinnedSums()),
               kernelsums(d, g, smooth, POINTS)) < 1e-3
  @test_throws DimensionMismatch kernelsums(d, EpanechnikovKernel(), smooth[1:10], POINTS)
  @test_throws ArgumentError kernelsums(d, EpanechnikovKernel(), [-1.0; smooth[2:end]], POINTS)
  @test_throws ArgumentError kernelsums(d, EpanechnikovKernel(), [Inf; smooth[2:end]], POINTS)
end

@testset "engine choice" begin
  d = SmoothingData(XR)
  # few small windows: windowed sums; many large ones: fast sums
  @test _choose_engine(EpanechnikovKernel(), d.x, 0.01, POINTS[1:5], 0) isa WindowSums
  @test _choose_engine(EpanechnikovKernel(), d.x, 0.5, d.x, 2) isa FastSums
  @test _choose_engine(GaussianKernel(), d.x, 0.5, d.x, 2) isa DirectSums
  # one tiny bandwidth among many would make the blocks of fast sums tiny for all
  hs = fill(0.5, length(d.x)); hs[100] = 1e-6
  @test _choose_engine(EpanechnikovKernel(), d.x, hs, d.x, 2) isa WindowSums
  # windowed terms cost more for kernels whose pieces are vectors: windows 0.004
  # wide are summed as windows for the Epanechnikov kernel, while for its
  # convolution with itself windows 0.08 wide take fast sums
  dl = SmoothingData([mod(0.7548776662466927i, 1.0) for i in 1:100_000])
  pts = collect(range(0, 1; length = 100))
  @test _choose_engine(EpanechnikovKernel(), dl.x, 0.002, pts, 0) isa WindowSums
  @test _choose_engine(convolve(EpanechnikovKernel(), EpanechnikovKernel()), dl.x, 0.02, pts, 0) isa FastSums
  # Gaussian kernels: the observations within 38.62 standard deviations, beyond
  # which the kernel is exactly zero, give the direct sums exactly
  @test _choose_engine(GaussianKernel(), d.x, 0.01, POINTS, 0) isa KernelSmoothing._ZeroWindowSums
  dy = SmoothingData(XR, YR; weights = WR)
  for k in (GaussianKernel(), derivative(GaussianKernel(), 3)), h in (0.005, 0.05, 0.5)
    @test kernelsums(dy, k, h, POINTS; kmax = 2, z = dy.y) ==
          kernelsums(dy, k, h, POINTS; kmax = 2, z = dy.y, engine = DirectSums())
  end
  g = fit(KDE, dy; kernel = GaussianKernel(), bandwidth = 0.01)
  @test [pdf(g, t) for t in POINTS] == vec(kernelsums(dy, GaussianKernel(), 0.01, POINTS; engine = DirectSums())) ./
                                       (0.01 * dy.sumw)
  for (h, points) in ((0.01, POINTS[1:5]), (0.5, d.x))
    @test check_exact(d, EpanechnikovKernel(), h, points, 2)
    @test kernelsums(d, EpanechnikovKernel(), h, points; kmax = 2) ==
          kernelsums(d, EpanechnikovKernel(), h, points; kmax = 2,
                     engine = _choose_engine(EpanechnikovKernel(), d.x, h, points, 2))
  end
end

@testset "workspace" begin
  d = SmoothingData(XR, YR; weights = WR)
  ws = SumsWorkspace()
  for (k, e, kmax) in ((EpanechnikovKernel(), FastSums(), 2), (TriangularKernel(), WindowSums(), 1),
                       (GaussianKernel(), BinnedSums(), 0), (BiweightKernel(), FastSums(), 4),
                       (EpanechnikovKernel(), DirectSums(), 3))
    @test kernelsums(d, k, 0.3, POINTS; kmax, z = d.y, engine = e, workspace = ws) ==
          kernelsums(d, k, 0.3, POINTS; kmax, z = d.y, engine = e)
  end
  out = zeros(3, nobs(d))
  sumsinto(ws) =
    kernelsums!(out, d, EpanechnikovKernel(), 0.3, d.x; engine = FastSums(), workspace = ws)
  sumsinto(ws); sumsinto(SumsWorkspace())
  @test @allocated(sumsinto(ws)) < @allocated(sumsinto(SumsWorkspace())) / 2
  @test sprint(show, ws) == "SumsWorkspace()"
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

# a kernel whose evaluation fails
struct FailingKernel <: AbstractKernel end
(::FailingKernel)(u::Real) = throw(DomainError(u, "this kernel fails"))
KernelSmoothing.support(::FailingKernel) = (-1.0, 1.0)

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
  @test_throws "at least one row" kernelsums!(zeros(0, 81), d, k, 0.3, POINTS)
  @test_throws MethodError kernelsums!(zeros(Float32, 1, 81), d, k, 0.3, POINTS)
  @test kernelsums(d, k, Float64[], Float64[]) == zeros(1, 0)
  # an error raised in a task is thrown as itself, whatever the number of points
  for m in (5, 300, 3000)
    @test_throws DomainError kernelsums(d, FailingKernel(), 0.3, collect(range(0, 1; length = m)); engine = DirectSums())
  end
end

@testset "FastSums with bandwidths far above the spread of the data" begin
  # one block, a window wide, holds all the data; its power sums are taken about the
  # midpoint of the observations, not of the block, so that they do not cancel
  x = [mod(0.6180339887498949 * i, 1) for i in 1:1000]
  d = SmoothingData(x)
  k = EpanechnikovKernel()
  pts = collect(range(0, 1; length = 11))
  for h in (1e3, 1e5, 1e7)
    F = kernelsums(d, k, h, pts; kmax = 2, engine = FastSums())
    D = kernelsums(d, k, h, pts; kmax = 2, engine = DirectSums())
    for r in 1:3   # relative to the sum of the absolute terms, since the sums cancel
      scale = [sum(abs(k((xi - p) / h) * ((xi - p) / h)^(r - 1)) for xi in x) for p in pts]
      @test all(abs.(F[r, :] .- D[r, :]) .<= 1e-13 .* scale)
    end
  end
  @test !any(isnan, kernelsums(d, k, 1e308, pts; kmax = 2))
  # the slope of a linear function, from the sums of the chosen engine
  m = fit(LocalPolynomialRegression, x, 2 .+ 3 .* x; bandwidth = 1e7, derivative = 1, points = [0.5])
  @test predict(m, collect(range(0, 1; length = 201)))[101] ≈ 3 rtol = 1e-12
  # at a tie the sums with u and u² vanish exactly
  @test kernelsums(SmoothingData(fill(-3.0, 50)), k, 0.1, [-3.0]; kmax = 2, engine = FastSums())[2:3] == [0.0, 0.0]
end

@testset "many powers, overflowing arguments, other output arrays and interrupts" begin
  d = SmoothingData(XR)
  k = EpanechnikovKernel()
  pts = collect(range(-6, 6; length = 300))
  # beyond 16 powers the accumulators are the columns of out, with the same order of
  # summation, so that windowed and direct sums stay identical
  for kmax in (15, 16, 40)
    @test kernelsums(d, k, 0.8, pts; kmax, engine = WindowSums()) == kernelsums(d, k, 0.8, pts; kmax, engine = DirectSums())
  end
  @test kernelsums(d, k, 0.8, pts; kmax = 40, engine = FastSums()) ≈ kernelsums(d, k, 0.8, pts; kmax = 40, engine = DirectSums())
  @test (@allocated kernelsums(d, k, 0.8, pts; kmax = 40, engine = WindowSums())) < 10^6
  # an argument that overflows to ±Inf, where the kernel vanishes, adds nothing
  @test kernelsums(SmoothingData([0.0, 1e300]), k, 1e-10, [0.0, 1.0]; kmax = 2, engine = DirectSums()) ==
        [0.75 0.0; 0.0 0.0; 0.0 0.0]
  @test !any(isnan, kernelsums(SmoothingData([0.0, 1e300]), GaussianKernel(), 1e-10, [0.0]; kmax = 2, engine = DirectSums()))
  # the tasks write columns of out, which only a strided array allows; others, such as
  # a sparse matrix or this transpose, receive a copy
  @test kernelsums!(view(zeros(5, 300), 1:3, :), d, k, 0.8, pts) == kernelsums(d, k, 0.8, pts; kmax = 2)
  T = transpose(zeros(300, 3))
  @test kernelsums!(T, d, k, 0.8, pts) === T
  @test T == kernelsums(d, k, 0.8, pts; kmax = 2)
  # an interrupt while waiting for the tasks is thrown at once, not after them
  c = Channel{Int}(2)
  tasks = [Threads.@spawn take!(c) for _ in 1:2]
  waiter = @async KernelSmoothing._fetchall(tasks)
  sleep(0.1)
  schedule(waiter, InterruptException(); error = true)
  @test timedwait(() -> istaskdone(waiter), 10.0) == :ok
  @test istaskfailed(waiter) && waiter.result isa InterruptException
  put!(c, 1); put!(c, 2)
  @test fetch.(tasks) == [1, 2] || fetch.(tasks) == [2, 1]
end

end
