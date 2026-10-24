module KDETest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, GaussianDerivativeKernel, _reflect,
  _isnonnegative, _kernelquantile, _randkernel
using Distributions: Distributions, ccdf, cquantile, logcdf, logccdf, insupport
using ForwardDiff
using QuadGK
using Random
using Statistics

# grid data with many ties and observations exactly at window edges
const XS = round.(7 .* sin.(1:500) .+ cos.(3 .* (1:500)); digits = 1)
const WS = 1.0 .+ mod.(1:500, 4)
# smooth data
const XR = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:2000]
# a small sample, for integrals
const XQ = [-2.1, -0.4, 0.0, 0.3, 0.3, 1.7, 2.5]
const WQ = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0]

# an asymmetric density kernel on [-1, 2]: 2/3 (1 + u) on [-1, 0], 2/3 (1 - u/2) on [0, 2]
const ASYM = PiecewisePolynomialKernel([-1.0, 0.0, 2.0], [[2/3, 2/3], [2/3, -1/3]])

const DENSITY_KERNELS = [UniformKernel(), TriangularKernel(), EpanechnikovKernel(),
                         BiweightKernel(), TriweightKernel(), GaussianKernel(), ASYM,
                         convolve(EpanechnikovKernel(), EpanechnikovKernel()),
                         GaussianDerivativeKernel(2.0, 0)]

# the estimates by their definitions, with weights w
bf_pdf(x, w, k, h, t) = sum(w[i] * k((t - x[i]) / h) for i in eachindex(x)) / (h * sum(w))
bf_cdf(x, w, k, h, t) = sum(w[i] * cdf(k, (t - x[i]) / h) for i in eachindex(x)) / sum(w)
function bf_se(x, w, k, h, t)
  g = [k((t - xi) / h) / h for xi in x]
  fx = sum(w .* g) / sum(w)
  return sqrt(sum(w .^ 2 .* (g .- fx) .^ 2)) / sum(w)
end

points(x, h) = sort!(unique!(vcat(range(minimum(x) - 2h, maximum(x) + 2h; length = 61),
                                  x[1:7:end], x[1:11:end] .+ h, x[2:13:end] .- h)))

@testset "density and distribution function" begin
  for (x, w) in ((XS, WS), (XR, ones(length(XR)))), k in DENSITY_KERNELS, h in (0.3, 1.5)
    f = fit(KDE, x; kernel = k, bandwidth = h, weights = w)
    ts = points(x, h)
    P = [bf_pdf(x, w, k, h, t) for t in ts]
    C = [bf_cdf(x, w, k, h, t) for t in ts]
    S = [bf_se(x, w, k, h, t) for t in ts]
    @test pdf.(Ref(f), ts) ≈ P rtol = 1e-12 atol = 1e-14 * maximum(P)   # point by point
    @test pdf.(f, ts) ≈ P rtol = 1e-11 atol = 1e-13 * maximum(P)        # with an engine
    @test [cdf(f, t) for t in ts] ≈ C atol = 1e-13
    @test cdf.(f, ts) ≈ C atol = 1e-12
    @test [stderror(f, t) for t in ts] ≈ S rtol = 1e-10 atol = 1e-13 * maximum(S)
    @test stderror.(f, ts) ≈ S rtol = 1e-9 atol = 1e-12 * maximum(S)
    @test all(>=(-1e-14), diff(cdf.(f, ts))) && all(>=(0), pdf.(f, ts))
    @test cdf(f, minimum(f)) == 0 && cdf(f, maximum(f)) == 1
  end
end

@testset "integrals" begin
  for w in (ones(length(XQ)), WQ), k in DENSITY_KERNELS, h in (0.4, 2.0)
    f = fit(KDE, XQ; kernel = k, bandwidth = h, weights = w)
    lo, hi = minimum(f), maximum(f)
    # integrate between the kinks of the estimate
    a, b = support(k)
    kinks = isfinite(a) ? sort!(unique!(vcat([XQ .+ h * t for t in KernelSmoothing.pieces(k)[1]]...))) :
                          [-Inf, Inf]
    integral(g) = sum(quadgk(g, kinks[i], kinks[i + 1]; rtol = 1e-12)[1] for i in 1:length(kinks)-1)
    @test integral(t -> pdf(f, t)) ≈ 1 rtol = 1e-10
    @test integral(t -> t * pdf(f, t)) ≈ mean(f) rtol = 1e-9 atol = 1e-10
    @test integral(t -> (t - mean(f))^2 * pdf(f, t)) ≈ var(f) rtol = 1e-9
    for t in (-1.0, 0.3, 1.9)
      @test cdf(f, t) ≈ quadgk(s -> pdf(f, s), isfinite(lo) ? lo : -Inf, t; rtol = 1e-12)[1] atol = 1e-10
    end
  end
end

@testset "quantiles" begin
  for (x, w) in ((XS, WS), (XQ, WQ)), k in DENSITY_KERNELS, h in (0.3, 1.5)
    f = fit(KDE, x; kernel = k, bandwidth = h, weights = w)
    for p in (1e-9, 0.01, 0.25, 0.5, 0.9, 1 - 1e-9)
      q = quantile(f, p)
      @test cdf(f, q) ≈ p atol = 1e-12
      @test cdf(f, q - 1e-9 * max(1, abs(q))) < p + 1e-12   # the smallest such point
    end
    @test quantile(f, 0) == minimum(f) && quantile(f, 1) == maximum(f)
    @test median(f) == quantile(f, 0.5)
  end
  # where the distribution function is flat, the left end of the flat part; it is
  # determined to about √eps only, since F̂(x) = 1/2 - O((1.1 - x)²) below it
  f = fit(KDE, [0.0, 0.1, 10.0, 10.1]; bandwidth = 1.0)
  @test quantile(f, 0.5) ≈ 1.1 atol = 1e-7
  @test_throws DomainError quantile(f, 1.5)
  @test quantile(fit(KDE, [2.0]; kernel = GaussianKernel(), bandwidth = 0.5), 0.975) ≈
        2 + 0.5 * 1.959963984540054 rtol = 1e-12
end

@testset "sampling" begin
  rng = MersenneTwister(20261023)
  N = 100_000
  for k in DENSITY_KERNELS
    u = [_randkernel(rng, k) for _ in 1:N]
    a, b = support(k)
    @test all(t -> a <= t <= b, u)
    ts = range(isfinite(a) ? a : -3.0, isfinite(b) ? b : 3.0; length = 41)
    @test maximum(abs(mean(u .<= t) - cdf(k, t)) for t in ts) < 2 / sqrt(N)
  end
  for k in (EpanechnikovKernel(), GaussianKernel(), ASYM)
    f = fit(KDE, XS; kernel = k, bandwidth = 0.8, weights = WS)
    s = rand(rng, f, N)
    @test abs(mean(s) - mean(f)) < 5 * std(f) / sqrt(N)
    @test var(s) ≈ var(f) rtol = 0.03
    ts = quantile.(f, 0.05:0.05:0.95)
    @test maximum(abs(mean(s .<= t) - cdf(f, t)) for t in ts) < 2 / sqrt(N)
  end
  f = fit(KDE, XQ; bandwidth = 0.8)
  @test rand(f) isa Float64 && size(rand(f, 2, 3)) == (2, 3)
end

nthderiv(g, t, r) = r == 0 ? g(t) : ForwardDiff.derivative(s -> nthderiv(g, s, r - 1), t)

@testset "derivatives" begin
  for (k, rs) in ((EpanechnikovKernel(), 1:1), (TriangularKernel(), 1:1), (BiweightKernel(), 1:2),
                  (TriweightKernel(), 1:3), (GaussianKernel(), 1:4), (ASYM, 1:1))
    for (x, w) in ((XS, WS), (XQ, WQ)), r in rs
      h = 0.7
      f = fit(KDE, x; kernel = k, bandwidth = h, weights = w)
      g = derivative(f, r)
      # off the kinks of the estimate: at a jump of k⁽ʳ⁾ its derivative does not exist
      kinks = isfinite(support(k)[1]) ? KernelSmoothing.pieces(k)[1] : ()
      ts = [t for t in points(x, h) if all(!((t - xi) / h in kinks) for xi in x)]
      # the r-th derivative of the estimate, by automatic differentiation
      D = [nthderiv(t -> bf_pdf(x, w, k, h, t), t, r) for t in ts]
      @test g.(ts) ≈ D rtol = 1e-8 atol = 1e-10 * maximum(abs, D)
      @test [g(t) for t in ts] ≈ g.(ts) rtol = 1e-11 atol = 1e-13 * maximum(abs, D)
    end
  end
  @test_throws ArgumentError derivative(fit(KDE, XQ; bandwidth = 0.8), 2)   # the Epanechnikov kernel has one derivative
  @test derivative(fit(KDE, XQ; bandwidth = 0.8), 0)(0.3) ≈ pdf(fit(KDE, XQ; bandwidth = 0.8), 0.3)
end

@testset "kernels for densities" begin
  signed = [derivative(BiweightKernel()), PiecewisePolynomialKernel([-1, 1], [[9/8, 0, -15/8]]),
            GaussianDerivativeKernel(1.0, 2), GaussianDerivativeKernel(1.0, 0, -1.0)]
  for k in signed
    @test !_isnonnegative(k)
    @test_throws ArgumentError fit(KDE, XQ; kernel = k, bandwidth = 1.0)
  end
  for k in DENSITY_KERNELS
    @test _isnonnegative(k)
  end
  @test _isnonnegative(EpanechnikovKernel() * TriangularKernel())
  @test_throws ArgumentError fit(KDE, XQ; kernel = EpanechnikovKernel() * EpanechnikovKernel(), bandwidth = 0.8)   # mass 3/5
  # reflection: u ↦ k(-u)
  for k in vcat(DENSITY_KERNELS, signed, [derivative(TriweightKernel(), 3), derivative(ASYM)])
    r = _reflect(k)
    @test all(r(u) ≈ k(-u) for u in range(-2.37, 2.41; length = 97))
    @test support(r) == (-support(k)[2], -support(k)[1])
  end
  for k in DENSITY_KERNELS, p in (0.001, 0.3, 0.5, 0.999)
    @test cdf(k, _kernelquantile(k, p)) ≈ p atol = 1e-13
  end
  # the test for negative values does not depend on the scale of u: the fourth-order
  # kernel 315/512 (3 - 11v²)(1 - v²)³ rescaled to [-s, s], whose minimum is -0.2/s
  c = [315 / 512 * a for a in (3.0, 0, -20, 0, 42, 0, -36, 0, 11)]
  for s in (1.0, 0.1, 0.03, 1e-3)
    k = PiecewisePolynomialKernel([-s, s], [[c[i] / s^i for i in eachindex(c)]])
    @test moment(k, 0) ≈ 1
    @test !_isnonnegative(k)
    @test_throws ArgumentError fit(KDE, XQ; kernel = k, bandwidth = 1.0)
    @test _isnonnegative(PiecewisePolynomialKernel([-s, s], [[0.75 / s, 0, -0.75 / s^3]]))
  end
  # a kernel whose mass is just below one has no quantile above its mass
  kq = PiecewisePolynomialKernel([-1, 1], [[0.5 - 1e-12]])
  @test _kernelquantile(kq, nextfloat(moment(kq, 0))) == 1
  fq = fit(KDE, [0.0, 1.0]; kernel = kq, bandwidth = 0.5)
  @test quantile(fq, 1 - 1e-13) <= maximum(fq)
end

@testset "non-finite points and broadcasting" begin
  f = fit(KDE, XS; weights = WS)
  @test pdf(f, Inf) == 0 && pdf(f, -Inf) == 0 && isnan(pdf(f, NaN))
  @test cdf(f, Inf) == 1 && cdf(f, -Inf) == 0 && isnan(cdf(f, NaN))
  @test logpdf(f, 100.0) == -Inf && logpdf(f, 0.0) ≈ log(pdf(f, 0.0))
  v = pdf.(f, [NaN, Inf, -Inf, 0.0])
  @test isnan(v[1]) && v[2:3] == [0, 0] && v[4] ≈ pdf(f, 0.0)
  @test isequal(cdf.(f, [NaN, Inf, -Inf]), [NaN, 1.0, 0.0])
  M = [0.0 1.0; -2.0 3.5]
  @test pdf.(f, M) ≈ [pdf(f, t) for t in M] && size(pdf.(f, M)) == (2, 2)
  @test pdf.(f, -3:0.5:3) ≈ [pdf(f, t) for t in -3:0.5:3]
  @test logpdf.(f, [0.0, 1.0]) ≈ log.([pdf(f, 0.0), pdf(f, 1.0)])
  @test 2 .* pdf.(f, [0.0, 1.0]) ≈ 2 .* [pdf(f, 0.0), pdf(f, 1.0)]
  @test pdf.(f, Float64[]) == Float64[]
  g = derivative(f)
  @test isnan(g(NaN)) && g(Inf) == 0 && g.([Inf, 0.5]) ≈ [0, g(0.5)]
  # fused arguments take the route of arrays; dual numbers that of the scalar methods
  ts = collect(-3:0.5:3)
  @test pdf.(f, ts .+ 0.25) == pdf.(f, ts .+ 0.25 .+ 0.0)
  @test cdf.(f, 2 .* ts) ≈ [cdf(f, 2t) for t in ts] && ccdf.(f, ts .- 1) ≈ [ccdf(f, t - 1) for t in ts]
  @test g.(ts .+ 0.25) ≈ [g(t + 0.25) for t in ts]
  @test ForwardDiff.derivative(t -> sum(pdf.(f, [t, t + 1])), 0.3) ≈ g(0.3) + g(1.3)
  @test ForwardDiff.derivative(t -> sum(cdf.(f, [t, 2t])), 0.3) ≈ pdf(f, 0.3) + 2pdf(f, 0.6)
  @test ForwardDiff.derivative(t -> sum(logpdf.(f, [t])), 0.3) ≈ g(0.3) / pdf(f, 0.3)
end

@testset "engines" begin
  ts = collect(range(-8, 8; length = 101))
  for k in (EpanechnikovKernel(), TriweightKernel())
    f = fit(KDE, XR; kernel = k, bandwidth = 0.5)
    D = pdf.(fit(KDE, XR; kernel = k, bandwidth = 0.5, engine = DirectSums()), ts)
    @test pdf.(fit(KDE, XR; kernel = k, bandwidth = 0.5, engine = WindowSums()), ts) == D
    @test pdf.(fit(KDE, XR; kernel = k, bandwidth = 0.5, engine = FastSums()), ts) ≈ D rtol = 1e-11
    @test cdf.(fit(KDE, XR; kernel = k, bandwidth = 0.5, engine = FastSums()), ts) ≈ cdf.(f, ts) atol = 1e-12
  end
  fg = fit(KDE, XR; kernel = GaussianKernel(), bandwidth = 0.5)
  fb = fit(KDE, XR; kernel = GaussianKernel(), bandwidth = 0.5, engine = BinnedSums())
  @test pdf.(fb, ts) ≈ pdf.(fg, ts) rtol = 1e-3
end

@testset "distribution interface" begin
  f = fit(KDE, XQ; weights = WQ, bandwidth = 0.9)
  @test insupport(f, 0.0) && !insupport(f, 10.0)
  @test Distributions.support(f) == Distributions.RealInterval(minimum(f), maximum(f))
  @test minimum(f) == -2.1 - 0.9 && maximum(f) == 2.5 + 0.9
  @test ccdf(f, 0.2) ≈ 1 - cdf(f, 0.2) && logcdf(f, 0.2) ≈ log(cdf(f, 0.2))
  @test std(f) ≈ sqrt(var(f))
  @test bandwidth(f) == 0.9 && nobs(f) == 7
  g = fit(KDE, XQ; kernel = GaussianKernel(), bandwidth = 0.8)
  @test minimum(g) == -Inf && maximum(g) == Inf && insupport(g, 1e6)
  @test endswith(sprint(show, f), "EpanechnikovKernel(), bandwidth = 0.9, 7 observations, weighted)")
  @test occursin("derivative of order 1", sprint(show, derivative(f)))
  @test_throws ArgumentError fit(KDE, XQ; bandwidth = -1.0)
  # an engine that cannot sum the kernel is refused when fitting, not at each broadcast
  @test_throws ArgumentError fit(KDE, XQ; kernel = GaussianKernel(), bandwidth = 0.8, engine = FastSums())
  @test_throws ArgumentError fit(KDE, XQ; kernel = GaussianKernel(), bandwidth = 0.8, engine = WindowSums())
  @test_throws ArgumentError fit(KDE, XQ; bandwidth = "wide")
  d = SmoothingData(XQ, ones(7))
  @test pdf(fit(KDE, d; bandwidth = 0.8), 0.5) == pdf(fit(KDE, XQ; bandwidth = 0.8), 0.5)   # responses are ignored
end

# a kernel defined outside the package, with unbounded support
struct LaplaceKernel <: AbstractKernel end
(::LaplaceKernel)(u::Real) = exp(-abs(u)) / 2
KernelSmoothing.support(::LaplaceKernel) = (-Inf, Inf)
KernelSmoothing.cdf(::LaplaceKernel, u::Real) = u < 0 ? exp(u) / 2 : 1 - exp(-u) / 2
KernelSmoothing.moment(::LaplaceKernel, j::Integer) = iseven(j) ? Float64(factorial(j)) : 0.0

# and a compactly supported one
struct CosineKernel <: AbstractKernel end
(::CosineKernel)(u::Real) = abs(u) <= 1 ? π / 4 * cos(π * u / 2) : 0.0
KernelSmoothing.support(::CosineKernel) = (-1.0, 1.0)
KernelSmoothing.cdf(::CosineKernel, u::Real) = u <= -1 ? 0.0 : u >= 1 ? 1.0 : (1 + sin(π * u / 2)) / 2
KernelSmoothing.moment(::CosineKernel, j::Integer) = j == 0 ? 1.0 : quadgk(u -> u^j * CosineKernel()(u), -1, 1)[1]

@testset "other kernels" begin
  kc = CosineKernel()
  fc = fit(KDE, XS; kernel = kc, bandwidth = 0.8, weights = WS)
  ts = points(XS, 0.8)
  @test stderror.(fc, ts) ≈ [bf_se(XS, WS, kc, 0.8, t) for t in ts] rtol = 1e-10 atol = 1e-14
  @test cdf.(fc, ts) ≈ [bf_cdf(XS, WS, kc, 0.8, t) for t in ts] atol = 1e-12
  k, h = LaplaceKernel(), 0.6
  f = fit(KDE, XQ; kernel = k, bandwidth = h, weights = WQ)
  ts = collect(range(-5, 5; length = 41))
  @test pdf.(f, ts) ≈ [bf_pdf(XQ, WQ, k, h, t) for t in ts] rtol = 1e-12
  @test cdf.(f, ts) ≈ [bf_cdf(XQ, WQ, k, h, t) for t in ts] rtol = 1e-12
  @test stderror.(f, ts) ≈ [bf_se(XQ, WQ, k, h, t) for t in ts] rtol = 1e-10
  @test cdf(f, quantile(f, 0.01)) ≈ 0.01 atol = 1e-12
  @test var(f) ≈ sum(WQ .* (XQ .- sum(WQ .* XQ) / sum(WQ)) .^ 2) / sum(WQ) + 2h^2 rtol = 1e-12
  s = rand(MersenneTwister(5), f, 20_000)
  @test abs(mean(s) - mean(f)) < 5 * std(f) / sqrt(20_000)
  # a polynomial kernel written with zero coefficients of high powers
  e0 = PiecewisePolynomialKernel([-1, 1], [[0.75, 0, -0.75, 0, 0]])
  @test _isnonnegative(e0) && pdf(fit(KDE, XQ; kernel = e0, bandwidth = 1.0), 0.2) ≈
        pdf(fit(KDE, XQ; bandwidth = 1.0), 0.2)
  @test sprint(show, BandwidthSelection(0.5)) == "BandwidthSelection(0.5)"
end

@testset "tails" begin
  # a triweight estimate near the end of its support, where the expanded pieces cancel
  f = fit(KDE, [0.0, 10.0]; kernel = TriweightKernel(), bandwidth = 1.0)
  t = 0.9999997
  @test pdf(f, t) ≈ 35 / 64 * (1 - big(t)^2)^3 rtol = 1e-12
  @test pdf.(f, [t]) ≈ [pdf(f, t)] rtol = 1e-12
  @test logpdf(f, t) ≈ log(pdf(f, t))
  # for data symmetric about zero, 1 - F̂(t) = F̂(-t), also far in the upper tail
  v = [Distributions.quantile(Distributions.Normal(), (i - 0.5) / 100) for i in 1:50]
  x = [v; -v]
  for k in (GaussianKernel(), EpanechnikovKernel(), TriweightKernel()), h in (0.3, 1.1)
    f = fit(KDE, x; kernel = k, bandwidth = h)
    ts = [0.0, 0.4, 1.7, maximum(x) + 0.5h, maximum(x) + 0.99h, maximum(x) + 2h]
    k isa GaussianKernel && append!(ts, maximum(x) .+ [9h, 30h])
    for t in ts
      @test ccdf(f, t) ≈ cdf(f, -t) rtol = 1e-12
      @test ccdf(f, t) ≈ 1 - cdf(f, t) atol = 1e-14
    end
    @test ccdf.(f, ts) ≈ [ccdf(f, t) for t in ts] rtol = 1e-11
    @test ccdf.(f, ts) ≈ cdf.(f, -ts) rtol = 1e-11
    for p in (1e-12, 1e-3, 0.3, 0.5, 0.8)
      @test cquantile(f, p) ≈ -quantile(f, p) rtol = 1e-10 atol = 1e-12
    end
    @test cquantile(f, 0) == maximum(f) && cquantile(f, 1) == minimum(f)
  end
  g = fit(KDE, x; kernel = GaussianKernel(), bandwidth = 0.3)
  t = maximum(x) + 9 * 0.3
  @test ccdf(g, t) > 1e-25
  @test logccdf(g, t) ≈ log(cdf(g, -t)) rtol = 1e-12
  # tail quantiles, where Newton steps in a convex tail shrink slowly
  for p in (1e-200, 1e-300)
    @test cdf(g, quantile(g, p)) ≈ p rtol = 1e-8
    @test ccdf(g, cquantile(g, p)) ≈ p rtol = 1e-8
  end
  # asymmetric kernels and other kernels
  for k in (ASYM, CosineKernel(), LaplaceKernel())
    f = fit(KDE, XQ; kernel = k, bandwidth = 0.7, weights = WQ)
    for t in (-2.0, 0.1, 1.9, 3.0)
      @test ccdf(f, t) ≈ 1 - cdf(f, t) atol = 1e-13
    end
    @test ccdf.(f, [-2.0, 0.1, 1.9]) ≈ [ccdf(f, t) for t in (-2.0, 0.1, 1.9)] rtol = 1e-11
    @test cquantile(f, 0.01) ≈ quantile(f, 0.99) rtol = 1e-10
  end
end

@testset "type stability" begin
  for k in (EpanechnikovKernel(), GaussianKernel())
    f = fit(KDE, XQ; kernel = k, weights = WQ, bandwidth = 0.8)
    @inferred pdf(f, 0.3)
    @inferred logpdf(f, 0.3)
    @inferred cdf(f, 0.3)
    @inferred ccdf(f, 0.3)
    @inferred quantile(f, 0.3)
    @inferred cquantile(f, 0.3)
    @inferred stderror(f, 0.3)
    @inferred mean(f)
    @inferred var(f)
    @inferred rand(MersenneTwister(1), f)
    @inferred derivative(f)(0.3)
  end
end

# the Laplace kernel with its upper tail
struct LaplaceTailKernel <: AbstractKernel end
(::LaplaceTailKernel)(u::Real) = exp(-abs(u)) / 2
KernelSmoothing.support(::LaplaceTailKernel) = (-Inf, Inf)
KernelSmoothing.cdf(::LaplaceTailKernel, u::Real) = u < 0 ? exp(u) / 2 : 1 - exp(-u) / 2
KernelSmoothing.ccdf(::LaplaceTailKernel, u::Real) = u > 0 ? exp(-u) / 2 : 1 - exp(u) / 2
KernelSmoothing.moment(::LaplaceTailKernel, j::Integer) = iseven(j) ? Float64(factorial(j)) : 0.0

@testset "tails of broadcasts and of user kernels" begin
  xx = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:20000]
  # near the ends of the support, where fast sums lose their relative accuracy
  for k in (BiweightKernel(), TriweightKernel())
    f = fit(KDE, xx; kernel = k, bandwidth = 0.5)
    pts = [collect(range(-7.6, 7.6; length = 2000)); maximum(f) - 1e-4; minimum(f) + 1e-4]
    for (vector, scalar) in ((pdf.(f, pts), [pdf(f, t) for t in pts]),
                             (cdf.(f, pts), [cdf(f, t) for t in pts]),
                             (ccdf.(f, pts), [ccdf(f, t) for t in pts]))
      @test all(isapprox.(vector, scalar; rtol = 1e-10))
    end
    @test all(isapprox.(logpdf.(f, pts), [logpdf(f, t) for t in pts]; rtol = 1e-12))
  end
  # BinnedSums takes the weight beyond the window from the binned data too
  f0 = fit(KDE, [0.0, 0.501, 3.0]; bandwidth = 1.0)
  f1 = fit(KDE, [0.0, 0.501, 3.0]; bandwidth = 1.0, engine = BinnedSums())
  @test cdf.(f1, [1.505])[1] ≈ cdf(f0, 1.505) atol = 1e-4
  @test ccdf.(f1, [1.505])[1] ≈ ccdf(f0, 1.505) atol = 1e-4
  # a user kernel with an upper tail keeps it in ccdf and cquantile
  x, h = [-2.1, -0.4, 0.0, 0.3, 0.3, 1.7, 2.5], 0.6
  f = fit(KDE, x; kernel = LaplaceTailKernel(), bandwidth = h)
  exact = sum(exp(-(30 - xi) / h) / 2 for xi in x) / length(x)
  @test ccdf(f, 30.0) ≈ exact rtol = 1e-12
  @test ccdf.(f, [0.0, 30.0])[2] ≈ exact rtol = 1e-12
  @test ccdf(f, cquantile(f, 1e-20)) ≈ 1e-20 rtol = 1e-10
  @test ccdf(f, Distributions.invlogccdf(f, log(1e-20))) ≈ 1e-20 rtol = 1e-10
  # the reflection swaps the tails
  r = _reflect(LaplaceTailKernel())
  @test ccdf(r, -30.0) == cdf(LaplaceTailKernel(), 30.0) && cdf(r, -30.0) == ccdf(LaplaceTailKernel(), 30.0)
  # the Gaussian log-density far from the data, where the density underflows
  for w in (nothing, [1.0, 2.0, 0.5])
    xs = [0.0, 1.0, 1.5]
    g = fit(KDE, xs; kernel = GaussianKernel(), bandwidth = 0.5, weights = w)
    ws = w === nothing ? ones(3) : w
    for t in (40.0, -40.0, 1e3)
      a = [log(ws[i]) - ((t - xs[i]) / 0.5)^2 / 2 for i in 1:3]
      exact = maximum(a) + log(sum(exp.(a .- maximum(a)))) - log(sum(ws) * 0.5 * sqrt(2π))
      @test logpdf(g, t) ≈ exact rtol = 1e-13
    end
    @test logpdf.(g, [0.5, 40.0, -40.0]) ≈ [logpdf(g, t) for t in (0.5, 40.0, -40.0)] rtol = 1e-13
  end
end

@testset "standard errors at tiny bandwidths and incomplete data" begin
  x = [sin(1.618i) for i in 1:100]
  f, g = fit(KDE, x; bandwidth = 1e-160), fit(KDE, x .* 1e160; bandwidth = 1.0)
  # the standard error scales as 1 / h, also where h² underflows
  @test stderror(f, x[50]) * 1e-160 ≈ stderror(g, x[50] * 1e160) rtol = 1e-12
  @test stderror.(f, [x[50]]) == [stderror(f, x[50])]
  @test_throws "missing values" fit(KDE, [1.0, missing, 2.0])
  @test_throws "missing values" fit(PointwiseDensity, [1.0, missing, 2.0])
  @test_throws "missing values" fit(LocalPolynomialDensity, [1.0, missing, 2.0])
  @test_throws "missing values" fit(LocalPolynomialRegression, [1.0, missing, 2.0], [1.0, 2.0, 3.0])
  @test_throws "missing values" fit(LocalPolynomialRegression, [1.0, 2.0, 3.0], [1.0, missing, 3.0])
end

@testset "stderror broadcasts in one pass" begin
  x = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:5000]
  w = 1 .+ mod.(1:5000, 4)
  pts = [collect(range(-8, 8; length = 1500)); [0.3, -2.0, 7.9]]
  for k in (EpanechnikovKernel(), BiweightKernel(), GaussianKernel()), weights in (nothing, w)
    f = fit(KDE, x; kernel = k, bandwidth = 0.4, weights)
    @test stderror.(f, pts) == [stderror(f, t) for t in pts]
  end
  # an engine fixed by the estimate sums the three sums with it
  fw = fit(KDE, x; bandwidth = 0.4, weights = w, engine = WindowSums())
  @test all(isapprox.(stderror.(fw, pts), [stderror(fw, t) for t in pts]; rtol = 1e-12, atol = 1e-15))
end

end
