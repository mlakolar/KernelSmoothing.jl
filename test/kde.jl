module KDETest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, GaussianDerivativeKernel, _reflect,
  _isnonnegative, _kernelquantile, _randkernel
using Distributions: Distributions, ccdf, logcdf, insupport
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
  f = fit(KDE, XQ)
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
  @test_throws ArgumentError derivative(fit(KDE, XQ), 2)   # the Epanechnikov kernel has one derivative
  @test derivative(fit(KDE, XQ), 0)(0.3) ≈ pdf(fit(KDE, XQ), 0.3)
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
  @test_throws ArgumentError fit(KDE, XQ; kernel = EpanechnikovKernel() * EpanechnikovKernel())   # mass 3/5
  # reflection: u ↦ k(-u)
  for k in vcat(DENSITY_KERNELS, signed, [derivative(TriweightKernel(), 3), derivative(ASYM)])
    r = _reflect(k)
    @test all(r(u) ≈ k(-u) for u in range(-2.37, 2.41; length = 97))
    @test support(r) == (-support(k)[2], -support(k)[1])
  end
  for k in DENSITY_KERNELS, p in (0.001, 0.3, 0.5, 0.999)
    @test cdf(k, _kernelquantile(k, p)) ≈ p atol = 1e-13
  end
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
  g = fit(KDE, XQ; kernel = GaussianKernel())
  @test minimum(g) == -Inf && maximum(g) == Inf && insupport(g, 1e6)
  @test endswith(sprint(show, f), "EpanechnikovKernel(), bandwidth = 0.9, 7 observations, weighted)")
  @test occursin("derivative of order 1", sprint(show, derivative(f)))
  @test_throws ArgumentError fit(KDE, XQ; bandwidth = -1.0)
  @test_throws ArgumentError fit(KDE, XQ; bandwidth = "wide")
  d = SmoothingData(XQ, ones(7))
  @test pdf(fit(KDE, d), 0.5) == pdf(fit(KDE, XQ), 0.5)   # responses are ignored
end

# a kernel defined outside the package, with unbounded support
struct LaplaceKernel <: AbstractKernel end
(::LaplaceKernel)(u::Real) = exp(-abs(u)) / 2
KernelSmoothing.support(::LaplaceKernel) = (-Inf, Inf)
KernelSmoothing.cdf(::LaplaceKernel, u::Real) = u < 0 ? exp(u) / 2 : 1 - exp(-u) / 2
KernelSmoothing.moment(::LaplaceKernel, j::Integer) = iseven(j) ? Float64(factorial(j)) : 0.0

@testset "other kernels" begin
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

@testset "type stability" begin
  for k in (EpanechnikovKernel(), GaussianKernel())
    f = fit(KDE, XQ; kernel = k, weights = WQ)
    @inferred pdf(f, 0.3)
    @inferred logpdf(f, 0.3)
    @inferred cdf(f, 0.3)
    @inferred quantile(f, 0.3)
    @inferred stderror(f, 0.3)
    @inferred mean(f)
    @inferred var(f)
    @inferred rand(MersenneTwister(1), f)
    @inferred derivative(f)(0.3)
  end
end

end
