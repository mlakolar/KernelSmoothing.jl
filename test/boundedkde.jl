module BoundedKDETest

using Test
using KernelSmoothing
using Distributions: Distributions, logpdf, ccdf, insupport
using QuadGK
using Random
using Statistics

include(joinpath(@__DIR__, "data", "bounded_golden.jl"))

const METHODS = [(Reflection(), XG, (0, Inf)), (BetaKernels(), YB, (0, 1)), (BetaKernels(modified = true), YB, (0, 1)),
                 (GammaKernels(), XG, (0, Inf)), (GammaKernels(modified = true), XG, (0, Inf)),
                 (Transformation(), XG, (0, Inf)), (Transformation(), YB, (0, 1)),
                 (Transformation(transform = :probit), YB, (0, 1)), (Reflection(), YB, (0, 1))]

@testset "agreement with evmix" begin
  f = fit(BoundedKDE, XG; support = (0, Inf), kernel = GaussianKernel(), bandwidth = 0.3)
  @test pdf.(f, PX) ≈ REFL_GAUSS rtol = 1e-12
  @test cdf.(f, PX) ≈ REFL_GAUSS_CDF rtol = 1e-11
  f = fit(BoundedKDE, XG; support = (0, Inf), bandwidth = 0.5)
  @test pdf.(f, PX) ≈ REFL_EPAN rtol = 1e-12
  @test cdf.(f, PX) ≈ REFL_EPAN_CDF rtol = 1e-11
  # beta and gamma kernels; evmix normalizes by adaptive quadrature to about 1e-9, and
  # takes the density of gamma kernels at the boundary as zero
  for (m, raw, nrm, c) in ((BetaKernels(), BETA1_RAW, BETA1, BETA1_CDF), (BetaKernels(modified = true), BETA2_RAW, BETA2, BETA2_CDF))
    f = fit(BoundedKDE, YB; support = (0, 1), method = m, bandwidth = 0.05)
    @test pdf.(f, PY) .* f.impl.c ≈ raw rtol = 1e-12
    @test pdf.(f, PY) ≈ nrm rtol = 1e-9
    @test cdf.(f, PY[2:7]) ≈ c rtol = 1e-9
  end
  for (m, raw, nrm, c) in ((GammaKernels(), GAMMA1_RAW, GAMMA1, GAMMA1_CDF), (GammaKernels(modified = true), GAMMA2_RAW, GAMMA2, GAMMA2_CDF))
    f = fit(BoundedKDE, XG; support = (0, Inf), method = m, bandwidth = 0.1)
    @test pdf.(f, PX[2:end]) .* f.impl.c ≈ raw[2:end] rtol = 1e-12
    @test pdf.(f, PX[2:end]) ≈ nrm[2:end] rtol = 1e-7
    @test cdf.(f, PX[2:8]) ≈ c rtol = 1e-6
    @test pdf(f, 0.0) > 0   # the gamma kernel with shape near one is positive at zero
  end
  # evmix shifts by an offset before the logarithm, and integrates the density
  f = fit(BoundedKDE, XG; support = (-1e-12, Inf), method = Transformation(), kernel = GaussianKernel(), bandwidth = 0.3)
  @test pdf.(f, PX[2:8]) ≈ LOGT rtol = 1e-12
  @test cdf.(f, PX[2:8]) ≈ LOGT_CDF atol = 1e-6
end

@testset "a distribution on the support" begin
  for (m, x, s) in METHODS, w in (nothing, 1.0 .+ mod.(1:length(x), 3))
    f = fit(BoundedKDE, x; support = s, method = m, weights = w)
    a, b = s
    # the density integrates to one on the support and is zero outside
    @test quadgk(t -> pdf(f, t), a, b; rtol = 1e-10)[1] ≈ 1 rtol = 1e-7
    @test pdf(f, a - 0.1) == 0 && cdf(f, a - 0.1) == 0 && cdf(f, a) == 0
    isfinite(b) && @test pdf(f, b + 0.1) == 0 && cdf(f, b) == 1
    # the distribution function is the integral of the density
    for t in (0.1, 0.4, 0.9)
      @test cdf(f, t) ≈ quadgk(u -> pdf(f, u), a, t; rtol = 1e-11)[1] rtol = 1e-7 atol = 1e-12
      @test ccdf(f, t) ≈ 1 - cdf(f, t) atol = 1e-14
      @test logpdf(f, t) ≈ log(pdf(f, t))
    end
    # quantiles invert the distribution function
    for p in (0.01, 0.3, 0.5, 0.97)
      q = quantile(f, p)
      @test insupport(f, q)
      @test cdf(f, q) ≈ p atol = 1e-10
    end
    @test quantile(f, 0) == minimum(f) || quantile(f, 0) >= a
    # draws: their mean and quantiles match the estimate
    r = rand(Xoshiro(7), f, 20_000)
    @test all(t -> a <= t <= b, r)
    @test mean(r) ≈ mean(f) rtol = 0.03
    @test quantile(r, 0.5) ≈ quantile(f, 0.5) rtol = 0.04
    μ = mean(f)
    @test μ ≈ quadgk(t -> t * pdf(f, t), a, b; rtol = 1e-11)[1] rtol = 1e-8
    @test var(f) ≈ quadgk(t -> (t - μ)^2 * pdf(f, t), a, b; rtol = 1e-11)[1] rtol = 1e-8
    @test nobs(f) == length(x) && bandwidth(f) > 0
  end
end

@testset "reflection" begin
  x = XG
  # with one boundary, the estimate is twice the kernel estimate of the data and
  # their images
  f = fit(BoundedKDE, x; support = (0, Inf), bandwidth = 0.4)
  g = fit(KDE, vcat(x, -x); bandwidth = 0.4)
  @test pdf(f, 0.7) ≈ 2 * pdf(g, 0.7)
  @test pdf(f, 0.0) ≈ 2 * pdf(g, 0.0)
  # the bandwidth rule sees the data and their images
  @test bandwidth(fit(BoundedKDE, x; support = (0, Inf))) == bandwidth(SheatherJones(), SmoothingData(vcat(x, -x)), EpanechnikovKernel())
  # an upper boundary, by symmetry
  fu = fit(BoundedKDE, -x; support = (-Inf, 0), bandwidth = 0.4)
  @test pdf(fu, -0.7) ≈ pdf(f, 0.7)
  @test cdf(fu, -0.7) ≈ 1 - cdf(f, 0.7)
  @test quantile(fu, 0.2) ≈ -quantile(f, 0.8)
  # two boundaries with a Gaussian kernel wider than the support: repeated images
  y = YB
  fw = fit(BoundedKDE, y; support = (0, 1), kernel = GaussianKernel(), bandwidth = 0.8)
  @test quadgk(t -> pdf(fw, t), 0, 1)[1] ≈ 1 rtol = 1e-10
  @test pdf(fw, 0.0) ≈ pdf(fw, 1.0) rtol = 0.05   # nearly flat
  # the derivative vanishes at the boundary
  @test (pdf(f, 1e-6) - pdf(f, 0.0)) / 1e-6 ≈ 0 atol = 1e-3
  @test_throws ArgumentError fit(BoundedKDE, x; support = (0, Inf),
                                 kernel = PiecewisePolynomialKernel([-1.0, 0.0, 2.0], [[2/3, 2/3], [2/3, -1/3]]))
end

@testset "transformation" begin
  x = XG
  f = fit(BoundedKDE, x; support = (0, Inf), method = Transformation(), bandwidth = 0.3)
  g = fit(KDE, log.(x); bandwidth = 0.3)
  @test pdf(f, 1.3) ≈ pdf(g, log(1.3)) / 1.3
  @test cdf(f, 1.3) ≈ cdf(g, log(1.3))
  @test quantile(f, 0.3) ≈ exp(quantile(g, 0.3))
  y = YB
  fl = fit(BoundedKDE, y; support = (0, 1), method = Transformation(), bandwidth = 0.3)
  gl = fit(KDE, log.(y ./ (1 .- y)); bandwidth = 0.3)
  @test pdf(fl, 0.4) ≈ pdf(gl, log(0.4 / 0.6)) / (0.4 * 0.6)
  @test fl.method.transform == :logit
  # an upper boundary
  fu = fit(BoundedKDE, -x; support = (-Inf, 0), method = Transformation(), bandwidth = 0.3)
  @test pdf(fu, -1.3) ≈ pdf(f, 1.3)
  @test cdf(fu, -1.3) ≈ 1 - cdf(f, 1.3)
  @test quantile(fu, 0.3) ≈ -quantile(f, 0.7)
  @test_throws ArgumentError fit(BoundedKDE, vcat(0.0, x); support = (0, Inf), method = Transformation())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = Transformation(transform = :log))
  @test_throws ArgumentError fit(BoundedKDE, x; support = (0, Inf), method = Transformation(transform = :logit))
  @test_throws ArgumentError Transformation(transform = :sqrt)
end

@testset "beta and gamma kernels" begin
  y = YB
  # the default smoothing parameter minimizes the cross-validation criterion over a grid
  f = fit(BoundedKDE, y; support = (0, 1), method = BetaKernels())
  @test f.selection.rule == LSCV(gridsize = 30)
  @test 0 < bandwidth(f) < 0.24
  @test bandwidth(f) == bandwidth(fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), bandwidth = LSCV(gridsize = 30)))
  # a support [a, b] rescales to [0, 1]
  f1 = fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), bandwidth = 0.05)
  f2 = fit(BoundedKDE, 2 .* y .+ 1; support = (1, 3), method = BetaKernels(), bandwidth = 0.05)
  @test pdf(f2, 1.8) ≈ pdf(f1, 0.4) / 2
  @test cdf(f2, 1.8) ≈ cdf(f1, 0.4)
  # gamma kernels on (-∞, b] reflect the data at b
  x = XG
  g1 = fit(BoundedKDE, x; support = (0, Inf), method = GammaKernels(), bandwidth = 0.1)
  g2 = fit(BoundedKDE, 5 .- x; support = (-Inf, 5), method = GammaKernels(), bandwidth = 0.1)
  @test pdf(g2, 5 - 1.2) ≈ pdf(g1, 1.2)
  @test cdf(g2, 5 - 1.2) ≈ 1 - cdf(g1, 1.2)
  @test isfinite(bandwidth(fit(BoundedKDE, x; support = (0, Inf), method = GammaKernels())))
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, Inf), method = BetaKernels())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = GammaKernels())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), bandwidth = SheatherJones())
end

@testset "errors and display" begin
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (1, Inf))          # data outside
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (-Inf, Inf))
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (2, 1))
  @test_throws ArgumentError fit(BoundedKDE, XG; support = "positive")
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (0, Inf), bandwidth = -1)
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (0, Inf), kernel = derivative(GaussianKernel(), 1))
  @test_throws DomainError quantile(fit(BoundedKDE, XG; support = (0, Inf)), 1.5)
  f = fit(BoundedKDE, XG; support = (0, Inf), bandwidth = 0.5)
  @test sprint(show, f) == "BoundedKDE(Reflection(), support = (0.0, Inf), bandwidth = 0.5, 150 observations)"
  @test sprint(show, BetaKernels(modified = true)) == "BetaKernels(modified = true)"
  @test isnan(pdf(f, NaN)) && isnan(cdf(f, NaN))
end

end
