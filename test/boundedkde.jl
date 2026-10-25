module BoundedKDETest

using Test
using KernelSmoothing
using Distributions: Distributions, logpdf, ccdf, insupport
using QuadGK
using Random
using Statistics

include(joinpath(@__DIR__, "data", "bounded_golden.jl"))

# every method on lower, upper and two-sided supports
const METHODS = [(Reflection(), XG, (0, Inf)), (Reflection(), -XG, (-Inf, 0)), (Reflection(), YB, (0, 1)),
                 (Reflection(), 2 .* YB .- 1, (-1, 1)),
                 (BetaKernels(), YB, (0, 1)), (BetaKernels(modified = true), 3 .* YB .+ 2, (2, 5)),
                 (GammaKernels(), XG, (0, Inf)), (GammaKernels(modified = true), -XG, (-Inf, 0)),
                 (Transformed(), XG, (0, Inf)), (Transformed(), -XG, (-Inf, 0)), (Transformed(), YB, (0, 1)),
                 (Transformed(transform = :probit), YB, (0, 1))]

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
  f = fit(BoundedKDE, XG; support = (-1e-12, Inf), method = Transformed(), kernel = GaussianKernel(), bandwidth = 0.3)
  @test pdf.(f, PX[2:8]) ≈ LOGT rtol = 1e-12
  @test cdf.(f, PX[2:8]) ≈ LOGT_CDF atol = 1e-6
end

@testset "a distribution on the support" begin
  for (m, x, s) in METHODS, w in (nothing, 1.0 .+ mod.(1:length(x), 3))
    f = fit(BoundedKDE, x; support = s, method = m, weights = w)
    a, b = s
    lo, hi = isfinite(a) ? a : -40.0, isfinite(b) ? b : 40.0
    # the density integrates to one on the support and is zero outside
    @test quadgk(t -> pdf(f, t), lo, hi; rtol = 1e-10)[1] ≈ 1 rtol = 1e-7
    isfinite(a) && @test pdf(f, a - 0.1) == 0 && cdf(f, a - 0.1) == 0 && cdf(f, a) == 0 && ccdf(f, a) == 1
    isfinite(b) && @test pdf(f, b + 0.1) == 0 && cdf(f, b) == 1 && ccdf(f, b) == 0
    @test pdf(f, Inf) == 0 && pdf(f, -Inf) == 0 && logpdf(f, Inf) == -Inf
    # the distribution function is the integral of the density, and its complement
    # the integral above
    for p in (0.1, 0.4, 0.9)
      t = quantile(f, p)
      @test cdf(f, t) ≈ quadgk(u -> pdf(f, u), lo, t; rtol = 1e-11)[1] rtol = 1e-7 atol = 1e-12
      @test ccdf(f, t) ≈ quadgk(u -> pdf(f, u), t, hi; rtol = 1e-11)[1] rtol = 1e-7 atol = 1e-12
      @test logpdf(f, t) ≈ log(pdf(f, t))
      @test cdf(f, t) ≈ p atol = 1e-10   # quantiles invert the distribution function
    end
    @test quantile(f, 0) == minimum(f) && quantile(f, 1) == maximum(f)
    @test insupport(f, quantile(f, 0.5)) && !insupport(f, isfinite(a) ? a - 1 : b + 1)
    # draws: their mean and quantiles match the estimate
    r = rand(Xoshiro(7), f, 20_000)
    @test all(t -> minimum(f) <= t <= maximum(f), r)
    @test mean(r) ≈ mean(f) atol = 0.03 * std(r) + 0.01 * abs(mean(f))
    @test quantile(r, 0.5) ≈ quantile(f, 0.5) atol = 0.05 * std(r)
    μ = mean(f)
    @test μ ≈ quadgk(t -> t * pdf(f, t), lo, hi; rtol = 1e-11)[1] rtol = 1e-8 atol = 1e-12
    @test var(f) ≈ quadgk(t -> (t - μ)^2 * pdf(f, t), lo, hi; rtol = 1e-11)[1] rtol = 1e-8
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
  @test cdf(fu, -0.7) ≈ ccdf(f, 0.7)
  @test quantile(fu, 0.2) ≈ -quantile(f, 0.8)
  # two boundaries: the folded density, by a brute-force sum over many images
  y = YB
  for (k, h) in ((GaussianKernel(), 0.8), (GaussianKernel(), 0.05), (EpanechnikovKernel(), 0.7))
    fw = fit(BoundedKDE, y; support = (0, 1), kernel = k, bandwidth = h)
    folded(t) = sum(k((t - (s * yi + 2m)) / h) for yi in y, m in -60:60, s in (1, -1)) / (h * length(y))
    @test pdf(fw, 0.3) ≈ folded(0.3) rtol = 1e-12
    @test pdf(fw, 1.0) ≈ folded(1.0) rtol = 1e-12
  end
  # the derivative vanishes at the boundary; the logarithm stays finite far out
  @test (pdf(f, 1e-6) - pdf(f, 0.0)) / 1e-6 ≈ 0 atol = 1e-3
  fg = fit(BoundedKDE, x; support = (0, Inf), kernel = GaussianKernel(), bandwidth = 0.1)
  @test pdf(fg, 50.0) == 0 && isfinite(logpdf(fg, 50.0))
  # a symmetric kernel that is not one of the named ones, and an asymmetric one
  kc = convolve(EpanechnikovKernel(), EpanechnikovKernel())
  fc = fit(BoundedKDE, x; support = (0, Inf), kernel = kc, bandwidth = 0.4)
  @test pdf(fc, 0.7) ≈ 2 * pdf(fit(KDE, vcat(x, -x); kernel = kc, bandwidth = 0.4), 0.7)
  @test_throws ArgumentError fit(BoundedKDE, x; support = (0, Inf),
                                 kernel = PiecewisePolynomialKernel([-1.0, 0.0, 2.0], [[2/3, 2/3], [2/3, -1/3]]))
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), kernel = GaussianKernel(), bandwidth = 1000.0)
end

@testset "transformation" begin
  x = XG
  f = fit(BoundedKDE, x; support = (0, Inf), method = Transformed(), bandwidth = 0.3)
  g = fit(KDE, log.(x); bandwidth = 0.3)
  @test pdf(f, 1.3) ≈ pdf(g, log(1.3)) / 1.3
  @test cdf(f, 1.3) ≈ cdf(g, log(1.3))
  @test quantile(f, 0.3) ≈ exp(quantile(g, 0.3))
  y = YB
  fl = fit(BoundedKDE, y; support = (0, 1), method = Transformed(), bandwidth = 0.3)
  gl = fit(KDE, log.(y ./ (1 .- y)); bandwidth = 0.3)
  @test pdf(fl, 0.4) ≈ pdf(gl, log(0.4 / 0.6)) / (0.4 * 0.6)
  @test pdf(fl, 0.9) ≈ pdf(gl, log(0.9 / 0.1)) / (0.9 * 0.1)
  @test fl.method.transform == :logit
  fp = fit(BoundedKDE, y; support = (0, 1), method = Transformed(transform = :probit), bandwidth = 0.3)
  gp = fit(KDE, Distributions.quantile.(Distributions.Normal(), y); bandwidth = 0.3)
  z = Distributions.quantile(Distributions.Normal(), 0.8)
  @test pdf(fp, 0.8) ≈ pdf(gp, z) / Distributions.pdf(Distributions.Normal(), z)
  # an upper boundary
  fu = fit(BoundedKDE, -x; support = (-Inf, 0), method = Transformed(), bandwidth = 0.3)
  @test pdf(fu, -1.3) ≈ pdf(f, 1.3)
  @test cdf(fu, -1.3) ≈ ccdf(f, 1.3)
  @test quantile(fu, 0.3) ≈ -quantile(f, 0.7)
  @test_throws ArgumentError fit(BoundedKDE, vcat(0.0, x); support = (0, Inf), method = Transformed())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = Transformed(transform = :log))
  @test_throws ArgumentError fit(BoundedKDE, x; support = (0, Inf), method = Transformed(transform = :logit))
  @test_throws ArgumentError Transformed(transform = :sqrt)
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
  @test cdf(g2, 5 - 1.2) ≈ ccdf(g1, 1.2)
  # the upper tail keeps its relative accuracy
  t = quantile(g1, 1 - 1e-12)
  @test 0 < ccdf(g1, t + 3) < 1e-13
  @test ccdf(g1, t) ≈ 1e-12 rtol = 1e-3
  # the scale of the data does not matter
  for c in (1e-20, 1e20)
    gc = fit(BoundedKDE, c .* x; support = (0, Inf), method = GammaKernels())
    gr = fit(BoundedKDE, x; support = (0, Inf), method = GammaKernels())
    @test bandwidth(gc) ≈ c * bandwidth(gr) rtol = 1e-8
    @test pdf(gc, c * 1.3) ≈ pdf(gr, 1.3) / c rtol = 1e-8
  end
  # data far from the boundary
  far = fit(BoundedKDE, 100 .+ x; support = (0, Inf), method = GammaKernels())
  @test quadgk(t -> pdf(far, t), 95, 115)[1] ≈ 1 rtol = 1e-6
  @test mean(far) ≈ 100 + mean(x) rtol = 1e-3
  # observations at a boundary would be lost
  @test_throws ArgumentError fit(BoundedKDE, vcat(0.0, x); support = (0, Inf), method = GammaKernels())
  @test_throws ArgumentError fit(BoundedKDE, vcat(1.0, y); support = (0, 1), method = BetaKernels(modified = true))
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(modified = true), bandwidth = 0.3)
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, Inf), method = BetaKernels())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = GammaKernels())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), bandwidth = SheatherJones())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), kernel = GaussianKernel())
  @test_throws ArgumentError fit(BoundedKDE, y; support = (0, 1), method = BetaKernels(), bandwidth = 1e-12)
end

@testset "weights, small samples, errors and display" begin
  # zero weights drop observations; extreme weights keep their ratios
  w = 1.0 .+ mod.(1:150, 3)
  for m in (Reflection(), GammaKernels(), Transformed())
    f0 = fit(BoundedKDE, vcat(XG, 50.0); support = (0, Inf), method = m, weights = vcat(w, 0.0), bandwidth = 0.3)
    f1 = fit(BoundedKDE, XG; support = (0, Inf), method = m, weights = w, bandwidth = 0.3)
    @test pdf(f0, 1.3) ≈ pdf(f1, 1.3)
    fe = fit(BoundedKDE, XG; support = (0, Inf), method = m, weights = 1e250 .* w, bandwidth = 0.3)
    @test pdf(fe, 1.3) ≈ pdf(f1, 1.3)
  end
  # one observation with a given bandwidth
  f = fit(BoundedKDE, [0.5]; support = (0, Inf), bandwidth = 1.0)
  @test quadgk(t -> pdf(f, t), 0, 2)[1] ≈ 1
  @test_throws ArgumentError fit(BoundedKDE, fill(0.5, 10); support = (0, Inf), method = Transformed())
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (1, Inf))          # data outside
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (-Inf, Inf))
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (2, 1))
  @test_throws ArgumentError fit(BoundedKDE, XG; support = "positive")
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (0, Inf), bandwidth = -1)
  @test_throws ArgumentError fit(BoundedKDE, XG; support = (0, Inf), kernel = derivative(GaussianKernel(), 1))
  @test_throws DomainError quantile(fit(BoundedKDE, XG; support = (0, Inf)), 1.5)
  f = fit(BoundedKDE, SmoothingData(XG); support = (0, Inf), bandwidth = 0.5)
  @test sprint(show, f) == "BoundedKDE(Reflection(), support = (0.0, Inf), bandwidth = 0.5, 150 observations)"
  @test sprint(show, BetaKernels(modified = true)) == "BetaKernels(modified = true)"
  @test sprint(show, Transformed(transform = :probit)) == "Transformed(:probit)"
  @test isnan(pdf(f, NaN)) && isnan(cdf(f, NaN))
end

end
