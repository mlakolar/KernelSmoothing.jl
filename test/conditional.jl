module ConditionalTest

using Test
using KernelSmoothing
using KernelSmoothing: _rqsimplex, _cdcv
using Distributions: Distributions
using QuadGK
using Random

include(joinpath(@__DIR__, "data", "conditional_golden.jl"))

rel(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-300))
check(z, y, β, τ, w) = sum(w[i] * (y[i] - z[i, :]' * β) * (τ - (y[i] < z[i, :]' * β)) for i in eachindex(y))

@testset "weighted quantile regression by the simplex method" begin
  w = 1.0 .+ mod.(1:150, 4)
  Z = hcat(ones(150), X, X .^ 2)
  for (τ, ref) in ((0.1, RQ_10), (0.5, RQ_50), (0.75, RQ_75))
    β, basis = _rqsimplex(Z, Y, w, τ)
    @test β ≈ ref rtol = 1e-12
    @test length(basis) == 3 && Z[basis, :] * β ≈ Y[basis]   # interpolates its basis
  end
  # small problems against the best line through two observations, which the
  # optimum of a regression on an intercept and a slope is
  rng = Xoshiro(3)
  for _ in 1:30
    n = rand(rng, 4:9)
    x, y, w = randn(rng, n), randn(rng, n), rand(rng, n) .+ 0.1
    τ = rand(rng, (0.2, 0.5, 0.8))
    z = hcat(ones(n), x)
    best = minimum(check(z, y, z[[i, j], :] \ y[[i, j]], τ, w) for i in 1:n for j in i+1:n)
    β, _ = _rqsimplex(z, y, w, τ)
    @test check(z, y, β, τ, w) ≈ best rtol = 1e-12
  end
  # ties in x, zero weights, and designs that do not identify β
  z = hcat(ones(6), [0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
  β, _ = _rqsimplex(z, [0.1, 0.3, 1.2, 0.9, 2.2, 1.7], ones(6), 0.5)
  @test all(isfinite, β)
  @test all(isnan, _rqsimplex(hcat(ones(3), [1.0, 1.0, 1.0]), [1.0, 2.0, 3.0], ones(3), 0.5)[1])
  @test all(isnan, _rqsimplex(hcat(ones(3), [1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], [1.0, 0.0, 0.0], 0.5)[1])
end

@testset "local quantile regression" begin
  for (τ, fv, dv) in ((0.25, LPRQ_FV_25, LPRQ_DV_25), (0.5, LPRQ_FV_50, LPRQ_DV_50), (0.9, LPRQ_FV_90, LPRQ_DV_90))
    m = fit(LocalQuantileRegression, X, Y; tau = τ, bandwidth = 0.4, kernel = GaussianKernel(), points = LPRQ_XX)
    @test coef(m) ≈ fv rtol = 1e-12
    m1 = fit(LocalQuantileRegression, X, Y; tau = τ, bandwidth = 0.4, kernel = GaussianKernel(), points = LPRQ_XX, derivative = 1)
    @test coef(m1) ≈ dv rtol = 1e-11
  end
  # the bandwidth of Yu and Jones rescales the mean-regression bandwidth, the same
  # for τ and 1 - τ
  d = SmoothingData(X, Y)
  s = select_bandwidth(YuJones(), d, EpanechnikovKernel(); tau = 0.25)
  @test s.h ≈ bandwidth(CrossValidation(), d, EpanechnikovKernel()) * s.details.factor
  @test s.h ≈ bandwidth(YuJones(), d, EpanechnikovKernel(); tau = 0.75)
  @test s.h > bandwidth(YuJones(), d, EpanechnikovKernel(); tau = 0.5)
  m = fit(LocalQuantileRegression, X, Y; tau = 0.25)
  @test bandwidth(m) == s.h && m.selection.rule == YuJones()
  # weights: integer weights act as repeated observations
  w = 1 .+ mod.(1:150, 3)
  mw = fit(LocalQuantileRegression, X, Y; weights = w, tau = 0.5, bandwidth = 0.5, points = [-1.0, 0.0, 1.0])
  idx = reduce(vcat, [fill(i, w[i]) for i in 1:150])
  mr = fit(LocalQuantileRegression, X[idx], Y[idx]; tau = 0.5, bandwidth = 0.5, points = [-1.0, 0.0, 1.0])
  @test coef(mw) ≈ coef(mr) rtol = 1e-10
  # standard errors and intervals
  se = stderror(m)
  @test all(v -> isfinite(v) && v > 0, se)
  ci = confint(m; level = 0.9)
  @test all(ci[:, 1] .< coef(m) .< ci[:, 2])
  @test ci[:, 2] .- ci[:, 1] ≈ 2 * Distributions.quantile(Distributions.Normal(), 0.95) .* se
  @test all(isfinite, stderror(fit(LocalQuantileRegression, X, Y; tau = 0.5, derivative = 1)))
  # ties in x
  xt = round.(X; digits = 1)
  @test all(isfinite, coef(fit(LocalQuantileRegression, xt, Y; tau = 0.5, bandwidth = 0.5)))
  # a window with too few distinct values gives NaN
  @test isnan(only(coef(fit(LocalQuantileRegression, X, Y; bandwidth = 0.01, points = [10.0]))))
  @test sprint(show, m) == "LocalQuantileRegression(tau = 0.25, degree = 1, bandwidth = $(round(m.h; sigdigits = 4)), 30 points, 150 observations)"
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; tau = 1.0)
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; degree = 1, derivative = 2)
  @test_throws ArgumentError fit(LocalQuantileRegression, SmoothingData(X); tau = 0.5)
  @test_throws ArgumentError confint(m; level = 1.5)
  @test_throws ArgumentError select_bandwidth(YuJones(), d, EpanechnikovKernel(); tau = 0.0)
end

@testset "conditional density" begin
  # the local constant estimate of np, whose bandwidths are ordered (y, x)
  for (bw, ref) in (((0.3, 0.5), NPCDENS_03_05), ((0.6, 0.25), NPCDENS_06_025))
    m = fit(ConditionalDensity, X, Y; bandwidth = reverse(bw))
    @test [pdf(m, EY[k], EX[k]) for k in 1:4] ≈ ref rtol = 1e-12
  end
  m = fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3))
  # a density in y, whose distribution is a weighted KDE
  for x0 in (-1.0, 0.0, 1.5)
    @test quadgk(t -> pdf(m, t, x0), -Inf, Inf)[1] ≈ 1 rtol = 1e-8
    k = conditional(m, x0)
    @test k isa KDE
    @test pdf(k, 0.2) ≈ pdf(m, 0.2, x0)
    @test cdf(k, 0.2) ≈ cdf(m, 0.2, x0)
    @test cdf(m, quantile(m, 0.3, x0), x0) ≈ 0.3 atol = 1e-10
  end
  # the local linear estimate smooths the kernels of the responses with the weights
  # of local linear regression
  m1 = fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3), degree = 1)
  L = smoother_matrix(X, [0.7]; bandwidth = 0.4, degree = 1, kernel = GaussianKernel())
  @test pdf(m1, 0.2, 0.7) ≈ sum(L[1, i] * pdf(Distributions.Normal(Y[i], 0.3), 0.2) for i in eachindex(Y))
  @test cdf(m1, 0.2, 0.7) ≈ sum(L[1, i] * Distributions.cdf(Distributions.Normal(Y[i], 0.3), 0.2) for i in eachindex(Y))
  @test_throws ArgumentError conditional(m1, 0.7)
  # cross-validation: the likelihood agrees with np's selection to its tolerance
  d = SmoothingData(X, Y)
  s = select_bandwidth(ConditionalCV(), d, GaussianKernel())
  @test s.h ≈ CVML rtol = 1e-3
  @test _cdcv(:likelihood, d, GaussianKernel(), GaussianKernel(), s.h...) <=
        _cdcv(:likelihood, d, GaussianKernel(), GaussianKernel(), CVML...) * (1 + 1e-12)
  @test bandwidth(fit(ConditionalDensity, X, Y)) == Tuple(s.h)
  # least squares: our criterion at our minimum is below its value at np's
  # bandwidths, whose own criterion differs
  sl = select_bandwidth(ConditionalCV(criterion = :leastsquares), d, GaussianKernel())
  @test _cdcv(:leastsquares, d, GaussianKernel(), GaussianKernel(), sl.h...) <=
        _cdcv(:leastsquares, d, GaussianKernel(), GaussianKernel(), CVLS...)
  # compact kernels, and weights
  me = fit(ConditionalDensity, X, Y; kernel = EpanechnikovKernel(), weights = 1 .+ mod.(1:150, 3))
  @test all(isfinite, bandwidth(me))
  @test quadgk(t -> pdf(me, t, 0.0), -10, 10; maxevals = 10^6)[1] ≈ 1 rtol = 1e-6
  @test isnan(pdf(fit(ConditionalDensity, X, Y; bandwidth = (0.1, 0.3), kernel = EpanechnikovKernel()), 0.0, 9.0))
  @test sprint(show, m) == "ConditionalDensity(degree = 0, bandwidth = (0.4, 0.3), 150 observations)"
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; bandwidth = 0.3)
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; degree = 1)
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3), ykernel = derivative(GaussianKernel(), 1))
  @test_throws ArgumentError ConditionalCV(criterion = :aic)
  @test_throws ArgumentError fit(ConditionalDensity, SmoothingData(X); bandwidth = (0.4, 0.3))
end

end
