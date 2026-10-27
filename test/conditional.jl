module ConditionalTest

using Test
using KernelSmoothing
using KernelSmoothing: _rqsimplex, _cdcv
using Distributions: Distributions
using LinearAlgebra
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
  # equivariance: in the scale and location of the responses and the scale of the
  # weights, also for weights far below one (w again for all observations, since the
  # loop above reassigned it)
  w = 1.0 .+ mod.(1:150, 4)
  β, _ = _rqsimplex(Z, Y, w, 0.3)
  γ = [3.0, -1.0, 0.5]
  @test _rqsimplex(Z, 2.5 .* Y .+ Z * γ, 1e-30 .* w, 0.3)[1] ≈ 2.5 .* β .+ γ rtol = 1e-10
  @test _rqsimplex(Z, Y .+ 1e8, w, 0.3)[1] ≈ β .+ [1e8, 0, 0] rtol = 1e-12
  # Bland's rule from the first pivot reaches the same optimum; the iteration limit warns
  @test _rqsimplex(Z, Y, w, 0.5; bland = -1)[1] ≈ RQ_50 rtol = 1e-10
  @test all(isnan, (@test_logs (:warn, r"iteration limit") _rqsimplex(Z, Y, w, 0.5; maxiter = 1))[1])
  # tied covariates and degenerate problems against the best line through two
  # observations with distinct covariates
  for _ in 1:30
    n = rand(rng, 5:10)
    x, y, w = Float64.(rand(rng, 1:3, n)), Float64.(rand(rng, -2:2, n)), rand(rng, (1.0, 1e4), n)
    length(unique(x)) >= 2 || continue
    τ = rand(rng, (0.25, 0.5))
    z = hcat(ones(n), x)
    best = minimum(check(z, y, z[[i, j], :] \ y[[i, j]], τ, w) for i in 1:n for j in i+1:n if x[i] != x[j])
    β, _ = _rqsimplex(z, y, w, τ)
    @test check(z, y, β, τ, w) ≈ best rtol = 1e-10 atol = 1e-10
  end
end

@testset "local quantile regression" begin
  for (τ, fv, dv) in ((0.25, LPRQ_FV_25, LPRQ_DV_25), (0.5, LPRQ_FV_50, LPRQ_DV_50), (0.9, LPRQ_FV_90, LPRQ_DV_90))
    m = fit(LocalQuantileRegression, X, Y; tau = τ, bandwidth = 0.4, kernel = GaussianKernel(), points = LPRQ_XX)
    @test coef(m) ≈ fv rtol = 1e-12
    m1 = fit(LocalQuantileRegression, X, Y; tau = τ, bandwidth = 0.4, kernel = GaussianKernel(), points = LPRQ_XX,
             derivative = 1, degree = 1)
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
  @test nobs(m) == length(X)
  # weights: integer weights act as repeated observations for the estimates
  w = 1 .+ mod.(1:150, 3)
  mw = fit(LocalQuantileRegression, X, Y; weights = w, tau = 0.5, bandwidth = 0.5, points = [-1.0, 0.0, 1.0])
  idx = reduce(vcat, [fill(i, w[i]) for i in 1:150])
  mr = fit(LocalQuantileRegression, X[idx], Y[idx]; tau = 0.5, bandwidth = 0.5, points = [-1.0, 0.0, 1.0])
  @test coef(mw) ≈ coef(mr) rtol = 1e-10
  @test coef(mw; method = RobustBiasCorrected()) ≈ coef(mr; method = RobustBiasCorrected()) rtol = 1e-10
  # standard errors and intervals
  se = stderror(m)
  @test all(v -> isfinite(v) && v > 0, se)
  ci = confint(m; level = 0.9, method = Conventional())
  @test all(ci[:, 1] .< coef(m) .< ci[:, 2])
  @test ci[:, 2] .- ci[:, 1] ≈ 2 * Distributions.quantile(Distributions.Normal(), 0.95) .* se
  @test all(isfinite, stderror(fit(LocalQuantileRegression, X, Y; tau = 0.5, derivative = 1)))
  # ties in x; discrete responses, whose sparsity the difference quotient cannot
  # estimate
  xt = round.(X; digits = 1)
  @test all(isfinite, coef(fit(LocalQuantileRegression, xt, Y; tau = 0.5, bandwidth = 0.5)))
  sed = stderror(fit(LocalQuantileRegression, X, round.(Y); tau = 0.5, bandwidth = 0.5))
  @test any(isnan, sed) && all(v -> isnan(v) || v > 0, sed)
  # the degree follows the derivative, as for local polynomial regression
  @test fit(LocalQuantileRegression, X, Y; derivative = 1, bandwidth = 0.5).degree == 2
  # a window with too few distinct values gives NaN
  @test isnan(only(coef(fit(LocalQuantileRegression, X, Y; bandwidth = 0.01, points = [10.0]))))
  @test sprint(show, m) == "LocalQuantileRegression(tau 0.25, degree 1, $(EpanechnikovKernel()), bandwidth $(round(m.h; sigdigits = 4)), 30 points, 150 observations)"
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = (0.3, 0.4))
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5,
                                 kernel = PiecewisePolynomialKernel([-1.0, 1.0], [[9/8, 0.0, -15/8]]))
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; tau = 1.0)
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; degree = 1, derivative = 2)
  @test_throws ArgumentError fit(LocalQuantileRegression, SmoothingData(X); tau = 0.5)
  @test_throws ArgumentError confint(m; level = 1.5)
  @test_throws ArgumentError select_bandwidth(YuJones(), d, EpanechnikovKernel(); tau = 0.0)
end

@testset "quantile conventional standard errors at a99bcc9" begin
  points = [-1.5, -0.5, 0.0, 0.7, 1.4]
  for (τ, h, p, ν, ref) in (
      (0.25, 0.5, 1, 0, [0.2514469273573268, 0.0876007640484024, 0.08493853473752128, 0.10724141770655232, 0.23063146375700266]),
      (0.5, 0.4, 1, 0, [0.29442388404994135, 0.10934543221195414, 0.08532069989335814, 0.11517667909259963, 0.3802786732296921]),
      (0.9, 0.6, 2, 1, [0.4301858532046626, 0.5057015763068441, 0.3725732516947525, 0.4145407389391761, 0.9908496348764847]))
    m = fit(LocalQuantileRegression, X, Y; tau = τ, bandwidth = h, degree = p, derivative = ν, points)
    @test stderror(m) ≈ ref rtol = 1e-12
    @test stderror(m; method = Conventional()) == stderror(m)
  end
end

# a dense reference using only the kernel and public quantile fits
function rq_reference(x, y, w, k, x0, h, b, p, ν, τ)
  u, v = (x .- x0) ./ h, (x .- x0) ./ b
  Zh = [t^j for t in u, j in 0:p]
  Zb = [t^j for t in v, j in 0:p+1]
  wh, wb = w .* k.(u), w .* k.(v)
  g = (Zh' * (wh .* Zh)) \ [j == ν ? 1.0 : 0.0 for j in 0:p]
  c = (Zb' * (wb .* Zb)) \ [j == p + 1 ? 1.0 : 0.0 for j in 0:p+1]
  Λ = Zh' * (wh .* u .^ (p + 1))
  ℓ = wh .* (Zh * g)
  ℓbc = ℓ .- (h / b)^(p + 1) * dot(g, Λ) .* wb .* (Zb * c)
  # Hall and Sheather's difference quotient, with Kish's effective window size
  neff = sum(wh)^2 / sum(abs2, wh)
  N = Distributions.Normal()
  z, qz = Distributions.quantile(N, 0.975), Distributions.quantile(N, τ)
  δ = neff^(-1 / 3) * z^(2 / 3) * (1.5 * Distributions.pdf(N, qz)^2 / (2 * qz^2 + 1))^(1 / 3)
  δ = min(δ, τ / 2, (1 - τ) / 2)
  hi = only(coef(fit(LocalQuantileRegression, x, y; weights = w, kernel = k, tau = τ + δ,
                    degree = p, derivative = 0, bandwidth = h, points = [x0])))
  lo = only(coef(fit(LocalQuantileRegression, x, y; weights = w, kernel = k, tau = τ - δ,
                    degree = p, derivative = 0, bandwidth = h, points = [x0])))
  s = (hi - lo) / (2δ)
  dpilot = only(coef(fit(LocalQuantileRegression, x, y; weights = w, kernel = k, tau = τ,
                        degree = p + 1, derivative = p + 1, bandwidth = b, points = [x0])))
  bias = factorial(ν) * h^(p + 1 - ν) * dot(g, Λ) * dpilot / factorial(p + 1)
  scale = factorial(ν) / h^ν
  return (se = scale * s * sqrt(τ * (1 - τ) * sum(abs2, ℓ)),
          se_rbc = scale * s * sqrt(τ * (1 - τ) * sum(abs2, ℓbc)), bias = bias)
end

@testset "quantile robust bias correction" begin
  # an interior and a boundary point, with non-uniform observation weights
  x = repeat(collect(range(-1.0, 1.0; length = 17)); inner = 5)
  errors = repeat([-1.0, -0.5, 0.0, 0.5, 1.0], 17)
  y = 0.3 .+ x .+ 0.2 .* x .^ 2 .- 0.1 .* x .^ 3 .+ 0.05 .* x .^ 4 .+ (0.5 .+ 0.1 .* x) .* errors
  w = repeat(1 .+ mod.(1:17, 4) ./ 3; inner = 5)
  points, h, τ = [-1.0, 0.15], 0.85, 0.5
  rbc = RobustBiasCorrected()
  for k in (EpanechnikovKernel(), GaussianKernel()), (p, ν) in ((1, 0), (2, 1), (3, 2)), ratio in (0.75, 1.0, 1.3)
    b = h * ratio
    m = fit(LocalQuantileRegression, x, y; weights = w, kernel = k, tau = τ,
            degree = p, derivative = ν, bandwidth = h, pilot = b, points)
    se, se_rbc = stderror(m), stderror(m; method = rbc)
    estimate, estimate_bc = coef(m), coef(m; method = rbc)
    @test all(isfinite, se) && all(>(0), se)
    @test all(isfinite, se_rbc) && all(>(0), se_rbc)
    for (j, x0) in enumerate(points)
      ref = rq_reference(x, y, w, k, x0, h, b, p, ν, τ)
      @test se[j] ≈ ref.se rtol = 1e-9
      @test se_rbc[j] ≈ ref.se_rbc rtol = 1e-9
      @test estimate_bc[j] ≈ estimate[j] - ref.bias rtol = 1e-9 atol = 1e-11
    end
  end
  # the order-2 pilot recovers the unscaled quadratic coefficient, without an
  # exactness claim for the bias-corrected local linear estimate
  xp = collect(range(-1.0, 1.0; length = 13))
  mp = fit(LocalQuantileRegression, xp, xp .^ 2; tau = 0.37, bandwidth = 2.0,
           pilot = 1.7, points = [-0.4, 0.0, 0.6])
  @test mp.pilot_coefficients ≈ hcat(mp.points .^ 2, 2 .* mp.points, ones(3)) atol = 1e-12
  # changing the units of x scales derivatives and their standard errors
  for (p, ν) in ((1, 0), (2, 1), (3, 2))
    m = fit(LocalQuantileRegression, x, y; weights = w, degree = p, derivative = ν,
            bandwidth = h, pilot = 1.1, points)
    c = 2.5
    mc = fit(LocalQuantileRegression, c .* x, y; weights = w, degree = p, derivative = ν,
             bandwidth = c * h, pilot = c * 1.1, points = c .* points)
    @test coef(mc; method = rbc) ≈ coef(m; method = rbc) ./ c^ν rtol = 1e-9 atol = 1e-11
    @test stderror(mc; method = rbc) ≈ stderror(m; method = rbc) ./ c^ν rtol = 1e-9
  end
  # multiplying all weights leaves every column unchanged; repeated rows are
  # tested above only for estimates, since the sandwich uses squared weights
  mw = fit(LocalQuantileRegression, X, Y; weights = 1 .+ mod.(1:150, 3), bandwidth = 0.6,
           pilot = 0.8, degree = 2, derivative = 1, points = [-0.5, 0.5])
  ms = fit(LocalQuantileRegression, X, Y; weights = 7 .* (1 .+ mod.(1:150, 3)), bandwidth = 0.6,
           pilot = 0.8, degree = 2, derivative = 1, points = [-0.5, 0.5])
  tw, ts = coeftable(mw), coeftable(ms)
  for name in keys(tw)
    @test getproperty(tw, name) ≈ getproperty(ts, name) rtol = 1e-9 atol = 1e-11
  end
end

@testset "quantile inference failures and API" begin
  rbc = RobustBiasCorrected()
  # the primary fit and its sparsity survive a pilot with too few distinct x
  m = fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = 1e-6, points = [0.0])
  @test all(isfinite, coef(m)) && all(isfinite, stderror(m)) && only(stderror(m)) > 0
  @test all(isnan, coef(m; method = rbc)) && all(isnan, stderror(m; method = rbc))
  @test all(isfinite, confint(m; method = Conventional())) && all(isnan, confint(m))
  # continuous covariates and constant responses identify both fits, but no sparsity
  md = fit(LocalQuantileRegression, X, zeros(length(X)); bandwidth = 0.5, points = [0.0])
  td = coeftable(md)
  @test all(isfinite, td.estimate) && all(isfinite, td.estimate_bc)
  @test all(isnan, td.se) && all(isnan, td.se_rbc)
  @test all(isnan, confint(md; method = Conventional())) && all(isnan, confint(md))
  # primary failure leaves all inference missing, even with a wide pilot
  mf = fit(LocalQuantileRegression, X, Y; bandwidth = 0.01, pilot = 20.0, points = [10.0])
  tf = coeftable(mf)
  @test all(isnan, vcat(tf.estimate, tf.se, tf.estimate_bc, tf.se_rbc))
  m = fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, points = [-0.5, 0.0, 0.5])
  @test m.b == m.h && size(m.pilot_coefficients) == (3, m.degree + 2)
  @test confint(m) == confint(m; method = rbc)
  @test confint(rearrange(m)) == confint(rearrange(m); method = rbc)
  for method in (Conventional(), rbc)
    z = Distributions.quantile(Distributions.Normal(), 0.95)
    estimate, se = coef(m; method), stderror(m; method)
    @test confint(m; level = 0.9, method) ≈ hcat(estimate .- z .* se, estimate .+ z .* se)
  end
  t = coeftable(m; level = 0.9)
  @test keys(t) == (:point, :h, :b, :estimate, :se, :lower, :upper, :estimate_bc, :se_rbc, :lower_rbc, :upper_rbc)
  @test all(col -> length(col) == length(m.points), values(t))
  @test t.point == m.points && t.h == fill(m.h, 3) && t.b == fill(m.b, 3)
  @test t.estimate == coef(m) && t.se == stderror(m)
  @test t.estimate_bc == coef(m; method = rbc) && t.se_rbc == stderror(m; method = rbc)
  @test hcat(t.lower, t.upper) == confint(m; level = 0.9, method = Conventional())
  @test hcat(t.lower_rbc, t.upper_rbc) == confint(m; level = 0.9)
  for method in (Honest(1.0), Honest(1.0; bias_corrected = true), :unsupported)
    @test_throws ArgumentError coef(m; method)
    @test_throws ArgumentError stderror(m; method)
    @test_throws ArgumentError confint(m; method)
  end
  for v in (0.0, -1.0, Inf, NaN)
    @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, rho = v, points = [0.0])
    @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = v, points = [0.0])
    @test fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = 0.7, rho = v, points = [0.0]).b == 0.7
  end
  @test fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, rho = 2, points = [0.0]).b == 0.25
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = [0.5], points = [0.0])
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = :selected, points = [0.0])
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, pilot = big"1e1000", points = [0.0])
  @test_throws ArgumentError fit(LocalQuantileRegression, X, Y; bandwidth = 0.5, rho = big"1e1000", points = [0.0])
  @test_throws ArgumentError coeftable(m; level = 0)
end

@testset "conditional density" begin
  # the local constant estimate of np, whose bandwidths are ordered (y, x)
  for (bw, ref) in (((0.3, 0.5), NPCDENS_03_05), ((0.6, 0.25), NPCDENS_06_025))
    m = fit(ConditionalDensity, X, Y; bandwidth = reverse(bw))
    @test [pdf(m, EY[k], EX[k]) for k in 1:4] ≈ ref rtol = 1e-12
  end
  m = fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3))
  @test nobs(m) == length(X) && bandwidth(m) == (0.4, 0.3)
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
  # the criterion is np's: its value at np's bandwidths is minus np's log-likelihood
  @test _cdcv(:likelihood, d, GaussianKernel(), GaussianKernel(), CVML...) * 150 ≈ -CVML_FVAL[1] rtol = 1e-10
  @test bandwidth(fit(ConditionalDensity, X, Y)) == Tuple(s.h)
  # least squares: our criterion at our minimum is below its value at np's
  # bandwidths, whose own criterion differs
  # the least-squares criterion by quadrature on a small sample
  ds = SmoothingData(X[1:30], Y[1:30])
  G = GaussianKernel()
  h, b = 0.5, 0.4
  function fminus(i, t)
    ws = [j == i ? 0.0 : G((X[j] - X[i]) / h) for j in 1:30]
    return sum(ws[j] * G((t - Y[j]) / b) / b for j in 1:30) / sum(ws)
  end
  brute = sum(quadgk(t -> fminus(i, t)^2, -Inf, Inf; rtol = 1e-11)[1] - 2 * fminus(i, Y[i]) for i in 1:30) / 30
  @test _cdcv(:leastsquares, ds, G, G, h, b) ≈ brute rtol = 1e-9
  # np's least-squares criterion differs from this one by about 0.6% at its bandwidths
  @test 0.003 < abs(_cdcv(:leastsquares, d, G, G, CVLS...) + CVLS_FVAL[1]) / CVLS_FVAL[1] < 0.01
  # an isolated covariate does not set a floor on h
  me2 = fit(ConditionalDensity, vcat(X, 40.0), vcat(Y, 0.0); kernel = EpanechnikovKernel())
  @test bandwidth(me2)[1] < 5
  sl = select_bandwidth(ConditionalCV(criterion = :leastsquares), d, GaussianKernel())
  @test _cdcv(:leastsquares, d, GaussianKernel(), GaussianKernel(), sl.h...) <=
        _cdcv(:leastsquares, d, GaussianKernel(), GaussianKernel(), CVLS...)
  # compact kernels, and weights
  me = fit(ConditionalDensity, X, Y; kernel = EpanechnikovKernel(), weights = 1 .+ mod.(1:150, 3))
  @test all(isfinite, bandwidth(me))
  @test quadgk(t -> pdf(me, t, 0.0), -10, 10; maxevals = 10^6)[1] ≈ 1 rtol = 1e-6
  @test isnan(pdf(fit(ConditionalDensity, X, Y; bandwidth = (0.1, 0.3), kernel = EpanechnikovKernel()), 0.0, 9.0))
  @test sprint(show, m) == "ConditionalDensity(degree 0, $(GaussianKernel()), bandwidth (0.4, 0.3), 150 observations)"
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3),
                                 kernel = PiecewisePolynomialKernel([-1.0, 1.0], [[9/8, 0.0, -15/8]]))
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; bandwidth = 0.3)
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; degree = 1)
  @test_throws ArgumentError fit(ConditionalDensity, X, Y; bandwidth = (0.4, 0.3), ykernel = derivative(GaussianKernel(), 1))
  @test_throws ArgumentError ConditionalCV(criterion = :aic)
  @test_throws ArgumentError fit(ConditionalDensity, SmoothingData(X); bandwidth = (0.4, 0.3))
end

end
