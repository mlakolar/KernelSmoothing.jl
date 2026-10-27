module MultivariateKDETest

using Test
using KernelSmoothing
using KernelSmoothing: _mvdata, _psihat, _multiindices, _psinormal, _psitensor, _picriterion, _lscvcriterion,
  _scvcriterion, _minimizematrix, _pretransform, _mvcov
using Distributions: Distributions, MvNormal, logpdf, insupport
using LinearAlgebra
using ForwardDiff
using QuadGK
using Random
using Statistics

include(joinpath(@__DIR__, "data", "ks_golden.jl"))

const XS = Dict(2 => X2, 3 => X3)
rel(a, b) = norm(a .- b) / norm(b)

# the estimate by its definition
bf_pdf(x, w, H, t) = sum(w[i] * Distributions.pdf(MvNormal(x[:, i], H), t) for i in axes(x, 2)) / sum(w)

@testset "MultivariateKDE" begin
  x = X2
  n = size(x, 2)
  H = [0.3 0.1; 0.1 0.2]
  w = 1.0 .+ mod.(1:n, 3)
  f = fit(MultivariateKDE, x; bandwidth = H)
  fw = fit(MultivariateKDE, x; bandwidth = H, weights = w)
  t = [0.3, -0.2]
  @test pdf(f, t) ≈ bf_pdf(x, ones(n), H, t) rtol = 1e-12
  @test pdf(fw, t) ≈ bf_pdf(x, w, H, t) rtol = 1e-12
  @test logpdf(f, t) ≈ log(bf_pdf(x, ones(n), H, t)) rtol = 1e-12
  @test length(f) == 2 && nobs(f) == n && bandwidth(f) == H
  # a tuple of coordinates is the same sample
  @test pdf(fit(MultivariateKDE, (x[1, :], x[2, :]); bandwidth = H), t) == pdf(f, t)
  # many points at once, also on a grid
  P = [t [1.0, 2.0] [-3.0, 0.5]]
  @test pdf(f, P) ≈ [pdf(f, P[:, j]) for j in 1:3] rtol = 1e-14
  g1, g2 = -1.0:0.5:1.0, [-0.5, 0.0, 2.0]
  Z = pdf(f, (g1, g2))
  @test size(Z) == (5, 3)
  @test Z[2, 3] ≈ pdf(f, [g1[2], g2[3]]) rtol = 1e-14
  # far from the data the density underflows, its logarithm does not
  far = [60.0, -40.0]
  @test pdf(f, far) == 0
  @test isfinite(logpdf(f, far))
  @test logpdf(f, far) ≈ log(sum(exp(big(logpdf(MvNormal(x[:, i], H), far))) for i in 1:n) / n) rtol = 1e-10
  @test logpdf(f, [far far]) ≈ [logpdf(f, far), logpdf(f, far)]
  @test isnan(logpdf(f, [NaN, 0.0]))
  # infinite points, and points whose distance overflows
  @test pdf(f, [Inf, 0.0]) == 0 && logpdf(f, [0.0, -Inf]) == -Inf
  @test logpdf(f, [1e200, 0.0]) == -Inf
  @test logpdf(f, [far [Inf, 0.0] [NaN, 1.0] t]) ≈ [logpdf(f, far), -Inf, NaN, logpdf(f, t)] nans = true
  @test logpdf(fit(MultivariateKDE, x; bandwidth = [0.5, 0.2]), [Inf, 0.0]) == -Inf   # diagonal H
  # in place, and integer points
  out = zeros(3)
  Distributions.logpdf!(out, f, P)
  @test out ≈ logpdf(f, P)
  @test pdf(f, [1 0; 2 -1]) ≈ [pdf(f, [1.0, 2.0]), pdf(f, [0.0, -1.0])]
  @test_throws DimensionMismatch pdf(f, ones(3, 2))
  # dual numbers go through the scalar method
  gr = ForwardDiff.gradient(p -> logpdf(f, p), t)
  num = [(logpdf(f, t .+ e) - logpdf(f, t .- e)) / 2e-6 for e in ([1e-6, 0.0], [0.0, 1e-6])]
  @test gr ≈ num rtol = 1e-6
  @test ForwardDiff.gradient(p -> sum(Distributions.logpdf(f, reshape(p, 2, 1))), t) ≈ gr   # a matrix of duals
  # bandwidths as a number, as standard deviations and as a matrix
  @test bandwidth(fit(MultivariateKDE, x; bandwidth = 0.5)) == [0.25 0; 0 0.25]
  @test bandwidth(fit(MultivariateKDE, x; bandwidth = [0.5, 0.2])) ≈ [0.25 0; 0 0.04]
  @test bandwidth(fit(MultivariateKDE, x; bandwidth = (0.5, 0.2))) == bandwidth(fit(MultivariateKDE, x; bandwidth = [0.5, 0.2]))
  # in one dimension, the estimate with the Gaussian kernel
  @test pdf(fit(MultivariateKDE, x[1:1, :]; bandwidth = 0.4), [0.3]) ≈
        pdf(fit(KDE, x[1, :]; kernel = GaussianKernel(), bandwidth = 0.4), 0.3) rtol = 1e-13
  # a sample with more coordinates than observations is probably transposed
  @test_logs (:warn, r"more coordinates") fit(MultivariateKDE, permutedims(x[:, 1:3]); bandwidth = 1.0)
  @test_throws ArgumentError fit(KDE, x[1, :]; bandwidth = PlugInMatrix())
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = [1.0 2.0; 2.0 1.0])   # not positive definite
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = [1.0 0.1; 0.2 1.0])   # not symmetric
  @test_throws DimensionMismatch fit(MultivariateKDE, x; bandwidth = Matrix(1.0I, 3, 3))
  @test_throws DimensionMismatch fit(MultivariateKDE, x; bandwidth = [1.0])
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = -1.0)
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = "wide")
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = SheatherJones())
  @test_throws ArgumentError fit(MultivariateKDE, [1.0 NaN; 0.0 1.0]; bandwidth = 1.0)
  @test_throws ArgumentError fit(MultivariateKDE, x; bandwidth = H, weights = -w)
  @test_throws DimensionMismatch fit(MultivariateKDE, x; bandwidth = H, weights = w[1:5])
  @test_throws DimensionMismatch fit(MultivariateKDE, ([1.0, 2.0], [1.0]); bandwidth = 1.0)
  @test_throws ArgumentError fit(MultivariateKDE, ())
  @test sprint(show, fw) == "MultivariateKDE(2 dimensions, 200 observations, weighted)"
end

@testset "MultivariateKDE as a distribution" begin
  x = X2[:, 1:40]
  H = [0.3 0.1; 0.1 0.2]
  w = 1.0 .+ mod.(1:40, 3)
  for f in (fit(MultivariateKDE, x; bandwidth = H), fit(MultivariateKDE, x; bandwidth = H, weights = w))
    # integrates to one: the inner integral over the second coordinate is the marginal
    inner(s) = quadgk(u -> pdf(f, [s, u]), -Inf, Inf; rtol = 1e-10)[1]
    @test quadgk(inner, -Inf, Inf; rtol = 1e-8)[1] ≈ 1 rtol = 1e-7
    m1 = marginal(f, 1)
    @test m1 isa KDE && bandwidth(m1) ≈ sqrt(H[1, 1])
    @test pdf(m1, 0.4) ≈ inner(0.4) rtol = 1e-8
    @test pdf(marginal(f, 2), -0.3) ≈ quadgk(s -> pdf(f, [s, -0.3]), -Inf, Inf; rtol = 1e-10)[1] rtol = 1e-8
    @test_throws ArgumentError marginal(f, 3)
    # moments: the weighted mean, and the empirical covariance plus H
    ww = f.data.w === nothing ? ones(40) : w
    μ = x * ww / sum(ww)
    @test mean(f) ≈ μ
    C = (x .- μ) * Diagonal(ww) * (x .- μ)' / sum(ww)
    @test cov(f) ≈ C + H
    @test var(f) ≈ diag(C + H)
    # the smoothed bootstrap
    r = rand(Xoshiro(1), f, 200_000)
    @test size(r) == (2, 200_000)
    @test vec(mean(r; dims = 2)) ≈ μ atol = 0.01
    @test cov(r; dims = 2) ≈ C + H atol = 0.02
    @test rand(Xoshiro(3), f) == rand(Xoshiro(3), f)
    @test insupport(f, [1e3, -1e3]) && !insupport(f, [Inf, 0.0]) && !insupport(f, [1.0, 2.0, 3.0])
  end
  # observations of zero weight are dropped: never drawn, and no NaN near them
  fz = fit(MultivariateKDE, [0.0 100.0; 0.0 100.0]; bandwidth = 0.1, weights = [1.0, 0.0])
  @test nobs(fz) == 1
  @test all(<(5), rand(Xoshiro(2), fz, 1000))
  @test pdf(fz, [100.0, 100.0]) == 0 && isfinite(logpdf(fz, [100.0, 100.0]))
end

@testset "agreement with ks" begin
  for d in (2, 3)
    x = XS[d]
    n = size(x, 2)
    ev = d == 2 ? EVAL2 : EVAL3
    Hns = d == 2 ? HNS2 : HNS3
    # the estimate, without and with weights
    w = 1.0 .+ mod.(1:n, 3)
    @test pdf(fit(MultivariateKDE, x; bandwidth = Hns), ev) ≈ (d == 2 ? KDE2 : KDE3) rtol = 1e-12
    @test pdf(fit(MultivariateKDE, x; bandwidth = Hns, weights = w), ev) ≈ (d == 2 ? KDEW2 : KDEW3) rtol = 1e-12
    @test bandwidth(select_bandwidth(NormalScaleMatrix(), x)) ≈ Hns rtol = 1e-13
    # the functionals of order 4
    R = _multiindices(d, 4)
    ψ = _psihat(x, nothing, Float64(n), Float64(n), sqrt(0.3), R)
    ind = Int.(d == 2 ? KFE4IND2 : KFE4IND3)
    @test [ψ[ind[:, k]] for k in axes(ind, 2)] ≈ (d == 2 ? KFE4_2 : KFE4_3) rtol = 1e-12
    # the criteria
    data = _mvdata(x, nothing)
    A, y, _ = _pretransform(data, :sphere)
    G = 0.3 .* Matrix(1.0I, d, d)
    s = KernelSmoothing._gausspairs(y, nothing, Float64(n), [2 .* G], [false], false)[1]
    const0 = (2π)^(-d / 2) / sqrt(det(2 .* G)) / n + 2 * s[1] / n^2
    @test _scvcriterion(y, nothing, Float64(n), Float64(n), 0.2 .* Matrix(1.0I, d, d), G, const0)[1] ≈ (d == 2 ? SCVAT2 : SCVAT3)[1] rtol = 1e-12
    @test _lscvcriterion(x, nothing, Float64(n), Float64(n), Hns)[1] ≈ (d == 2 ? LSCVAT2 : LSCVAT3)[1] rtol = 1e-12
    # the plug-in matrices: ks stops its minimization early, at a matrix whose criterion
    # is above the minimum by a fraction of its tolerance
    for (stages, ref) in ((1, d == 2 ? HPI1_2 : HPI1_3), (2, d == 2 ? HPI2_2 : HPI2_3))
      sel = select_bandwidth(PlugInMatrix(; stages, compat = true, binned = false), x)
      @test rel(bandwidth(sel), ref) < 5e-4
      @test sel.details.converged
    end
    @test rel(bandwidth(select_bandwidth(PlugInMatrix(; diagonal = true, compat = true, binned = false), x)), d == 2 ? HPIDIAG2 : HPIDIAG3) < 2e-4
    @test rel(bandwidth(select_bandwidth(LSCVMatrix(; diagonal = true, binned = false), x)), d == 2 ? HLSCVDIAG2 : HLSCVDIAG3) < 2e-3
  end
  @test rel(bandwidth(select_bandwidth(NormalScaleMatrix(diagonal = true), X2)), HNSDIAG2) < 1e-7
  # LSCV in two dimensions: our minimum is at least as low as ks's
  sel = select_bandwidth(LSCVMatrix(binned = false), X2)
  @test rel(bandwidth(sel), HLSCV2) < 2e-3
  @test sel.details.criterion <= LSCVMIN2[1] + 1e-12
  # in three dimensions ks stops its minimization at a higher criterion
  @test select_bandwidth(LSCVMatrix(binned = false), X3).details.criterion < LSCVMIN3[1]
  # the smoothed cross-validation of ks uses a different pilot
  @test rel(bandwidth(select_bandwidth(SCVMatrix(binned = false), X2)), HSCV2) < 0.05
end

@testset "binned bandwidth matrix selections" begin
  for d in (2, 3), diagonal in (false, true), R in (PlugInMatrix, SCVMatrix, LSCVMatrix)
    @testset "$R, d = $d, diagonal = $diagonal" begin
      if d == 3 && !diagonal && R == LSCVMatrix
        @test_logs (:warn, r"coarse") match_mode = :any select_bandwidth(LSCVMatrix(binned = true), XS[d])
        continue
      end
      exact = select_bandwidth(R(; diagonal, binned = false), XS[d])
      # Three-dimensional binning is opt-in; the small golden sample needs a
      # finer grid here to test selection accuracy.
      gridsize = d == 3 ? 61 : nothing
      binned = select_bandwidth(R(; diagonal, binned = true, gridsize), XS[d])
      λ = eigvals(Symmetric(bandwidth(binned)), Symmetric(bandwidth(exact)))
      @test maximum(abs, λ .- 1) <= (d == 2 ? 0.01 : R == LSCVMatrix ? 0.05 : 0.02)
      @test exact.details.converged && binned.details.converged
    end
  end
end

@testset "bandwidth matrix selectors" begin
  x = X2
  n = size(x, 2)
  for rule in (NormalScaleMatrix(), PlugInMatrix(binned = false), SCVMatrix(binned = false), LSCVMatrix(binned = false),
               NormalScaleMatrix(diagonal = true), PlugInMatrix(diagonal = true, binned = false),
               SCVMatrix(diagonal = true, stages = 1, binned = false), LSCVMatrix(diagonal = true, binned = false))
    H = bandwidth(select_bandwidth(rule, x))
    @test isposdef(H) && H == H'
    rule.diagonal && @test isdiag(H)
    # scaling the data scales the matrix, up to the resolution of the minimization
    @test bandwidth(select_bandwidth(rule, 3 .* x)) ≈ 9 .* H rtol = 1e-5
    # weights near the ends of the floating-point range keep their ratios
  if rule isa PlugInMatrix
    for c in (1e-200, 1e200)
      @test bandwidth(select_bandwidth(rule, x; weights = c .* (1.0 .+ mod.(1:n, 4)))) ≈
            bandwidth(select_bandwidth(rule, x; weights = 1.0 .+ mod.(1:n, 4))) rtol = 1e-5
    end
  end
  # the weights matter only through their ratios
    w = 1.0 .+ mod.(1:n, 4)
    @test bandwidth(select_bandwidth(rule, x; weights = 2.5 .* w)) ≈ bandwidth(select_bandwidth(rule, x; weights = w)) rtol = 1e-5
    # a fit takes the rule
    f = fit(MultivariateKDE, x; bandwidth = rule)
    @test bandwidth(f) == H && f.selection.rule == rule
  end
  # the default is the plug-in rule
  @test bandwidth(fit(MultivariateKDE, x)) == bandwidth(select_bandwidth(PlugInMatrix(), x))
  @test bandwidth(fit(MultivariateKDE, x; bandwidth = nothing)) == bandwidth(fit(MultivariateKDE, x))
  # compat changes only the two-stage rule
  @test bandwidth(select_bandwidth(PlugInMatrix(stages = 1, compat = true), x)) ==
        bandwidth(select_bandwidth(PlugInMatrix(stages = 1), x))
  @test bandwidth(select_bandwidth(PlugInMatrix(compat = true), x)) != bandwidth(select_bandwidth(PlugInMatrix(), x))
  # in one dimension the normal scale rule is the normal reference rule of the Gaussian kernel
  @test sqrt(only(bandwidth(select_bandwidth(NormalScaleMatrix(), reshape(x[1, :], 1, :))))) ≈
        bandwidth(NormalReference(robust = false), SmoothingData(x[1, :]), GaussianKernel()) rtol = 1e-12
  # the plug-in and cross-validation rules work in one dimension too
  @test isposdef(bandwidth(select_bandwidth(PlugInMatrix(), reshape(x[1, :], 1, :))))
  # the rules print as their keyword constructors
  @test sprint(show, PlugInMatrix(stages = 1)) == "PlugInMatrix(stages = 1)"
  @test sprint(show, LSCVMatrix(diagonal = true)) == "LSCVMatrix(diagonal = true)"
  @test_throws ArgumentError PlugInMatrix(stages = 3)
  @test_throws ArgumentError SCVMatrix(pre = :rotate)
  @test_throws ArgumentError PlugInMatrix(diagonal = true, pre = :sphere)
  # singular data: the columns are the observations
  @test_throws ArgumentError select_bandwidth(PlugInMatrix(), permutedims(x[:, 1:5]))
  @test_throws ArgumentError select_bandwidth(LSCVMatrix(), x[:, 1:1])
  @test_throws ArgumentError select_bandwidth(LSCVMatrix(), x[:, 1:3]; weights = [1.0, 0.0, 0.0])
  # nearly collinear coordinates
  @test_throws ArgumentError select_bandwidth(PlugInMatrix(), [x; x[1:1, :] .+ x[2:2, :]])
end

@testset "criteria, gradients and the minimization" begin
  for d in (2, 3)
    x = XS[d]
    n = size(x, 2)
    w = 1.0 .+ mod.(1:n, 3)
    W = sum(w)
    neff = W^2 / sum(abs2, w)
    H = [0.3 0.05 0.02; 0.05 0.25 -0.03; 0.02 -0.03 0.35][1:d, 1:d]
    G = 0.2 .* Matrix(1.0I, d, d)
    φ(S, u) = Distributions.pdf(MvNormal(zeros(d), S), u)
    # the weighted criteria by their definitions
    lscv = sum(w[i] * w[j] * φ(2H, x[:, i] - x[:, j]) for i in 1:n, j in 1:n) / W^2 -
           2 * sum(w[i] * w[j] * φ(H, x[:, i] - x[:, j]) / (W * (W - w[i])) for i in 1:n, j in 1:n if i != j)
    @test _lscvcriterion(x, w, W, neff, H)[1] ≈ lscv rtol = 1e-11
    scv = (4π)^(-d / 2) / sqrt(det(H)) / neff +
          sum(w[i] * w[j] * (φ(2H + 2G, x[:, i] - x[:, j]) - 2φ(H + 2G, x[:, i] - x[:, j]) + φ(2G, x[:, i] - x[:, j]))
              for i in 1:n, j in 1:n) / W^2
    s0 = KernelSmoothing._gausspairs(x, w, W, [2 .* G], [false], false)[1][1]
    const0 = (2π)^(-d / 2) / sqrt(det(2 .* G)) / neff + 2 * s0 / W^2
    @test _scvcriterion(x, w, W, neff, H, G, const0)[1] ≈ scv rtol = 1e-11
    R = _multiindices(d, 4)
    ψ = _psihat(x, w, W, neff, 0.7, R)
    r = R[:, 2]
    # one functional by its definition, with the Hermite polynomials written out
    hermite4(u, k) = k == 4 ? u^4 - 6u^2 + 3 : k == 3 ? u^3 - 3u : k == 2 ? u^2 - 1 : k == 1 ? u : 1.0
    brute = sum(w[i] * w[j] * prod(hermite4((x[a, i] - x[a, j]) / 0.7, r[a]) for a in 1:d) *
                exp(-sum(abs2, x[:, i] - x[:, j]) / 0.98) for i in 1:n, j in 1:n) * (2π)^(-d / 2) * 0.7^(-d - 4) / W^2
    @test ψ[r] ≈ brute rtol = 1e-11
    # analytic gradients against central differences
    data = _mvdata(x, w)
    T = _psitensor(_psinormal(_mvcov(data), R), d)
    for fg in (h -> _lscvcriterion(x, w, W, neff, h), h -> _scvcriterion(x, w, W, neff, h, G, 0.0),
               h -> _picriterion(h, T, (4π)^(-d / 2) / neff))
      _, Gr = fg(H)
      for a in 1:d, b in 1:a
        E = zeros(d, d); E[a, b] = E[b, a] = 1
        num = (fg(H .+ 1e-6 .* E)[1] - fg(H .- 1e-6 .* E)[1]) / 2e-6
        @test (a == b ? Gr[a, b] : 2Gr[a, b]) ≈ num rtol = 1e-5 atol = 1e-12
      end
    end
    # the minimizer of the normal-reference criterion is the normal scale matrix
    S = _mvcov(data)
    Hns = (4 / ((d + 2) * neff))^(2 / (d + 4)) .* S
    Hmin, _, _, converged = _minimizematrix(h -> _picriterion(h, T, (4π)^(-d / 2) / neff), 1.3 .* Hns, false)
    @test converged
    @test Hmin ≈ Hns rtol = 1e-5
  end
  # a trial matrix that does not factorize is rejected, and the search restarts from
  # the gradient before it stops
  fgbad(H) = H[1, 1] > 2.5 ? throw(LinearAlgebra.PosDefException(1)) : ((H[1, 1] - 3)^2, fill(2 * (H[1, 1] - 3), 1, 1))
  Hb, _, _, convb = _minimizematrix(fgbad, fill(1.0, 1, 1), true)
  @test convb && 2.4 < Hb[1, 1] <= 2.5
  @test_throws ArgumentError _minimizematrix(H -> throw(DomainError(H)), fill(1.0, 1, 1), true)
  # at a kink the quasi-Newton step stops lowering the criterion: the search restarts
  # from the gradient once, then stops at the kink
  Hk, _, _, convk = _minimizematrix(H -> (abs(H[1, 1] - 2), fill(sign(H[1, 1] - 2), 1, 1)), fill(0.5, 1, 1), true)
  @test convk
  @test Hk[1, 1] ≈ 2 rtol = 1e-6
  # a vanishing gradient stops the search where it starts
  H1, _, it1, conv1 = _minimizematrix(H -> (1.0, zeros(1, 1)), fill(0.7, 1, 1), true)
  @test conv1 && it1 == 1 && H1 ≈ fill(0.7, 1, 1)
end

end
