module SemiparametricTest

using Test
using KernelSmoothing
using LinearAlgebra: Diagonal, I, diag

const N = 400
const U = [mod(0.7548776662466927i, 1.0) for i in 1:N]
const X2 = sin.(3.1 .* (1:N)) .+ 0.5 .* cos.(1.7 .* (1:N))
const X3 = cos.(2.3 .* (1:N))
const E = 0.2 .* sin.(17.3 .* (1:N))
const Y = sin.(2π .* U) .+ (1 .+ U .^ 2) .* X2 .+ E
const W = 1.0 .+ mod.(1:N, 3)
const PTS = [0.0, 0.1, 0.5, 0.9, 1.0]
const K = EpanechnikovKernel()

@testset "varying coefficients" begin
  X = hcat(ones(N), X2)
  for p in (0, 1, 2), w in (nothing, W), vce in (HCVariance(0), HCVariance(1))
    h = 0.2
    m = fit(VaryingCoefficient, U, X, Y; bandwidth = h, points = PTS, degree = p, weights = w, vce)
    @test size(coef(m)) == (5, 2) && size(stderror(m)) == (5, 2)
    ww = w === nothing ? ones(N) : w
    for (j, u0) in enumerate(PTS)
      kw = ww .* K.((U .- u0) ./ h)
      Z = hcat([X[:, c] .* ((U .- u0) ./ h) .^ l for c in 1:2 for l in 0:p]...)
      A = Z' * Diagonal(kw) * Z
      θ = A \ (Z' * (kw .* Y))
      e = Y .- Z * θ
      meat = Z' * Diagonal((kw .* e) .^ 2) * Z
      m_w = count(>(0), kw)
      vce.type == 1 && (meat .*= m_w / (m_w - 2(p + 1)))
      V = A \ meat / A
      idx = [1, p + 2]
      @test coef(m)[j, :] ≈ θ[idx] rtol = 1e-8
      @test stderror(m)[j, :] ≈ sqrt.(diag(V)[idx]) rtol = 1e-8
    end
  end
  # with one column of ones, the fits of LocalPolynomialRegression
  m1 = fit(VaryingCoefficient, U, ones(N), Y; bandwidth = 0.15, points = PTS)
  l = fit(LocalPolynomialRegression, U, Y; bandwidth = 0.15, points = PTS, vce = HCVariance(0), minobs = 0)
  @test vec(coef(m1)) ≈ l.estimate rtol = 1e-10
  @test vec(stderror(m1)) ≈ l.se rtol = 1e-10
  ci = confint(m1)
  @test ci.lower ≈ coef(m1) .- 1.959963984540054 .* stderror(m1)
  # the leave-one-out criterion, against fits without the observation
  d, Xs = KernelSmoothing._semidata(U, hcat(ones(N), X2), Y, nothing)
  cv = 0.0
  for i in 1:N
    keep = setdiff(1:N, i)
    kw = K.((d.x[keep] .- d.x[i]) ./ 0.2)
    Z = hcat(Xs[keep, 1], Xs[keep, 1] .* (d.x[keep] .- d.x[i]), Xs[keep, 2], Xs[keep, 2] .* (d.x[keep] .- d.x[i]))
    θ = (Z' * Diagonal(kw) * Z) \ (Z' * (kw .* d.y[keep]))
    cv += (d.y[i] - θ[1] * Xs[i, 1] - θ[3] * Xs[i, 2])^2
  end
  @test KernelSmoothing._vccriterion(d, Xs, K, 0.2, 1) ≈ cv / N rtol = 1e-10
  mc = fit(VaryingCoefficient, U, hcat(ones(N), X2), Y; points = PTS)
  @test mc.selection.rule == CrossValidation() && 0.02 < mc.h < 0.5
  @test mc.selection.details.criterion[argmin(mc.selection.details.criterion)] >= KernelSmoothing._vccriterion(d, Xs, K, mc.h, 1) - 1e-12
  @test nobs(m1) == N && bandwidth(m1) == 0.15
  @test startswith(sprint(show, m1), "VaryingCoefficient(1 coefficients")
  # a window without enough observations
  @test all(isnan, coef(fit(VaryingCoefficient, U, X, Y; bandwidth = 0.1, points = [3.0])))
  @inferred fit(VaryingCoefficient, U, X, Y; bandwidth = 0.2, points = PTS)
end

@testset "partially linear" begin
  Yp = sin.(2π .* U) .+ 1.5 .* X2 .- 0.5 .* X3 .+ E
  X = hcat(X2, X3)
  for w in (nothing, W), vce in (HCVariance(0), HCVariance(1))
    h = 0.15
    m = fit(PartiallyLinear, U, X, Yp; bandwidth = h, points = PTS, weights = w, vce)
    # Robinson's estimator with the smoother matrix of the observations
    S = smoothermatrix(U, U; bandwidth = h, weights = w)
    X̃ = (I - S) * X
    ỹ = (I - S) * Yp
    ww = w === nothing ? ones(N) : w
    G = X̃' * Diagonal(ww) * X̃
    β = G \ (X̃' * (ww .* ỹ))
    e = ỹ .- X̃ * β
    meat = X̃' * Diagonal((ww .* e) .^ 2) * X̃
    vce.type == 1 && (meat .*= N / (N - 2))
    @test coef(m) ≈ β rtol = 1e-8
    @test vcov(m) ≈ G \ meat / G rtol = 1e-8
    @test m.smooth ≈ localpolynomial(U, Yp .- X * β, PTS; bandwidth = h, weights = w) rtol = 1e-8
  end
  m = fit(PartiallyLinear, U, X, Yp)
  @test m.selection.rule == CrossValidation()
  @test nobs(m) == N && bandwidth(m) == m.selection.h
  @test coef(m) ≈ [1.5, -0.5] atol = 0.05
  t = coeftable(m)
  @test t.estimate == coef(m) && t.se == stderror(m) && t.upper ≈ confint(m)[:, 2]
  @test startswith(sprint(show, m), "PartiallyLinear(2 coefficients")
  @inferred fit(PartiallyLinear, U, X, Yp; bandwidth = 0.15)
end

@testset "errors" begin
  @test_throws DimensionMismatch fit(VaryingCoefficient, U, X2[1:10], Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(VaryingCoefficient, U, X2, Y; bandwidth = 0.2, vce = HCVariance(2))
  @test_throws ArgumentError fit(VaryingCoefficient, U, X2, Y; bandwidth = MSEOptimal())
  @test_throws ArgumentError fit(VaryingCoefficient, U, X2, Y; bandwidth = 0.2, degree = -1)
  @test_throws ArgumentError fit(PartiallyLinear, U, hcat(ones(N), X2), Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.2, vce = HCVariance(3))
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.001)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = MSEOptimal())
end

end
