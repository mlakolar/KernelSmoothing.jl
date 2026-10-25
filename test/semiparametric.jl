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

@testset "partially linear" begin
  Yp = sin.(2π .* U) .+ 1.5 .* X2 .- 0.5 .* X3 .+ E
  X = hcat(X2, X3)
  for w in (nothing, W), vce in (HCVariance(0), HCVariance(1))
    h = 0.15
    m = fit(PartiallyLinear, U, X, Yp; bandwidth = h, points = PTS, weights = w, vce)
    # Robinson's estimator with the smoother matrix of the observations
    S = smoother_matrix(U, U; bandwidth = h, weights = w)
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
    @test m.smooth ≈ local_polynomial(U, Yp .- X * β, PTS; bandwidth = h, weights = w) rtol = 1e-8
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
  @test_throws DimensionMismatch fit(PartiallyLinear, U, X2[1:10], Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(PartiallyLinear, U, hcat(ones(N), X2), Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.2, vce = HCVariance(3))
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.001)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = MSEOptimal())
end

end
