module SmootherTest

using Test
using KernelSmoothing

const XS = [mod(0.7548776662466927i, 1.0) for i in 1:500]
const YS = cos.(3 .* XS) .+ 0.2 .* sin.(11.1 .* (1:500))
const WS = 1.0 .+ mod.(1:500, 3)
const PTS = [0.0, 0.1, 0.37, 0.5, 0.93, 1.0]

@testset "local polynomial estimates" begin
  for k in (EpanechnikovKernel(), TriangularKernel(), GaussianKernel()), (p, ν) in ((0, 0), (1, 0), (2, 1), (3, 2))
    for w in (nothing, WS)
      m = fit(LocalPolynomialRegression, XS, YS; weights = w, degree = p, derivative = ν, kernel = k,
              bandwidth = 0.15, points = PTS, minobs = 0, vce = HCVariance(0))
      e = local_polynomial(XS, YS, PTS; weights = w, degree = p, derivative = ν, kernel = k, bandwidth = 0.15)
      @test e ≈ m.estimate rtol = 1e-9
      # the weights give the same estimates, in the input order of the observations
      L = smoother_matrix(XS, PTS; weights = w, degree = p, derivative = ν, kernel = k, bandwidth = 0.15)
      @test L * YS ≈ e rtol = 1e-9
      # and reproduce polynomials up to the order of the fit
      for j in 0:p
        target = j < ν ? zeros(length(PTS)) : factorial(j) / factorial(j - ν) .* PTS .^ (j - ν)
        @test L * XS .^ j ≈ target atol = 1e-8
      end
    end
  end
  # a bandwidth per point, and the default degree derivative + 1
  @test local_polynomial(XS, YS, PTS[1:2]; bandwidth = [0.1, 0.2]) ≈
        [local_polynomial(XS, YS, PTS[1:1]; bandwidth = 0.1)[1], local_polynomial(XS, YS, PTS[2:2]; bandwidth = 0.2)[1]]
  @test local_polynomial(XS, YS, PTS; derivative = 1, bandwidth = 0.2) ≈
        local_polynomial(XS, YS, PTS; derivative = 1, degree = 2, bandwidth = 0.2)
  # SmoothingData, whose columns follow the sorted order of x
  d = SmoothingData(XS, YS)
  @test local_polynomial(d, PTS; bandwidth = 0.15) == local_polynomial(XS, YS, PTS; bandwidth = 0.15)
  @test smoother_matrix(d, PTS; bandwidth = 0.15) * d.y ≈ local_polynomial(d, PTS; bandwidth = 0.15)
  # observations with zero weight get zero columns
  w0 = copy(WS); w0[1:10] .= 0
  L0 = smoother_matrix(XS, PTS; weights = w0, bandwidth = 0.15)
  @test all(iszero, L0[:, 1:10]) && L0 * YS ≈ local_polynomial(XS, YS, PTS; weights = w0, bandwidth = 0.15)
  # a window with too few distinct values
  @test isnan(local_polynomial(XS, YS, [2.0]; bandwidth = 0.1)[1])
  @test all(isnan, smoother_matrix(XS, [2.0]; bandwidth = 0.1))
  @inferred local_polynomial(XS, YS, PTS; bandwidth = 0.15)
  @inferred smoother_matrix(XS, PTS; bandwidth = 0.15)
end

@testset "errors" begin
  @test_throws ArgumentError local_polynomial(SmoothingData(XS), PTS; bandwidth = 0.1)
  @test_throws ArgumentError local_polynomial(XS, YS, PTS; bandwidth = -0.1)
  @test_throws ArgumentError local_polynomial(XS, YS, PTS; bandwidth = 0.1, degree = 1, derivative = 2)
  @test_throws DimensionMismatch smoother_matrix(XS, PTS; bandwidth = [0.1, 0.2])
  @test_throws UndefKeywordError local_polynomial(XS, YS, PTS)
end

end
