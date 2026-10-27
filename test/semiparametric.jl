module SemiparametricTest

using Test
using KernelSmoothing
using LinearAlgebra: Diagonal, I

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
  @testset "smoother matrix oracle" begin
    for w in (nothing, W), h in (0.15, [0.12, 0.19, 0.27])
      hs = h isa Real ? fill(h, 3) : h
      # Robinson's estimator with a separate smoother matrix for each response
      Sy = smoother_matrix(U, U; bandwidth = hs[1], weights = w)
      X̃ = hcat([(I - smoother_matrix(U, U; bandwidth = hs[j + 1], weights = w)) * X[:, j]
                 for j in axes(X, 2)]...)
      ỹ = (I - Sy) * Yp
      Sg = smoother_matrix(U, PTS; bandwidth = hs[1], weights = w)
      ww = w === nothing ? ones(N) : w
      G = X̃' * Diagonal(ww) * X̃
      β = G \ (X̃' * (ww .* ỹ))
      e = ỹ .- X̃ * β
      meat = X̃' * Diagonal((ww .* e) .^ 2) * X̃
      for vce in (HCVariance(0), HCVariance(1))
        m = fit(PartiallyLinear, U, X, Yp; bandwidth = h, points = PTS, weights = w, vce)
        correction = vce.type == 1 ? N / (N - 2) : 1.0
        @test coef(m) ≈ β rtol = 1e-8
        @test vcov(m) ≈ correction .* (G \ meat / G) rtol = 1e-8
        @test m.smooth ≈ Sg * (Yp .- X * β) rtol = 1e-8
        @test m.bandwidths == hs
        @test bandwidth(m) == hs[1] == m.selection.h
        @test m.selection.rule === nothing && m.selections === nothing
      end
    end
  end

  @testset "selectors and response reference" begin
    # Default fits at 37e6195, with and without weights
    references = (
      (h = 0.1670507964111671,
       coef = [1.5043826174758221, -0.49955393952324467],
       vcov = [8.010377729435364e-5, 2.2979679724259692e-8,
               2.2979679724259692e-8, 0.00010114132419316274],
       smooth = [-0.07625960048329118, 0.45381042899666285, 0.08431871402663756,
                 -0.5994053485538163, -0.1419127125234023]),
      (h = 0.16713998976569977,
       coef = [1.5040373651245378, -0.4999689106122242],
       vcov = [9.378615489911753e-5, 5.257242405278155e-7,
               5.257242405278155e-7, 0.00011740922775933086],
       smooth = [-0.07755317634902498, 0.4526925509331564, 0.08355306209939613,
                 -0.6015596210785551, -0.1459590200351563]))
    for (w, ref) in zip((nothing, W), references)
      chosen = [bandwidth(CrossValidation(), SmoothingData(U, v; weights = w), K; degree = 1)
                for v in (Yp, X2, X3)]
      for mode in (:each, :smallest, :response)
        m = fit(PartiallyLinear, U, X, Yp; bandwidth_mode = mode, weights = w, points = PTS)
        h = mode === :each ? chosen : mode === :smallest ? minimum(chosen) : chosen[1]
        numeric = fit(PartiallyLinear, U, X, Yp; bandwidth = h, weights = w, points = PTS)
        @test m.bandwidth_mode === mode
        @test m.bandwidths == (h isa Real ? fill(h, 3) : h)
        @test bandwidth(m) == m.bandwidths[1]
        @test m.selections isa Vector{typeof(KernelSmoothing._anyselection(m.selection))}
        @test getproperty.(m.selections, :h) == chosen
        @test all(s -> s.rule == CrossValidation(), m.selections)
        @test m.selection === m.selections[1]
        @test coef(m) ≈ coef(numeric) rtol = 1e-12
        @test vcov(m) ≈ vcov(numeric) rtol = 1e-12
        @test m.smooth ≈ numeric.smooth rtol = 1e-12
        if mode === :response
          @test bandwidth(m) ≈ ref.h rtol = 1e-12
          @test coef(m) ≈ ref.coef rtol = 1e-12
          @test vec(vcov(m)) ≈ ref.vcov rtol = 1e-12
          @test m.smooth ≈ ref.smooth rtol = 1e-12
        elseif mode === :smallest
          default = fit(PartiallyLinear, U, X, Yp; weights = w, points = PTS)
          @test default.bandwidth_mode === :smallest
          @test default.selection.rule == CrossValidation()
          @test default.selection === default.selections[1]
          @test default.bandwidths == fill(minimum(s.h for s in default.selections), 3) == m.bandwidths
          @test bandwidth(default) == bandwidth(m)
          @test coef(default) ≈ coef(m) rtol = 1e-12
          @test vcov(default) ≈ vcov(m) rtol = 1e-12
          @test default.smooth ≈ m.smooth rtol = 1e-12
        end
      end
    end
  end

  @testset "bandwidth edge cases" begin
    hs = [0.12, 0.19, 0.27]
    scalar = fit(PartiallyLinear, U, X, Yp; bandwidth = 0.15, points = PTS)
    unequal = fit(PartiallyLinear, U, X, Yp; bandwidth = hs, points = PTS)
    for mode in (:each, :smallest, :response)
      equal = fit(PartiallyLinear, U, X, Yp; bandwidth = fill(0.15, 3), bandwidth_mode = mode, points = PTS)
      numeric = fit(PartiallyLinear, U, X, Yp; bandwidth = hs, bandwidth_mode = mode, points = PTS)
      common = fit(PartiallyLinear, U, X, Yp; bandwidth = 0.15, bandwidth_mode = mode, points = PTS)
      @test equal.bandwidths == common.bandwidths == fill(0.15, 3)
      @test numeric.bandwidths == hs
      @test numeric.bandwidth_mode === equal.bandwidth_mode === common.bandwidth_mode === mode
      for actual in (equal, common)
        @test coef(actual) ≈ coef(scalar) rtol = 1e-12
        @test vcov(actual) ≈ vcov(scalar) rtol = 1e-12
        @test actual.smooth ≈ scalar.smooth rtol = 1e-12
      end
      @test coef(numeric) ≈ coef(unequal) rtol = 1e-12
      @test vcov(numeric) ≈ vcov(unequal) rtol = 1e-12
      @test numeric.smooth ≈ unequal.smooth rtol = 1e-12
    end
    @test unequal.bandwidths !== hs
    hs[1] = 0.4
    @test bandwidth(unequal) == unequal.bandwidths[1] == 0.12
    converted = fit(PartiallyLinear, U, X, Yp; bandwidth = Float32[0.12, 0.19, 0.27])
    @test converted.bandwidths isa Vector{Float64}
    @test converted.bandwidths == Float64.(Float32[0.12, 0.19, 0.27])

    wz = copy(W)
    wz[1:9:end] .= 0.0
    order = sort(findall(>(0), wz); by = i -> U[i])
    for h in ([0.12, 0.19, 0.27], CrossValidation(gridsize = 9))
      m = fit(PartiallyLinear, U, X, Yp; bandwidth = h, bandwidth_mode = :each, weights = wz, points = PTS)
      sorted = fit(PartiallyLinear, U[order], X[order, :], Yp[order]; bandwidth = h,
        bandwidth_mode = :each, weights = wz[order], points = PTS)
      @test nobs(m) == nobs(sorted) == length(order)
      @test m.bandwidths == sorted.bandwidths
      @test coef(m) ≈ coef(sorted) rtol = 1e-12
      @test vcov(m) ≈ vcov(sorted) rtol = 1e-12
      @test m.smooth ≈ sorted.smooth rtol = 1e-12
      hp = h isa AbstractVector ? h[[1, 3, 2]] : h
      permuted = fit(PartiallyLinear, U, X[:, [2, 1]], Yp; bandwidth = hp,
        bandwidth_mode = :each, weights = wz, points = PTS)
      @test permuted.bandwidths == m.bandwidths[[1, 3, 2]]
      @test coef(permuted) ≈ coef(m)[[2, 1]] rtol = 1e-12
      @test vcov(permuted) ≈ vcov(m)[[2, 1], [2, 1]] rtol = 1e-12
      @test permuted.smooth ≈ m.smooth rtol = 1e-12
      if m.selections !== nothing
        @test getproperty.(permuted.selections, :h) == getproperty.(m.selections, :h)[[1, 3, 2]]
      end
    end

    for h in ([0.12, 0.19], CrossValidation(gridsize = 9))
      vector = fit(PartiallyLinear, U, X2, Yp; bandwidth = h, bandwidth_mode = :each, points = PTS)
      matrix = fit(PartiallyLinear, U, reshape(X2, :, 1), Yp; bandwidth = h, bandwidth_mode = :each, points = PTS)
      @test length(vector.bandwidths) == 2
      @test vector.bandwidths == matrix.bandwidths
      @test coef(vector) ≈ coef(matrix) rtol = 1e-12
      @test vcov(vector) ≈ vcov(matrix) rtol = 1e-12
      @test vector.smooth ≈ matrix.smooth rtol = 1e-12
    end

    # Cancellation makes the response smoother than the second covariate's mean
    Xc = hcat(U .+ 0.2 .* X2, sin.(2π .* U) .+ 0.2 .* X3)
    yc = Xc[:, 1] .- 0.5 .* Xc[:, 2] .+ 0.5 .* sin.(2π .* U) .+ E
    rule, kernel, degree = CrossValidation(gridsize = 9, exact = 50), BiweightKernel(), 2
    chosen = [bandwidth(rule, SmoothingData(U, v; weights = W), kernel; degree)
              for v in (yc, Xc[:, 1], Xc[:, 2])]
    @test chosen[3] < chosen[1]
    for mode in (:each, :smallest, :response)
      m = fit(PartiallyLinear, U, Xc, yc; bandwidth = rule, bandwidth_mode = mode,
        kernel, degree, weights = W, points = PTS)
      h = mode === :each ? chosen : mode === :smallest ? minimum(chosen) : chosen[1]
      numeric = fit(PartiallyLinear, U, Xc, yc; bandwidth = h, kernel, degree, weights = W, points = PTS)
      @test m.degree == degree && m.kernel == kernel
      @test getproperty.(m.selections, :h) == chosen
      @test all(s -> s.rule == rule && length(s.details.grid) == rule.gridsize, m.selections)
      @test m.bandwidths == (h isa Real ? fill(h, 3) : h)
      @test bandwidth(m) == m.bandwidths[1]
      @test coef(m) ≈ coef(numeric) rtol = 1e-12
      @test vcov(m) ≈ vcov(numeric) rtol = 1e-12
      @test m.smooth ≈ numeric.smooth rtol = 1e-12
      if mode === :smallest
        default = fit(PartiallyLinear, U, Xc, yc; bandwidth = rule, kernel, degree, weights = W, points = PTS)
        @test default.bandwidth_mode === :smallest
        @test default.bandwidths == fill(minimum(s.h for s in default.selections), 3) == m.bandwidths
        @test bandwidth(default) < default.selection.h
        @test coef(default) ≈ coef(m) rtol = 1e-12
        @test vcov(default) ≈ vcov(m) rtol = 1e-12
        @test default.smooth ≈ m.smooth rtol = 1e-12
      end
    end
  end
  m = fit(PartiallyLinear, U, X, Yp)
  @test m.selection.rule == CrossValidation()
  @test nobs(m) == N && bandwidth(m) == minimum(s.h for s in m.selections)
  @test coef(m) ≈ [1.5, -0.5] atol = 0.05
  t = coeftable(m)
  @test t.estimate == coef(m) && t.se == stderror(m) && t.upper ≈ confint(m)[:, 2]
  @test startswith(sprint(show, m), "PartiallyLinear(2 coefficients")
  @inferred fit(PartiallyLinear, U, X, Yp; bandwidth = 0.15)
  @inferred fit(PartiallyLinear, U, X, Yp; bandwidth = [0.12, 0.19, 0.27])
  @inferred fit(PartiallyLinear, U, X, Yp; bandwidth_mode = :each)
end

@testset "errors" begin
  @test_throws DimensionMismatch fit(PartiallyLinear, U, X2[1:10], Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(PartiallyLinear, U, hcat(ones(N), X2), Y; bandwidth = 0.2)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.2, vce = HCVariance(3))
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = 0.001)
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = MSEOptimal())
  @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = NormalReference())
  for h in ([0.1], [0.1, 0.2, 0.3])
    @test_throws DimensionMismatch(
      "bandwidth has $(length(h)) entries for 1 + size(X, 2) = 2 smooths (y first, then the columns of X)") fit(
      PartiallyLinear, U, X2, Y; bandwidth = h)
  end
  for h in (0.0, -0.1, NaN, Inf)
    @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = h)
    for hs in ([h, 0.2], [0.2, h])
      @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = hs)
    end
  end
  for h in (0.15, [0.15, 0.2], CrossValidation()), mode in (:unknown, "each", nothing)
    @test_throws ArgumentError fit(PartiallyLinear, U, X2, Y; bandwidth = h, bandwidth_mode = mode)
  end
end

end
