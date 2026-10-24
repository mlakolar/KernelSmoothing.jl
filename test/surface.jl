module SurfaceTest

using Test
using KernelSmoothing
using LinearAlgebra: Diagonal

const N = 300
const X1 = [mod(0.7548776662466927i, 1.0) for i in 1:N]
const X2 = [mod(0.5698402909980532i, 1.0) for i in 1:N]
const Y = sin.(2π .* X1) .* cos.(π .* X2) .+ 0.1 .* sin.(17.3 .* (1:N))
const W = 1.0 .+ mod.(1:N, 3)
const G = repeat(1:30, inner = 10)

# the fit at (s, t), by its definition
function wls(s, t, h, p, k, w)
  D = Diagonal(w .* k.((X1 .- s) ./ h[1]) .* k.((X2 .- t) ./ h[2]))
  X = p == 0 ? ones(N, 1) : [ones(N) (X1 .- s) (X2 .- t)]
  return ((X' * D * X) \ (X' * D * Y))[1]
end

@testset "bivariate local polynomial fits" begin
  g1, g2 = [0.0, 0.3, 0.75], [0.1, 0.5, 0.9, 1.0]
  for k in (EpanechnikovKernel(), TriangularKernel(), GaussianKernel()), p in (0, 1), w in (nothing, W)
    h = k isa GaussianKernel ? (0.08, 0.1) : (0.2, 0.25)
    E = localpolynomial((X1, X2), Y, (g1, g2); bandwidth = h, degree = p, kernel = k, weights = w)
    @test size(E) == (3, 4)
    @test E ≈ [wls(s, t, h, p, k, w === nothing ? ones(N) : w) for s in g1, t in g2] rtol = 1e-10
    # scattered points as a matrix or as pairs
    @test localpolynomial((X1, X2), Y, [0.3 0.5; 0.75 0.1]; bandwidth = h, degree = p, kernel = k, weights = w) ≈
          [E[2, 2], E[3, 1]] rtol = 1e-12
    @test localpolynomial((X1, X2), Y, [(0.3, 0.5)]; bandwidth = h, degree = p, kernel = k, weights = w) ≈ [E[2, 2]]
    # the weights, in the input order and column-major over the grid
    L = smoothermatrix((X1, X2), (g1, g2); bandwidth = h, degree = p, kernel = k, weights = w)
    @test L * Y ≈ vec(E) rtol = 1e-10
    @test all(sum(L; dims = 2) .≈ 1)
    p == 1 && @test L * X2 ≈ vec([t for s in g1, t in g2]) atol = 1e-10
  end
  # one bandwidth for both coordinates
  @test localpolynomial((X1, X2), Y, (g1, g2); bandwidth = 0.2) == localpolynomial((X1, X2), Y, (g1, g2); bandwidth = (0.2, 0.2))
  # a window without enough observations
  @test isnan(localpolynomial((X1, X2), Y, [(3.0, 3.0)]; bandwidth = 0.1)[1])
  @inferred localpolynomial((X1, X2), Y, (g1, g2); bandwidth = 0.2)
end

@testset "cross-validation" begin
  b = KernelSmoothing._bivariate((X1, X2), Y, nothing, G)
  k = EpanechnikovKernel()
  h = (0.15, 0.15)
  # leave-one-out and leave-group-out, against fits without the observations
  loo, lgo = 0.0, 0.0
  for i in 1:N
    for (keep, acc) in ((setdiff(1:N, i), :loo), (findall(!=(G[i]), G), :lgo))
      D = Diagonal(k.((X1[keep] .- X1[i]) ./ h[1]) .* k.((X2[keep] .- X2[i]) ./ h[2]))
      X = [ones(length(keep)) (X1[keep] .- X1[i]) (X2[keep] .- X2[i])]
      r = (Y[i] - ((X' * D * X) \ (X' * D * Y[keep]))[1])^2
      acc === :loo ? (loo += r) : (lgo += r)
    end
  end
  @test KernelSmoothing._surfacecv(b, k, h, 1, false) ≈ loo / N rtol = 1e-10
  @test KernelSmoothing._surfacecv(b, k, h, 1, true) ≈ lgo / N rtol = 1e-10
  s = selectbandwidth(CrossValidation(), (X1, X2), Y)
  @test s.h[1] == s.h[2] && 0.05 < s.h[1] < 0.5
  @test s.rule == CrossValidation() && length(s.details.grid) == 40 && !s.details.groups
  sg = selectbandwidth(CrossValidation(), (X1, X2), Y; groups = G, aspect = 2)
  @test sg.h[2] ≈ 2 * sg.h[1] && sg.details.groups
  # candidate bandwidths
  sc = selectbandwidth(CrossValidation(), (X1, X2), Y; bandwidths = [0.3, 0.1, 0.2])
  @test sc.details.grid == [0.1, 0.2, 0.3] && sc.h[1] == sc.details.grid[argmin(sc.details.criterion)]
end

@testset "errors" begin
  @test_throws ArgumentError localpolynomial((X1, X2), Y, (X1, X2); bandwidth = 0.1, degree = 2)
  @test_throws ArgumentError localpolynomial((X1, X2), Y, [0.1, 0.2]; bandwidth = 0.1)
  @test_throws ArgumentError localpolynomial((X1, X2), Y, [0.1 0.2 0.3]; bandwidth = 0.1)
  @test_throws ArgumentError localpolynomial((X1, X2), Y, [(0.1, 0.2)]; bandwidth = (0.1, -0.2))
  @test_throws DimensionMismatch localpolynomial((X1, X2[1:10]), Y, [(0.1, 0.2)]; bandwidth = 0.1)
  @test_throws DimensionMismatch localpolynomial((X1, X2), Y[1:10], [(0.1, 0.2)]; bandwidth = 0.1)
  @test_throws ArgumentError localpolynomial((X1, X2), Y, [(0.1, 0.2)]; bandwidth = 0.1, weights = -W)
  @test_throws ArgumentError selectbandwidth(CrossValidation(), (X1, X2), Y; aspect = 0)
  @test_throws ArgumentError selectbandwidth(CrossValidation(), (X1, X2), Y; bandwidths = [0.001])
  @test_throws DimensionMismatch selectbandwidth(CrossValidation(), (X1, X2), Y; groups = G[1:5])
end

end
