module ShapeTest

using Test
using KernelSmoothing
using KernelSmoothing: _nnls
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "data", "shape_golden.jl"))

const SHAPES = (:increasing, :decreasing, :convex, :concave, :increasing_convex, :increasing_concave,
                :decreasing_convex, :decreasing_concave)
const WEIGHTED = Dict(:increasing => PROJ_INCREASING, :decreasing => PROJ_DECREASING, :convex => PROJ_CONVEX,
                      :concave => PROJ_CONCAVE, :increasing_convex => PROJ_INCREASING_CONVEX,
                      :increasing_concave => PROJ_INCREASING_CONCAVE, :decreasing_convex => PROJ_DECREASING_CONVEX,
                      :decreasing_concave => PROJ_DECREASING_CONCAVE)
const UNWEIGHTED = Dict(:increasing => PROJU_INCREASING, :decreasing => PROJU_DECREASING, :convex => PROJU_CONVEX,
                        :concave => PROJU_CONCAVE, :increasing_convex => PROJU_INCREASING_CONVEX,
                        :increasing_concave => PROJU_INCREASING_CONCAVE, :decreasing_convex => PROJU_DECREASING_CONVEX,
                        :decreasing_concave => PROJU_DECREASING_CONCAVE)

# whether θ has the shape at the points t, to a tolerance
function hasshape(θ, t, shape; tol = 1e-10)
  d1 = diff(θ) ./ diff(t)
  d2 = diff(d1)
  s = string(shape)
  ok = true
  occursin("increasing", s) && (ok &= all(>=(-tol), d1))
  occursin("decreasing", s) && (ok &= all(<=(tol), d1))
  occursin("convex", s) && (ok &= all(>=(-tol), d2))
  occursin("concave", s) && (ok &= all(<=(tol), d2))
  return ok
end

@testset "projections onto cones of shapes" begin
  for s in SHAPES
    θ = shape_projection(Y, T; shape = s, weights = W)
    @test θ ≈ WEIGHTED[s] rtol = 1e-9 atol = 1e-12
    @test shape_projection(Y, T; shape = s) ≈ UNWEIGHTED[s] rtol = 1e-9 atol = 1e-12
    @test hasshape(θ, T, s)
    # a projection: its residual is orthogonal to θ in the weighted inner product,
    # and projecting again changes nothing
    @test dot(W .* (Y .- θ), θ) ≈ 0 atol = 1e-10
    @test shape_projection(θ, T; shape = s, weights = W) ≈ θ atol = 1e-11
    # a sequence of the shape is its own projection
    good = WEIGHTED[s] .+ 0.0
    @test shape_projection(good, T; shape = s, weights = W) ≈ good atol = 1e-10
  end
  @test shape_projection(Y, T; shape = :increasing) ≈ ISOREG rtol = 1e-12
  # the projection does not increase the distance to any sequence of the shape
  rng = Xoshiro(5)
  for s in SHAPES, _ in 1:5
    target = shape_projection(randn(rng, length(T)) .* 3, T; shape = s)
    θ = shape_projection(Y, T; shape = s, weights = W)
    @test sum(W .* (θ .- target) .^ 2) <= sum(W .* (Y .- target) .^ 2) + 1e-10
  end
  # short sequences
  @test shape_projection([2.0], [0.0]) == [2.0]
  @test shape_projection([2.0, 1.0], [0.0, 1.0]; shape = :convex) ≈ [2.0, 1.0]
  @test shape_projection([2.0, 1.0], [0.0, 1.0]) ≈ [1.5, 1.5]
  @test_throws ArgumentError shape_projection(Y, T; shape = :wiggly)
  @test_throws ArgumentError shape_projection(Y, reverse(T))
  @test_throws ArgumentError shape_projection(Y, T; weights = -W)
  @test_throws DimensionMismatch shape_projection(Y, T[1:3])
  # nonnegative least squares against a brute force over the active sets
  for _ in 1:20
    A, b = randn(rng, 8, 4), randn(rng, 8)
    x = _nnls(A, b)
    @test all(>=(0), x)
    best = minimum(norm(A[:, S] * max.(A[:, S] \ b, 0) - b) for S in ([i for i in 1:4 if (k >> (i - 1)) & 1 == 1]
                   for k in 1:15) if all(A[:, S] \ b .>= 0); init = norm(b))
    @test norm(A * x - b) ≈ best rtol = 1e-10
  end
end

@testset "rearrangement and constrained fits" begin
  x, y = SX, SY                                   # an increasing truth, x²
  m = fit(LocalPolynomialRegression, x, y; bandwidth = 0.08, points = SPTS)
  r = rearrange(m)
  @test coef(r) == sort(coef(m))
  @test issorted(coef(r))
  truth = collect(m.points) .^ 2
  @test sum(abs2, coef(r) .- truth) <= sum(abs2, coef(m) .- truth)
  ci = confint(m)
  @test confint(r) == hcat(sort(ci[:, 1]), sort(ci[:, 2]))
  b, br = conf_band(m; rng = Xoshiro(1)), conf_band(r; rng = Xoshiro(1))
  @test br.lower == sort(b.lower)
  @test br.upper == sort(b.upper)
  @test br.critical == b.critical
  # a band that covers a monotone function still covers it
  @test all(br.lower .<= truth .<= br.upper) || !all(b.lower .<= truth .<= b.upper)
  @test coef(rearrange(m; decreasing = true)) == sort(coef(m); rev = true)
  # other fits
  d = fit(PointwiseDensity, x; points = 0.2:0.1:0.8)
  @test coef(rearrange(d)) == sort(coef(d))
  @test size(confint(rearrange(d))) == (7, 2)
  # the constrained fit is the projection of the estimates
  c = constrain(m, :increasing)
  @test coef(c) ≈ shape_projection(coef(m), m.points)
  @test issorted(coef(c))
  @test sum(abs2, coef(c) .- truth) <= sum(abs2, coef(m) .- truth) * (1 + 1e-12)
  # close to the constrained spline fits of cgam and scam, which smooth differently
  # (cgam's increasing fit is a step function, scam's a monotone P-spline)
  @test maximum(abs.(coef(c) .- SCAM_MPI)) < 0.03
  @test maximum(abs.(coef(c) .- CGAM_INCR)) < 0.05
  @test maximum(abs.(coef(c) .- truth)) <= maximum(abs.(SCAM_MPI .- truth))
  cw = constrain(m, :convex; weights = 1 ./ stderror(m) .^ 2)
  @test hasshape(coef(cw), m.points, :convex)
  @test sprint(show, r) == "Rearranged(increasing, 31 points, LocalPolynomialRegression)"
  @test sprint(show, c) == "ShapeConstrained(:increasing, 31 points, LocalPolynomialRegression)"
  @test_throws ArgumentError rearrange(fit(LocalPolynomialRegression, x, y; bandwidth = 0.08, points = [0.5, 0.2]))
end

end
