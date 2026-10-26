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
  # the units of the points and the values do not matter, and the solver converges
  for s in SHAPES
    @test (@test_logs shape_projection(Y, 1e7 .* T .+ 1e9; shape = s)) ≈ UNWEIGHTED[s] rtol = 1e-8 atol = 1e-11
    @test (@test_logs shape_projection(1e-6 .* Y, T; shape = s)) ≈ 1e-6 .* UNWEIGHTED[s] rtol = 1e-8 atol = 1e-17
    @test (@test_logs shape_projection(1e6 .* Y, 1e-5 .* T; shape = s, weights = W)) ≈ 1e6 .* WEIGHTED[s] rtol = 1e-8 atol = 1e-5
  end
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
  # nonnegative least squares against a brute force over the active sets, also with
  # columns of very different scales and a zero column
  for c in 1:30
    A, b = randn(rng, 8, 4), randn(rng, 8)
    c > 20 && (A .*= [1e-6 1.0 1e5 1.0])
    c == 30 && (A[:, 2] .= 0)
    x = _nnls(A, b)
    @test all(>=(0), x)
    best = minimum(norm(A[:, S] * max.(A[:, S] \ b, 0) - b) for S in ([i for i in 1:4 if (k >> (i - 1)) & 1 == 1]
                   for k in 1:15) if all(A[:, S] \ b .>= 0); init = norm(b))
    @test norm(A * x - b) ≈ best rtol = 1e-10
  end
  @test _nnls(randn(rng, 5, 3), zeros(5)) == zeros(3)
  # entering indices whose coefficient is not positive are refused: with a negative
  # tolerance, those with a negative gradient try to enter, and the optimum is the same
  for _ in 1:5
    A, b = randn(rng, 10, 6), randn(rng, 10)
    @test _nnls(A, b; tol = -1.0) ≈ _nnls(A, b) atol = 1e-12
  end
  @test_logs (:warn, r"iteration limit") _nnls([1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, 2.0, 3.0]; maxiter = 1)
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
  cd = confint(d)
  @test confint(rearrange(d)) == hcat(sort(cd[:, 1]), sort(cd[:, 2]))
  # intervals with NaN ends stay in place, and the others are rearranged among themselves
  dn = fit(PointwiseDensity, x; points = [-5.0; 0.2:0.1:0.8], bandwidth = 0.1, minobs = 0)
  cn, crn = confint(dn), confint(rearrange(dn))
  bad = findall(i -> isnan(cn[i, 1]) || isnan(cn[i, 2]), axes(cn, 1))
  good = setdiff(axes(cn, 1), bad)
  @test !isempty(bad) && isequal(crn[bad, :], cn[bad, :])
  @test crn[good, 1] == sort(cn[good, 1]) && crn[good, 2] == sort(cn[good, 2])
  # the other fits with one estimate per point
  lpd = fit(LocalPolynomialDensity, x; points = 0.2:0.1:0.8)
  @test coef(rearrange(lpd)) == sort(coef(lpd))
  yb = Float64.(y .> x .^ 2)                      # the sign of the noise
  ll = fit(LocalLikelihood, x, yb; bandwidth = 0.2, points = 0.1:0.1:0.9)
  @test coef(constrain(ll, :increasing)) ≈ shape_projection(coef(ll), ll.points)
  # the link is increasing, so that rearranging the linear predictor rearranges the mean
  @test coef(rearrange(ll)) == sort(coef(ll)) && sortperm(predict(ll)) == sortperm(coef(ll))
  # a fit whose coefficients are not estimates at its points
  pl = fit(PartiallyLinear, x, reshape(sin.(1:length(x)), :, 1), y; bandwidth = 0.1)
  @test_throws ArgumentError rearrange(pl)
  @test_throws ArgumentError constrain(pl, :increasing)
  # repeated points share one value: the projection of the pooled estimates
  dt = fit(PointwiseDensity, round.(x; digits = 1))
  ct = constrain(dt, :decreasing)
  @test issorted(coef(ct); rev = true)
  u = unique(dt.points)
  @test length(u) < length(dt.points)
  @test all(i -> coef(ct)[i] == coef(ct)[findfirst(==(dt.points[i]), dt.points)], eachindex(dt.points))
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
