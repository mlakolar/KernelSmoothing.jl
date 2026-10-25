module BuildingTest

using Test
using KernelSmoothing
using LinearAlgebra: Diagonal, I, diag
using Distributions: Normal, cdf as ncdf, quantile as nquantile
using Random: MersenneTwister

const N = 300
const X = [mod(0.7548776662466927i, 1.0) for i in 1:N]
const Y = sin.(2π .* X) .+ 0.3 .* sin.(17.3 .* (1:N))
const W = 1.0 .+ mod.(1:N, 3)
const CL = ["c$(mod(i, 25))" for i in 1:N]   # 25 clusters, labelled by strings

@testset "kernel weights" begin
  for w in (nothing, W), k in (EpanechnikovKernel(), GaussianKernel(), UniformKernel()),
      (x0, h) in ((0.5, 0.1), (0.02, 0.05), (2.0, 0.1))
    d = SmoothingData(X, Y; weights = w)
    r = kernelweights(d, k, x0, h)
    full = (w === nothing ? ones(N) : w) .* k.((X .- x0) ./ h)   # in the input order
    @test r.index == d.perm[r.window]
    @test sort(r.index) == findall(>(0), full)
    @test r.weights == full[r.index]
    @test issorted(d.x[r.window])
  end
  d = SmoothingData(X, Y)
  @test isempty(kernelweights(d, EpanechnikovKernel(), 2.0, 0.1).window)
  @test_throws ArgumentError kernelweights(d, EpanechnikovKernel(), 0.5, 0.0)
  @test_throws ArgumentError kernelweights(d, EpanechnikovKernel(), NaN, 0.1)
  @test_throws ArgumentError kernelweights(d, derivative(EpanechnikovKernel(), 1), 0.5, 0.1)
end

@testset "regression points" begin
  d = SmoothingData(X, Y)
  @test regressionpoints(d) == fit(LocalPolynomialRegression, X, Y; bandwidth = 0.2).points
  @test regressionpoints(d; compat = true) == collect(range(minimum(X), maximum(X); length = 30))
  @test length(regressionpoints(d)) == 30 && minimum(X) < regressionpoints(d)[1] < 0.05
end

@testset "bandwidth grids" begin
  d = SmoothingData(X, Y)
  g = bandwidthgrid(d; minobs = 10, mindistinct = 3, gridsize = 12)
  hmin = maximum(i -> max(sort(abs.(X .- X[i]))[10], sort(abs.(unique(X) .- X[i]))[3]), 1:N) * 1.01
  @test length(g) == 12
  @test g[1] ≈ hmin rtol = 1e-12
  @test g[end] ≈ maximum(X) - minimum(X) rtol = 1e-12
  @test all(≈(log(g[2] / g[1])), diff(log.(g)))
  @test all(i -> length(window(d, X[i], g[1])) >= 10, 1:N)
  # the smallest bandwidth falls back on half the range when it would exceed it
  @test bandwidthgrid(SmoothingData([0.0, 1.0, 2.0]); minobs = 3)[1] == 1.0
  # the grid of local likelihood's cross-validation
  yb = Float64.(sin.(7.1 .* (1:N)) .> 0)
  m = fit(LocalLikelihood, X, yb; family = :logit)
  @test m.selection.details.grid ==
        bandwidthgrid(SmoothingData(X, yb); minobs = 22, mindistinct = 3, gridsize = m.selection.rule.gridsize)
  @test_throws ArgumentError bandwidthgrid(d; gridsize = 1)
  @test_throws ArgumentError bandwidthgrid(d; minobs = 0)
  @test_throws ArgumentError bandwidthgrid(d; mindistinct = N + 1)
  @test_throws ArgumentError bandwidthgrid(SmoothingData(fill(1.0, 10)))
  @test_throws ArgumentError bandwidthgrid(SmoothingData([0.0, 0.0, 1.0, 1.0]); minobs = 2, mindistinct = 1)
end

@testset "grid search" begin
  f(h) = (log(h) - log(0.123))^2
  h, details = gridminimize(f, exp.(range(log(0.01), log(1.0); length = 15)))
  @test h ≈ 0.123 rtol = 2e-3
  @test !details.atbound && details.criterion == f.(details.grid)
  h, details = gridminimize(identity, [1.0, 2.0, 3.0])
  @test h == 1.0 && details.atbound
  # values that are not finite count as Inf
  h, _ = gridminimize(t -> t < 0.45 ? Inf : (t - 0.7)^2, collect(0.1:0.1:1.0))
  @test h ≈ 0.7 rtol = 1e-2
  # the largest local minimizer within half of the lowest local minimum
  g(t) = min((t - 0.2)^2 + 0.01, (t - 0.8)^2 + 0.012)
  grid = collect(0.05:0.05:0.95)
  @test gridminimize(g, grid)[1] ≈ 0.2 rtol = 1e-2
  @test gridminimize(g, grid; largest = true)[1] ≈ 0.8 rtol = 1e-2
  @test_throws ArgumentError gridminimize(t -> Inf, [1.0, 2.0])
  @test_throws ArgumentError gridminimize(identity, [2.0, 1.0])
  @test_throws ArgumentError gridminimize(identity, Float64[])
  @test_throws ArgumentError gridminimize(identity, [1.0, 2.0]; rtol = 0)
end

# the sandwich by its definition, with matrix functions for CR2 and CR3
function reference(Z, v, e, vce, cl)
  keep = v .> 0
  Z, v, e = Z[keep, :], v[keep], e[keep]
  n, D = size(Z)
  Ginv = inv(Z' * Diagonal(v) * Z)
  H = Z * Ginv * Z' * Diagonal(v)
  t = vce.type
  if vce isa HCVariance
    f = t == 0 ? ones(n) : t == 1 ? fill(sqrt(n / (n - D)), n) :
        t == 2 ? 1 ./ sqrt.(1 .- diag(H)) : 1 ./ (1 .- diag(H))
    return Ginv * (Z' * Diagonal((v .* e .* f) .^ 2) * Z) * Ginv
  end
  cl = cl[keep]
  labels = unique(cl)
  meat = zeros(D, D)
  for g in labels
    idx = findall(==(g), cl)
    B = I - H[idx, idx]
    et = t == 2 ? real(inv(sqrt(B))) * e[idx] : t == 3 ? B \ e[idx] : e[idx]
    s = Z[idx, :]' * (v[idx] .* et)
    meat += s * s'
  end
  G = length(labels)
  c = t == 1 ? (n - 1) / (n - D) * G / (G - 1) : t == 3 ? (G - 1) / G : 1.0
  return c * Ginv * meat * Ginv
end

@testset "sandwich variances" begin
  x0, h = 0.5, 0.25
  K = EpanechnikovKernel()
  d = SmoothingData(X, Y; weights = W)
  r = kernelweights(d, K, x0, h)
  u = (d.x[r.window] .- x0) ./ h
  X2 = cos.(3.7 .* r.index)
  designs = (hcat(ones(length(u)), u), hcat(ones(length(u)), u, X2, X2 .* u))
  for Z in designs
    # zero weights are dropped
    v = vcat(r.weights, 0.0)
    Zv = vcat(Z, fill(9.0, 1, size(Z, 2)))
    yv = vcat(d.y[r.window], 100.0)
    θ = (Z' * (r.weights .* Z)) \ (Z' * (r.weights .* d.y[r.window]))
    e = yv .- Zv * θ
    cl = vcat(CL[r.index], "c99")
    for t in 0:3
      @test sandwich(Zv, v, e; vce = HCVariance(t)) ≈ reference(Zv, v, e, HCVariance(t), nothing) rtol = 1e-9
      @test sandwich(Zv, v, e; vce = ClusterVariance(t), cluster = cl) ≈
            reference(Zv, v, e, ClusterVariance(t), cl) rtol = 1e-9
    end
  end
  # the conventional standard errors of LocalPolynomialRegression
  Z = designs[1]
  θ = (Z' * (r.weights .* Z)) \ (Z' * (r.weights .* d.y[r.window]))
  e = d.y[r.window] .- Z * θ
  for t in 0:3
    m = fit(LocalPolynomialRegression, X, Y; weights = W, points = [x0], bandwidth = h, vce = HCVariance(t), minobs = 0)
    @test sqrt(sandwich(Z, r.weights, e; vce = HCVariance(t))[1, 1]) ≈ m.se[1] rtol = 1e-10
    mc = fit(LocalPolynomialRegression, X, Y; weights = W, cluster = CL, points = [x0], bandwidth = h,
             vce = ClusterVariance(t), minobs = 0)
    @test sqrt(sandwich(Z, r.weights, e; vce = ClusterVariance(t), cluster = CL[r.index])[1, 1]) ≈ mc.se[1] rtol = 1e-10
  end
  # where the variance cannot be computed
  @test all(isnan, sandwich(hcat(Z, 2 .* Z[:, 2]), r.weights, e))
  @test all(isnan, sandwich(Z, r.weights, e; vce = ClusterVariance(1), cluster = fill("one", length(e))))
  @test all(isnan, sandwich(Z[1:2, :], r.weights[1:2], e[1:2]; vce = HCVariance(1)))
  @test_throws DimensionMismatch sandwich(Z, r.weights[1:5], e)
  @test_throws DimensionMismatch sandwich(Z, r.weights, e[1:5])
  @test_throws ArgumentError sandwich(Z, -r.weights, e)
  @test_throws ArgumentError sandwich(Z, r.weights, e; vce = ClusterVariance(1))
  @test_throws ArgumentError sandwich(Z, r.weights, e; cluster = CL[r.index])
  @test_throws DimensionMismatch sandwich(Z, r.weights, e; vce = ClusterVariance(1), cluster = CL)
end

@testset "critical values of uniform bands" begin
  # independent estimates: P(max |Zⱼ| ≤ c) = (2Φ(c) - 1)^5 = level
  exact = nquantile(Normal(), (1 + 0.95^(1 / 5)) / 2)
  c = supt_critical(Matrix(1.0I, 5, 5); rng = MersenneTwister(1))
  @test abs(c - exact) < 0.05
  @test supt_critical(fill(4.0, 1, 1)) == nquantile(Normal(), 0.975)
  # estimates without variance are left out
  @test supt_critical(Matrix(Diagonal([1.0, 0.0, 2.0])); rng = MersenneTwister(3)) ==
        supt_critical(Matrix(Diagonal([1.0, 2.0])); rng = MersenneTwister(3))
  # the critical value of confband
  m = fit(LocalPolynomialRegression, X, Y; bandwidth = 0.1, points = [0.2, 0.4, 0.6, 0.8], vce = HCVariance(0))
  b = confband(m; rng = MersenneTwister(2))
  cb = supt_critical(vcov(m; method = RobustBiasCorrected()); rng = MersenneTwister(2))
  @test (b.upper .- m.estimate_bc) ./ m.se_rbc ≈ fill(cb, 4) rtol = 1e-8
  @test_throws ArgumentError supt_critical(Matrix(1.0I, 2, 2); level = 1)
  @test_throws ArgumentError supt_critical(Matrix(1.0I, 2, 2); draws = 10)
  @test_throws DimensionMismatch supt_critical(ones(2, 3))
end

end
