module RDMultiTest

using Test
using KernelSmoothing

include(joinpath(@__DIR__, "data", "rdmulti_golden.jl"))

# rdmulti's columns: the cutoffs, then "weighted" and "pooled"; Nh and H hold the left
# and the right side of each
function check(m, coefs, B, V, NH, H, W)
  for (k, f) in enumerate(m.fits)
    @test f.estimate ≈ coefs[k] rtol = 1e-7
    @test f.estimate_bc ≈ B[k] rtol = 1e-7
    @test f.se_rbc^2 ≈ V[k] rtol = 1e-7
    @test collect(f.nh) == NH[2k-1:2k]
    @test collect(f.h) ≈ H[2k-1:2k] rtol = 1e-7
  end
  K = length(m.fits)
  if m isa MultipleCutoffs
    @test m.weights ≈ W rtol = 1e-12
    @test m.estimate ≈ coefs[K + 1] rtol = 1e-7
    @test m.estimate_bc ≈ B[K + 1] rtol = 1e-7
    @test m.se_rbc^2 ≈ V[K + 1] rtol = 1e-7
    p = m.pooled
    @test p.estimate ≈ coefs[K + 2] rtol = 1e-7
    @test p.estimate_bc ≈ B[K + 2] rtol = 1e-7
    @test p.se_rbc^2 ≈ V[K + 2] rtol = 1e-7
    @test collect(p.nh) == NH[2K+3:2K+4]
    @test collect(p.h) ≈ H[2K+3:2K+4] rtol = 1e-7
  end
end

@testset "agreement with rdmulti" begin
  m = fit(MultipleCutoffs, X, Y, C; compat = true)
  check(m, RDMC_COEFS, RDMC_B, RDMC_V, RDMC_NH, RDMC_H, RDMC_W)
  mf = fit(MultipleCutoffs, X, YF, C; treatment = TR, compat = true)
  check(mf, RDMCF_COEFS, RDMCF_B, RDMCF_V, RDMCF_NH, RDMCF_H, RDMCF_W)
  @test all(f -> f.fuzzy, mf.fits) && mf.pooled.fuzzy
  s = fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], compat = true)
  check(s, RDMS_COEFS, RDMS_B, RDMS_V, RDMS_NH, RDMS_H, nothing)
  sr = fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [(0, 66), (33, 100)], compat = true)
  check(sr, RDMSR_COEFS, RDMSR_B, RDMSR_V, RDMSR_NH, RDMSR_H, nothing)
end

@testset "several cutoffs" begin
  m = fit(MultipleCutoffs, X, Y, C)
  # each fit is the fit of the units of its cutoff
  idx = findall(==(33), C)
  f = fit(RegressionDiscontinuity, X[idx], Y[idx]; cutoff = 33)
  @test m.fits[1].estimate == f.estimate && m.fits[1].h == f.h
  @test m.cutoffs == [33.0, 66.0]
  @test m.pooled.estimate == fit(RegressionDiscontinuity, X .- C, Y; cutoff = 0).estimate
  # given bandwidths, and observation weights restricted to each cutoff
  w = 1.0 .+ mod.(1:length(X), 3)
  mw = fit(MultipleCutoffs, X, Y, C; weights = w, bandwidth = 10.0)
  fw = fit(RegressionDiscontinuity, X[idx], Y[idx]; cutoff = 33, weights = w[idx], bandwidth = 10.0)
  @test mw.fits[1].estimate ≈ fw.estimate
  @test all(f -> f.h == (10.0, 10.0), mw.fits)
  # a matrix of covariates and clusters are restricted to the units of each cutoff
  Z = hcat(sin.(1:length(X)), cos.(1:length(X)))
  g = mod.(1:length(X), 40)
  mz = fit(MultipleCutoffs, X, Y, C; covariates = Z, cluster = g, bandwidth = 10.0)
  fz = fit(RegressionDiscontinuity, X[idx], Y[idx]; cutoff = 33, covariates = Z[idx, :], cluster = g[idx],
           bandwidth = 10.0)
  @test mz.fits[1].estimate ≈ fz.estimate && mz.fits[1].se ≈ fz.se
  @test_throws DimensionMismatch fit(MultipleCutoffs, X, Y, C; weights = w[1:10])
  # cutoffs of -0.0 and 0.0 are one cutoff
  c0 = [isodd(i) ? -0.0 : 0.0 for i in eachindex(X)]
  m0 = fit(MultipleCutoffs, X .- 33, Y, ifelse.(C .== 33, c0, 33.0))
  @test length(m0.cutoffs) == 2 && m0.fits[1].estimate ≈ m.fits[1].estimate
  # the weighted average and its intervals
  @test coef(m) == [m.estimate] && coef(m; method = RobustBiasCorrected()) == [m.estimate_bc]
  @test stderror(m) == [m.se]
  ci = confint(m; level = 0.9)
  @test ci[1] < m.estimate_bc < ci[2]
  @test ci[2] - ci[1] ≈ 2 * 1.6448536269514722 * m.se_rbc
  t = coeftable(m)
  @test t.cutoff == ["33.0", "66.0", "weighted", "pooled"]
  @test t.estimate ≈ vcat([f.estimate for f in m.fits], m.estimate, m.pooled.estimate)
  @test t.weight[1:2] == m.weights && sum(m.weights) ≈ 1
  @test occursin("weighted", sprint(show, MIME("text/plain"), m))
  @test sprint(show, m) == "MultipleCutoffs(2 cutoffs, 1200 observations)"
  s = fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66])
  @test coeftable(s).cutoff == ["33.0", "66.0"]
  @test occursin("cumulative", sprint(show, MIME("text/plain"), s))
  @test_throws ArgumentError fit(MultipleCutoffs, X, Y, fill(33.0, length(X)))
  @test_throws ArgumentError fit(MultipleCutoffs, X, Y, C; cutoff = 1)
  @test_throws DimensionMismatch fit(MultipleCutoffs, X, Y, C[1:10])
  @test_throws ArgumentError fit(CumulativeCutoffs, X2, Y2; cutoffs = [66, 33])
  @test_throws ArgumentError fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [(40, 66), (33, 100)])
  @test_throws DimensionMismatch fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [(0, 66)])
  # ranges as a matrix with one row per cutoff, as rdms's rangemat
  sm = fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [0 66; 33 100])
  @test sm.ranges == [(0.0, 66.0), (33.0, 100.0)]
  @test_throws DimensionMismatch fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [0 33 66; 33 66 100])
  @test_throws ArgumentError fit(CumulativeCutoffs, X2, Y2; cutoffs = [33, 66], ranges = [(0,), (33, 100)])
  @test_throws ArgumentError confint(m; method = Honest(1.0))
end

end
