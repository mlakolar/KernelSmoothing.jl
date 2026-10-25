module RDPowerTest

using Test
using KernelSmoothing
using Distributions: Normal, cdf, ccdf, quantile

include(joinpath(@__DIR__, "data", "rdpower_golden.jl"))

@testset "power calculations" begin
  m = fit(RegressionDiscontinuity, X, Y; compat = true)
  @test m.se_rbc ≈ SE[1] rtol = 1e-8
  # rdpower's power of the robust test, at another level, and for twice the sample
  @test rd_power(m; tau = 0.3) ≈ POWER[1] rtol = 1e-8
  @test rd_power(m; tau = 0.5, alpha = 0.1) ≈ POWER[2] rtol = 1e-8
  @test rd_power(m; tau = 0.3, n = 2 * nobs(m)) ≈ POWER[3] rtol = 1e-8
  # rdsampsi's sample size
  @test rd_samplesize(m; tau = 0.3) == SAMPSI[1]
  @test rd_samplesize(m; tau = 0.2, power = 0.9, alpha = 0.1) == SAMPSI[2]
  # the power is the level at no effect, symmetric in the sign of the effect, and
  # reaches the target at the sample size, but not one observation earlier
  @test rd_power(m; tau = 0) ≈ 0.05
  @test rd_power(m; tau = -0.3) ≈ rd_power(m; tau = 0.3)
  n = rd_samplesize(m; tau = 0.3)
  @test rd_power(m; tau = 0.3, n) >= 0.8 > rd_power(m; tau = 0.3, n = n - 1)
  z = quantile(Normal(), 0.975)
  @test rd_power(m; tau = 0.4) ≈ ccdf(Normal(), z - 0.4 / m.se_rbc) + cdf(Normal(), -z - 0.4 / m.se_rbc)
  @test_throws ArgumentError rd_power(m; tau = 0.3, alpha = 1.5)
  @test_throws ArgumentError rd_power(m; tau = 0.3, n = 0)
  @test_throws ArgumentError rd_samplesize(m; tau = 0)
  @test_throws ArgumentError rd_samplesize(m; tau = 0.3, power = 0.01)
end

end
