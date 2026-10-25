module RDLocRandTest

using Test
using KernelSmoothing
using Random: Xoshiro

include(joinpath(@__DIR__, "data", "rdlocrand_golden.jl"))

@testset "randomization inference as rdrandinf" begin
  for g in RDLOCRAND
    y = g.y === :Y ? Y : YT
    m = fit(LocalRandomization, X, y; cutoff = g.cutoff, window = g.window, nulltau = g.nulltau, statistic = :all,
            reps = 20000, rng = Xoshiro(1))
    @test (m.ncontrol, m.ntreated) == g.n
    @test collect(m.means) ≈ collect(g.means) rtol = 1e-12
    @test collect(m.sds) ≈ collect(g.sds) rtol = 1e-12
    @test m.observed ≈ g.observed rtol = 1e-10
    @test m.asymptotic ≈ g.asymptotic rtol = 1e-8
    # the randomization p-values agree within the Monte Carlo error of both
    for (p, q) in zip(m.pvalue, g.pvalue)
      @test abs(p - q) <= 4 * sqrt(max(q * (1 - q), 1e-4) * 2 / 20000)
    end
  end
  m = fit(LocalRandomization, X, Y; window = 0.2, rng = Xoshiro(2))
  @test m.statistic == [:diffmeans]
  @test [m.observed[1], m.asymptotic[1]] ≈ DEFAULT_STAT rtol = 1e-10
  @test m.window == (-0.2, 0.2) && m.reps == 1000 && nobs(m) == 81
  t = coeftable(m)
  @test t.statistic == ["diffmeans"] && t.observed == m.observed && t.pvalue == m.pvalue
  @test occursin("LocalRandomization(window (-0.2, 0.2), 40 and 41 observations, diffmeans)", sprint(show, m))
  # the same stream of permutations gives the same p-values
  @test fit(LocalRandomization, X, Y; window = 0.2, rng = Xoshiro(2)).pvalue == m.pvalue
end

@testset "exact Kolmogorov–Smirnov distribution" begin
  # against all the labelings of small samples, with and without ties
  for (n1, y) in ((3, 1.0:7), (5, 1.0:10), (2, 1.0:9), (4, [1.0, 1, 2, 3, 3, 3, 4, 5, 5]))
    n = length(y)
    labels = [digits(b; base = 2, pad = n) .== 1 for b in 0:2^n-1]
    labels = filter(l -> count(l) == n1, labels)
    D = [KernelSmoothing._ksstatistic(y[l], y[.!l]) for l in labels]
    for d in unique(D)
      @test KernelSmoothing._ksexact(y, n1, d) ≈ count(>=(d - 1e-12), D) / length(D)
    end
  end
  @test KernelSmoothing._kslimit(0.0) == 1
  @test KernelSmoothing._kslimit(3.0) ≈ 2 * exp(-18) rtol = 1e-6
  # ties in the statistic: the empirical distribution functions jump together
  @test KernelSmoothing._ksstatistic([1.0, 2.0, 2.0], [2.0, 3.0]) ≈ 1 / 2
end

@testset "checks" begin
  @test_throws DimensionMismatch fit(LocalRandomization, X, Y[1:10]; window = 0.2)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = (0.1, 0.2))
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = 0.2, statistic = :ttest)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = 0.2, reps = 0)
  @test_throws ArgumentError fit(LocalRandomization, [-0.1, 0.1, 0.2], [1.0, 2.0, 3.0]; window = 0.5)
end

end
