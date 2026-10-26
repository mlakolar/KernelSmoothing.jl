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
    pre = KernelSmoothing._lrprep(y)
    D = [KernelSmoothing._lrstatistic(:ksmirnov, pre, l) for l in labels]
    for d in unique(D)
      @test KernelSmoothing._ksexact(y, n1, d) ≈ count(>=(d - 1e-12), D) / length(D)
    end
  end
  # small p-values keep their relative accuracy: separated samples of 50
  @test KernelSmoothing._ksexact(1.0:100, 50, 1.0) ≈ 2 / binomial(big(100), 50) rtol = 1e-8
  x = vcat(-(50:-1:1) ./ 100, (0:49) ./ 100)
  @test fit(LocalRandomization, x, x; window = 0.6, statistic = :ksmirnov).asymptotic[1] ≈
        2 / binomial(big(100), 50) rtol = 1e-8
  # the limiting distribution on both sides of t = 1
  @test KernelSmoothing._kslimit(0.0) == 1
  @test KernelSmoothing._kslimit(0.01) == 1
  @test KernelSmoothing._kslimit(0.5) ≈ 0.9639452436648751 rtol = 1e-10
  @test KernelSmoothing._kslimit(1.0) ≈ 0.26999967167735456 rtol = 1e-10
  @test KernelSmoothing._kslimit(prevfloat(1.0)) ≈ KernelSmoothing._kslimit(1.0) rtol = 1e-10
  @test KernelSmoothing._kslimit(3.0) ≈ 2 * exp(-18) rtol = 1e-6
  # a tiny statistic in a large sample has a p-value of one
  x = vcat(-(1:120) ./ 1000, (0:120) ./ 1000)
  y = ones(241)
  y[1] = y[121] = 0
  @test fit(LocalRandomization, x, y; window = 0.2, statistic = :ksmirnov).asymptotic[1] ≈ 1
  # ties in the statistic: the empirical distribution functions jump together
  pre = KernelSmoothing._lrprep([1.0, 2.0, 2.0, 2.0, 3.0])
  @test KernelSmoothing._lrstatistic(:ksmirnov, pre, [true, true, true, false, false]) ≈ 1 / 2
end

@testset "checks" begin
  @test_throws DimensionMismatch fit(LocalRandomization, X, Y[1:10]; window = 0.2)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = (0.1, 0.2))
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = 0.2, statistic = :ttest)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = 0.2, reps = 0)
  @test_throws ArgumentError fit(LocalRandomization, [-0.1, 0.1, 0.2], [1.0, 2.0, 3.0]; window = 0.5)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = 0.2, nulltau = NaN)
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = [0.1])
  @test_throws ArgumentError fit(LocalRandomization, X, Y; window = (-0.1, 0.1, 0.2))
end

end
