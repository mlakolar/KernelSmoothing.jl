module DataTest

using Test
using KernelSmoothing
import StatsAPI

# Deterministic data on a 0.1 grid: many ties, and many observations exactly at
# the edges of windows centred on grid points.
const XS = round.(7 .* sin.(1:500) .+ cos.(3 .* (1:500)); digits = 1)
const YS = cos.(1:500)
const WS = Float64.(mod.(1:500, 4))   # 0, 1, 2, 3: every fourth weight is zero

brute_window(d, x0, h) = findall(xi -> abs((xi - x0) / h) <= 1, d.x)

@testset "construction" begin
  d = SmoothingData(XS)
  @test issorted(d.x)
  @test d.x == XS[d.perm]
  @test sort(d.perm) == 1:500
  @test nobs(d) == 500
  @test nobs === StatsAPI.nobs
  @test effective_nobs(d) == 500
  @test d.y === nothing && d.w === nothing
  # XS holds both -0.0 and 0.0, which are one value; `unique` alone would count two
  @test nunique(d) == length(unique(XS .+ 0.0))
  @test d.values == sort(unique(XS .+ 0.0))
  @test all(d.x[i] == d.values[d.valueindex[i]] for i in eachindex(d.x))
  @test all(all(d.x[d.starts[j]:d.starts[j + 1] - 1] .== d.values[j]) for j in eachindex(d.values))
  @test last(d.starts) == 501
  @test all(d.perm[i] < d.perm[i + 1] for i in 1:499 if d.x[i] == d.x[i + 1])   # ties keep their order

  dy = SmoothingData(XS, YS)
  @test dy.perm == d.perm
  @test dy.y == YS[dy.perm]

  dw = SmoothingData(XS, YS; weights = WS)
  keep = findall(>(0), WS)
  @test nobs(dw) == length(keep)
  @test sort(dw.perm) == keep
  @test dw.x == XS[dw.perm] && dw.y == YS[dw.perm] && dw.w == WS[dw.perm]
  @test dw.sumw ≈ sum(WS)
  @test effective_nobs(dw) ≈ sum(WS)^2 / sum(abs2, WS)
  @test effective_nobs(SmoothingData(XS; weights = fill(2.5, 500))) ≈ 500

  @test SmoothingData([3, 1, 2]).x == [1.0, 2.0, 3.0]
  @test SmoothingData(Float32[0.5, 0.25]).x == [0.25, 0.5]
  @test SmoothingData(1:5).x == 1.0:5.0
  @test SmoothingData(Any[2.0, 1]).x == [1.0, 2.0]
  z = SmoothingData([0.0, -0.0, 1.0])
  @test nunique(z) == 2
  @test !any(signbit, z.x)

  @test sprint(show, SmoothingData([2.0, 1.0, 2.0])) ==
        "SmoothingData(3 observations, 2 distinct values)"
  @test sprint(show, SmoothingData([2.0, 1.0], [0.0, 1.0]; weights = [1, 2])) ==
        "SmoothingData(2 observations, 2 distinct values, weighted, with responses)"
end

@testset "sorting and weight sums" begin
  # the integer-key sort gives the stable sortperm (with -0.0 sorted as 0.0)
  @test SmoothingData(XS).perm == sortperm(XS .+ 0.0)
  keep = findall(>(0), WS)
  @test SmoothingData(XS; weights = WS).perm == keep[sortperm(XS[keep] .+ 0.0)]
  xr = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:2000]
  @test SmoothingData(xr).perm == sortperm(xr)
  @test SmoothingData([3, 1, 2, 1]).perm == [2, 4, 3, 1]
  @test SmoothingData([-2.5, 1e300, -1e-300, 0.0, -0.0, 5e-324, -floatmax()]).perm ==
        sortperm([-2.5, 1e300, -1e-300, 0.0, 0.0, 5e-324, -floatmax()])

  d, dw = SmoothingData(XS), SmoothingData(XS; weights = WS)
  @test d.cumw == 0:500 && weightsum(d) == 500
  @test dw.cumw[end] == dw.sumw && weightsum(dw) ≈ sum(WS)
  for x0 in range(-9, 9; length = 19), h in (0.1, 0.7, 3.0)
    r, rw = window(d, x0, h), window(dw, x0, h)
    @test weightsum(d, r) == length(r)
    @test weightsum(dw, rw) ≈ sum(dw.w[rw]) atol = 1e-12
  end
  @test weightsum(dw, 1:0) == 0 && weightsum(dw, window(dw, 100.0, 1.0)) == 0
end

@testset "invalid input" begin
  @test_throws ArgumentError SmoothingData(Float64[])
  @test_throws ArgumentError SmoothingData([1.0, NaN])
  @test_throws ArgumentError SmoothingData([1.0, Inf])
  @test_throws ArgumentError SmoothingData([1.0, missing])
  @test_throws ArgumentError SmoothingData(["a", "b"])
  @test_throws ArgumentError SmoothingData([1.0, 2.0], [1.0, NaN])
  @test_throws DimensionMismatch SmoothingData([1.0, 2.0], [1.0])
  @test_throws DimensionMismatch SmoothingData([1.0, 2.0]; weights = [1.0])
  @test_throws ArgumentError SmoothingData([1.0, 2.0]; weights = [1.0, -1.0])
  @test_throws ArgumentError SmoothingData([1.0, 2.0]; weights = [0.0, 0.0])
  @test_throws ArgumentError SmoothingData([1.0, 2.0]; weights = [1.0, Inf])
end

@testset "window and nunique" begin
  d = SmoothingData(XS)
  for x0 in range(-9, 9; length = 37), h in (0.05, 0.1, 0.35, 1.0, 3.0, 20.0, Inf)
    r = window(d, x0, h)
    @test collect(r) == brute_window(d, x0, h)
    @test nunique(d, r) == length(unique(d.x[r]))
  end
  @test isempty(window(d, 100.0, 1.0)) && isempty(window(d, -100.0, 1.0))
  @test nunique(d, window(d, 100.0, 1.0)) == 0
  @test window(d, 0, 1) == window(d, 0.0, 1.0)
  @test @inferred(window(d, 0.3, 0.1)) isa UnitRange{Int}
  @test_throws ArgumentError window(d, 0.0, 0.0)
  @test_throws ArgumentError window(d, 0.0, -1.0)
  @test_throws ArgumentError window(d, 0.0, NaN)
  @test_throws ArgumentError window(d, Inf, 1.0)
  @test_throws ArgumentError window(d, NaN, 1.0)
end

@testset "knn_distance" begin
  d = SmoothingData(XS)
  for x0 in (-9.3, -2.25, 0.0, 0.1, 3.05, 7.7, 12.0), k in (1, 2, 5, 21, 100, 500)
    δ = knn_distance(d, x0, k)
    @test δ == sort(abs.(d.x .- x0))[k]
    δ > 0 && @test length(window(d, x0, δ)) >= k
  end
  for x0 in (-9.3, 0.0, 3.05), k in (1, 5, 21, nunique(d))
    δ = knn_distance(d, x0, k; distinct = true)
    @test δ == sort(abs.(d.values .- x0))[k]
    δ > 0 && @test nunique(d, window(d, x0, δ)) >= k
  end
  t = SmoothingData([1.0, 2.0, 2.0, 2.0, 3.5])
  @test knn_distance(t, 2.0, 3) == 0
  @test knn_distance(t, 2.0, 4) == 1
  @test knn_distance(t, 2.0, 2; distinct = true) == 1
  @test knn_distance(t, 2.0, 3; distinct = true) == 1.5
  @test knn_distance(t, 0, 1) == 1
  @test @inferred(knn_distance(d, 0.3, 5)) isa Float64
  @test_throws ArgumentError knn_distance(t, 2.0, 0)
  @test_throws ArgumentError knn_distance(t, 2.0, 6)
  @test_throws ArgumentError knn_distance(t, 2.0, 4; distinct = true)
  @test_throws ArgumentError knn_distance(t, NaN, 1)
end

@testset "weights at extreme scales" begin
  v = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:500]
  k = EpanechnikovKernel()
  hsj = bandwidth(SheatherJones(), SmoothingData(v), k)
  for s in (1e-170, 1e170, 1e-300)
    # products of such weights underflow or overflow, but only their ratios matter
    d = SmoothingData(v; weights = fill(s, 500))
    @test effective_nobs(d) ≈ 500
    @test weightsum(d) ≈ 500s rtol = 1e-12
    @test weightsum(d, 1:10) ≈ 10s rtol = 1e-12
    @test bandwidth(SheatherJones(), d, k) ≈ hsj rtol = 1e-6
    @test pdf(fit(KDE, d; bandwidth = 1.0), 0.3) ≈ pdf(fit(KDE, v; bandwidth = 1.0), 0.3) rtol = 1e-12
  end
  # ordinary weights are stored as given
  @test SmoothingData(v; weights = fill(3.0, 500)).w == fill(3.0, 500)
  # a small range after a large weight keeps its accuracy
  d = SmoothingData(collect(1.0:11.0); weights = [1e12; fill(0.3, 10)])
  @test weightsum(d, 2:11) ≈ 3.0 rtol = 1e-14
  @test weightsum(d, 5:4) == 0
  # values that are not finite as Float64
  @test_throws ArgumentError SmoothingData([1.0, big(10.0)^400])
  @test_throws ArgumentError SmoothingData([1.0, 2.0]; weights = [1.0, big(10.0)^400])
end

@testset "cumulative weights never decrease" begin
  # with these weights the compensated running sum stepped back by a unit in the last place
  w = [2.6e11, 9.634060192370734e12, 96020.14964112596, 7.260040318886021e7, 8.902446084591206e9,
       1.2751176404895992e19, 5.065850480910307e18, 9.559024014758572e7, 4.7133904770661127e11,
       1.3907487889932169e10, 1.4489369595417734e6, 5.379260538324541e19, 0.5908100074150856]
  d = SmoothingData(collect(1.0:13.0); weights = w)
  @test issorted(d.cumw)
  @test all(isapprox(weightsum(d, i:i), w[i]; rtol = 1e-10) for i in 1:13)
end

@testset "weights beyond the range of Float64" begin
  # divided by a power of two in their own type before they become Float64
  d = SmoothingData([1.0, 2.0]; weights = [big"1e-400", big"2e-400"])
  @test nobs(d) == 2
  @test d.w[2] / d.w[1] ≈ 2
  # a weight that vanishes next to the largest one is dropped, as a zero weight is
  d = SmoothingData([1.0, 2.0, 3.0]; weights = [1e300, 1e-300, 2e300])
  @test d.x == [1.0, 3.0] && d.perm == [1, 3]
  @test d.w[2] / d.w[1] ≈ 2
end

end
