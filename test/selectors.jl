module SelectorsTest

using Test
using KernelSmoothing
using KernelSmoothing: _findroot, _goldenmin, _lscvcriterion, _cvcriterion, _psi
using Distributions: Normal, quantile
using LinearAlgebra: LinearAlgebra

# a skewed sample: normal quantiles at a low-discrepancy sequence, a third of them shifted
const Q = [quantile(Normal(), mod(i * 0.6180339887498949, 1)) + 0.5 * (mod(i, 3) == 0) * 2.5 for i in 1:300]
# a less regular sample, for cross-validation
const Z = [quantile(Normal(), mod(i * 0.7548776662466927, 1)) + 0.4 * sin(i^2) for i in 1:400]

# Reference bandwidths from R 4.5.1: bw.SJ(Q, nb = 2000000L, method = "dpi") and
# bw.SJ(Q, nb = 2000000L, tol = 1e-12); at this binning R is converged to about 1e-7
const R_SJ_DPI = 0.433791922029472
const R_SJ_STE = 0.434594593895941

# ψ̂ᵣ(g) and the Sheather–Jones equation by their definitions
phi4(u) = (u^4 - 6u^2 + 3) * exp(-u^2 / 2) / sqrt(2π)
phi6(u) = (u^6 - 15u^4 + 45u^2 - 15) * exp(-u^2 / 2) / sqrt(2π)
psi(x, f, g, r) = sum(f((xi - xj) / g) for xi in x, xj in x) / (length(x) * (length(x) - 1) * g^(r + 1))

@testset "Sheather and Jones" begin
  d = SmoothingData(Q)
  g = GaussianKernel()
  @test bandwidth(SheatherJones(method = :dpi, exact = 1000), d, g) ≈ R_SJ_DPI rtol = 2e-6
  # R rounds the constant 1.35733 of α₂ to 1.357, which moves its bandwidth by 4e-5
  s = selectbandwidth(SheatherJones(exact = 1000), d, g)
  @test s.h ≈ R_SJ_STE rtol = 1e-4
  # the equation solved by its definition, with α₂'s constant unrounded
  n = length(Q)
  σ = min(KernelSmoothing._std(d), (quantile(Q, 0.75) - quantile(Q, 0.25)) / 1.349)
  a, b = 1.24σ * n^(-1 / 7), 1.23σ * n^(-1 / 9)
  α2 = (6 / sqrt(2π) / (1 / (2sqrt(π))))^(1 / 7) * (psi(Q, phi4, a, 4) / -psi(Q, phi6, b, 6))^(1 / 7)
  F(h) = (1 / (2sqrt(π) * n * psi(Q, phi4, α2 * h^(5 / 7), 4)))^(1 / 5) - h
  @test abs(F(s.h)) < 1e-9 * s.h
  @test s.details.alpha2 ≈ α2 rtol = 1e-12
  @test s.details.exact && !s.details.fallback
  # binned sums agree with exact ones, and are the default above 100 observations
  @test bandwidth(SheatherJones(exact = 0), d, g) ≈ s.h rtol = 1e-5
  @test !selectbandwidth(SheatherJones(), d, g).details.exact
  @test bandwidth(SheatherJones(method = :dpi, exact = 0), d, g) ≈ R_SJ_DPI rtol = 1e-5
  @test _psi(d, 4, 0.3, false) ≈ _psi(d, 4, 0.3, true) rtol = 1e-4
  # other kernels get the canonical bandwidth ratio, (R(k) / μ₂(k)²)^(1/5) over the Gaussian one
  for k in (EpanechnikovKernel(), BiweightKernel(), UniformKernel())
    ratio = (roughness(k) / moment(k, 2)^2 * 2sqrt(π))^(1 / 5)
    @test bandwidth(SheatherJones(exact = 1000), d, k) ≈ ratio * s.h rtol = 1e-9
  end
  # location and scale, and equal weights
  @test bandwidth(SheatherJones(exact = 1000), SmoothingData(7 .- 3 .* Q), g) ≈ 3 * s.h rtol = 1e-9
  @test bandwidth(SheatherJones(exact = 1000), SmoothingData(Q; weights = fill(2.0, n)), g) ≈ s.h rtol = 1e-12
  # the binned path is equivariant as well, to the accuracy of its root finding
  sb = bandwidth(SheatherJones(), d, g)
  @test bandwidth(SheatherJones(), SmoothingData(7 .- 3 .* Q), g) ≈ 3 * sb rtol = 1e-7
  # a sample too sparse for the estimates falls back to the normal reference, with a warning
  r = @test_logs (:warn,) KernelSmoothing._sparse(SheatherJones(), d, g, "a test", (fallback = true,))
  @test r.details.fallback && r.h == bandwidth(NormalReference(), d, g) && r.rule == SheatherJones()
  # one shape of the details in every case, so that the default density estimate is inferable
  @inferred selectbandwidth(SheatherJones(), d, g)
  @inferred selectbandwidth(SheatherJones(method = :dpi), d, g)
  @inferred fit(KDE, Q)
  @test_throws ArgumentError SheatherJones(method = :plugin)
  @test_throws ArgumentError selectbandwidth(SheatherJones(), SmoothingData([1.0]), g)
  # the default bandwidth of a density estimate
  @test bandwidth(fit(KDE, Q)) == bandwidth(SheatherJones(), d, EpanechnikovKernel())
end

@testset "small positive definite solves" begin
  for p in 0:3
    A = [1.0 / (a + b - 1) for a in 1:p+1, b in 1:p+1] + 0.1 * LinearAlgebra.I   # a shifted Hilbert matrix
    e = zeros(p + 1)
    @test KernelSmoothing._solvefirst!(e, copy(A))
    @test e ≈ A \ [j == 1 ? 1.0 : 0.0 for j in 1:p+1] rtol = 1e-12
  end
  @test !KernelSmoothing._solvefirst!(zeros(2), [1.0 2.0; 2.0 1.0])
end

@testset "binned sums for Sheather and Jones" begin
  # the FFT against the discrete Fourier transform, and the lagged products of the bins
  for N in (1, 2, 16, 128)
    z = [complex(sin(3j), cos(7j)) for j in 1:N]
    @test KernelSmoothing._fft!(copy(z)) ≈ [sum(z[j] * cispi(-2 * (j - 1) * (k - 1) / N) for j in 1:N) for k in 1:N]
  end
  @test_throws ArgumentError KernelSmoothing._fft!(zeros(ComplexF64, 3))
  b = [abs(sin(5k)) for k in 1:37]
  @test KernelSmoothing._lagproducts(b) ≈ [sum(b[k] * b[k + m] for k in 1:37-m) for m in 0:36]
  # a far outlier would need a grid of more than 2^20 points, so that the data are binned for each g
  x = [Q; 1e7]
  @test bandwidth(SheatherJones(exact = 0), SmoothingData(x), GaussianKernel()) ≈
        bandwidth(SheatherJones(), SmoothingData(x), GaussianKernel()) rtol = 1e-4
end

# a kernel defined outside the package, without a convolution
struct BumpKernel <: AbstractKernel end
(::BumpKernel)(u::Real) = abs(u) < 1 ? 0.75 * (1 - u^2) : 0.0
KernelSmoothing.support(::BumpKernel) = (-1.0, 1.0)
KernelSmoothing.moment(::BumpKernel, j::Integer) = j == 0 ? 1.0 : j == 2 ? 0.2 : 0.0
KernelSmoothing.roughness(::BumpKernel) = 0.6

@testset "least-squares cross-validation" begin
  for (k, w) in ((EpanechnikovKernel(), nothing), (GaussianKernel(), nothing),
                 (TriangularKernel(), 1.0 .+ mod.(1:400, 3)), (GaussianKernel(), 1.0 .+ mod.(1:400, 3)))
    ww = w === nothing ? ones(400) : w
    d = SmoothingData(Z; weights = w)
    W, S2 = sum(ww), sum(abs2, ww)
    kk = k isa GaussianKernel ? convolve(k, k) : convolve(k, k)
    # the criterion by its definition
    function cv(h)
      s1 = sum(ww[i] * ww[j] * kk((Z[i] - Z[j]) / h) for i in 1:400, j in 1:400) / (h * W^2)
      s2 = sum(ww[i] * ww[j] * k((Z[i] - Z[j]) / h) for i in 1:400, j in 1:400 if i != j) / (h * (W^2 - S2))
      return s1 - 2s2
    end
    crit = _lscvcriterion(d, k, true, 0.01)
    for h in (0.1, 0.35, 0.9)
      @test crit(h) ≈ cv(h) rtol = 1e-9
    end
    s = selectbandwidth(LSCV(), d, k)
    @test !s.details.atbound
    @test s.h ≈ s.details.minimizers[end] rtol = 0.15
    @test cv(s.h) <= min(cv(0.99 * s.h), cv(1.01 * s.h)) + 1e-12   # a local minimizer
    @test length(s.details.grid) == length(s.details.criterion) == 60
  end
  # with heavy ties the criterion falls as h shrinks, without an interior minimum
  st = @test_logs (:warn, r"end of the grid") selectbandwidth(LSCV(), SmoothingData(repeat([0.0, 1.0, 2.0, 3.0], 50)),
                                                            GaussianKernel())
  @test st.details.atbound && isempty(st.details.minimizers)
  @test st.h == st.details.grid[1]
  @test_throws ArgumentError selectbandwidth(LSCV(), SmoothingData(Z), BumpKernel())
  d = SmoothingData(Z)
  # binned once on a fine grid, the minimizer moves by about 1e-5
  @test bandwidth(LSCV(exact = 0), d, GaussianKernel()) ≈ bandwidth(LSCV(), d, GaussianKernel()) rtol = 2e-5
  crit0, crit = _lscvcriterion(d, GaussianKernel(), true, 0.05), _lscvcriterion(d, GaussianKernel(), false, 0.05)
  for h in (0.05, 0.2, 1.0)
    @test crit(h) ≈ crit0(h) rtol = 1e-5
  end
  @test_throws ArgumentError LSCV(gridsize = 2)
  @test_throws ArgumentError selectbandwidth(LSCV(), d, derivative(GaussianKernel(), 2))
end

@testset "cross-validation for regression" begin
  x = Z
  y = sin.(2 .* x) .+ 0.5 .* cos.(13.7 .* (1:400))
  for (k, p, w) in ((EpanechnikovKernel(), 1, nothing), (TriangularKernel(), 0, nothing),
                    (EpanechnikovKernel(), 2, 1.0 .+ mod.(1:400, 3)))
    d = SmoothingData(x, y; weights = w)
    ww = w === nothing ? ones(400) : w
    # the observations that cross-validation scores: all but the far-out isolated ones
    t = [knn_distance(d, xv, p + 2; distinct = true) for xv in d.values]
    q1, q3 = quantile(t, 0.25), quantile(t, 0.75)
    scored = (t .<= q3 + 3 * (q3 - q1))[d.valueindex][invperm(d.perm)]
    # leave-one-out fits by their definition, over the observations in `keepers`
    function cv(h, keepers = trues(400))
      s = 0.0
      for i in 1:400
        keepers[i] || continue
        keep = [j for j in 1:400 if j != i]
        X = [(x[j] - x[i])^l for j in keep, l in 0:p]
        D = ww[keep] .* k.((x[keep] .- x[i]) ./ h)
        G = X' * (D .* X)
        LinearAlgebra.rank(G) < p + 1 && return Inf   # no leave-one-out fit
        β = G \ (X' * (D .* y[keep]))
        s += ww[i] * (y[i] - β[1])^2
      end
      return s / sum(ww[keepers])
    end
    for h in (0.6, 0.9)   # above the smallest bandwidth with leave-one-out fits everywhere
      @test _cvcriterion(d, k, h, p, SumsWorkspace()) ≈ cv(h) rtol = 1e-9
    end
    @test _cvcriterion(d, k, 0.05, p, SumsWorkspace()) == Inf
    sel = selectbandwidth(CrossValidation(), d, k; degree = p)
    @test !sel.details.atbound
    @test sel.details.scored == count(scored)
    # the criterion is continuous but wiggles as observations enter and leave windows
    cvs(h) = cv(h, scored)
    @test cvs(sel.h) <= min(cvs(0.99 * sel.h), cvs(1.01 * sel.h)) * (1 + 1e-4)
  end
  m = fit(LocalPolynomialRegression, x, y; bandwidth = CrossValidation(), points = [0.0])
  @test m.selection.rule == CrossValidation()
  @test_throws ArgumentError fit(LocalPolynomialRegression, x, y; derivative = 1, degree = 2,
                                 bandwidth = CrossValidation())
  @test_throws ArgumentError fit(LocalPolynomialRegression, x, y; bandwidth = SheatherJones())
  @test_throws ArgumentError fit(KDE, x; bandwidth = CrossValidation())
  @test_throws ArgumentError fit(KDE, x; bandwidth = RuleOfThumb())
end

# the observations, in sorted order, that cross-validation scores: all but the far-out
# isolated ones
function cvscored(d, p)
  t = [knn_distance(d, xv, p + 2; distinct = true) for xv in d.values]
  q1, q3 = quantile(t, 0.25), quantile(t, 0.75)
  return (t .<= q3 + 3 * (q3 - q1))[d.valueindex]
end

@testset "grid minimization" begin
  # an interior local minimum at 2 and a lower end at 10
  f(h) = h < 5 ? (h - 2)^2 + 1 : 0.5 - 0.01h
  h, det = KernelSmoothing._gridminimize(f, collect(1.0:10.0); largest = false)
  @test h == 10 && det.atbound && det.minimizers == [2.0]
  h, det = KernelSmoothing._gridminimize(f, collect(1.0:10.0); largest = true)   # Sheather's rule for LSCV
  @test h ≈ 2 atol = 1e-5
  @test !det.atbound
  # a refinement that is worse than its grid value is not kept
  spike(h) = h == 2 ? 0.0 : 1 + (h - 2)^2
  @test KernelSmoothing._gridminimize(spike, collect(1.0:5.0); largest = false)[1] == 2
  @test_throws ArgumentError KernelSmoothing._gridminimize(h -> Inf, [1.0, 2.0, 3.0]; largest = false)
  # cross-validation takes the global minimum, also for linear regression functions
  # whose criterion falls towards the end of the grid
  for y in (Z .+ 0.5 .* sin.(1e4 .* (1:400)), 2 .* Z .+ 0.3 .* cos.(7e3 .* (1:400)), sin.(2 .* Z) .+ 0.5 .* cos.(13.7 .* (1:400)))
    d = SmoothingData(Z, y)
    sel = selectbandwidth(CrossValidation(), d, EpanechnikovKernel())
    @test _cvcriterion(d, EpanechnikovKernel(), sel.h, 1, SumsWorkspace(), cvscored(d, 1)) <=
          minimum(filter(isfinite, sel.details.criterion)) * (1 + 1e-12)
  end
  # an outlier inflates the standard deviation, but not the lower end of the LSCV grid
  for k in (EpanechnikovKernel(), GaussianKernel())
    @test bandwidth(LSCV(), SmoothingData([Z; 1e7]), k) ≈ bandwidth(LSCV(), SmoothingData(Z), k) rtol = 0.05
  end
  # binned Gaussian sums: a grid over the range of the outlier would need more than
  # 2^20 points, so that the data are binned for each bandwidth, run by run
  @test bandwidth(LSCV(exact = 0), SmoothingData([Z; 1e7]), GaussianKernel()) ≈
        bandwidth(LSCV(exact = 0), SmoothingData(Z), GaussianKernel()) rtol = 0.05
end

@testset "Sheather and Jones with outliers and heavy tails" begin
  k = GaussianKernel()
  # a far outlier: a grid over the whole range would need 2^60 points
  x = [Q; 1e14]
  @test bandwidth(SheatherJones(), SmoothingData(x), k) ≈ bandwidth(SheatherJones(exact = 10^6), SmoothingData(x), k) rtol = 1e-5
  @test bandwidth(fit(KDE, x)) ≈ bandwidth(SheatherJones(), SmoothingData(x), EpanechnikovKernel())
  # heavy tails: Cauchy quantiles, of which the dense center and the stretches with
  # enough pairs are binned and the sparse tails are summed exactly
  c = [tan(π * (mod(0.6180339887498949i, 1) - 0.5)) for i in 1:2000]
  d = SmoothingData(c)
  @test bandwidth(SheatherJones(), d, k) ≈ bandwidth(SheatherJones(exact = 10^6), d, k) rtol = 1e-5
  p = KernelSmoothing._LaggedBins(d, 1e-3, 30.0)
  @test length(p.cs) > 1
  for (r, g) in ((4, 0.3), (6, 0.5), (4, 2.0))
    @test _psi(p, r, g) ≈ _psi(d, r, g, true) rtol = 1e-5
  end
end

@testset "root finding and minimization" begin
  r, evals = _findroot(h -> h^3 - 2, 0.0, 3.0)
  @test r ≈ cbrt(2) rtol = 1e-10
  @test evals < 60
  @test _findroot(h -> h - 1, 0.0, 2.0)[1] ≈ 1
  @test _goldenmin(h -> (h - 0.3)^2, 0.0, 1.0) ≈ 0.3 rtol = 1e-5
end

@testset "Sheather and Jones at extreme scales and on clustered data" begin
  # the equations are solved for the standardized data, so that neither tiny nor huge
  # scales underflow or overflow
  k = GaussianKernel()
  for method in (:ste, :dpi), exact in (100, 1000)
    rule = SheatherJones(; method, exact)
    h = bandwidth(rule, SmoothingData(Z), k)
    for s in (1e-44, 1e-150, 1e150)
      @test bandwidth(rule, SmoothingData(s .* Z), k) ≈ s * h rtol = 1e-7
    end
  end
  # the root of tight clusters lies far below the pilots, where the grid of the binned
  # estimates is refined
  x = [0.01 .* Z; 10 .+ 0.01 .* reverse(Z)]
  @test bandwidth(SheatherJones(), SmoothingData(x), k) ≈
        bandwidth(SheatherJones(; exact = 10^4), SmoothingData(x), k) rtol = 1e-5
end

@testset "wide runs in several binned stretches, and cross-validation in one pass" begin
  # a dense run of 1.5 million bins, covered by several binned stretches, whose pairs
  # across their boundaries are summed exactly
  d = SmoothingData([1500 * mod(0.6180339887498949i, 1) for i in 1:50000])
  p = KernelSmoothing._LaggedBins(d, 1e-3, 30.0)
  @test length(p.parts) >= 5 && isempty(p.rest)
  for (r, g) in ((4, 0.2), (6, 0.3))
    @test _psi(p, r, g) ≈ _psi(d, r, g, true) rtol = 1e-4
  end
  # a scale mixture, whose sparse tails are summed exactly
  x = [isodd(i) ? quantile(Normal(), mod(0.6180339887498949i, 1)) :
                  1000 * quantile(Normal(), mod(0.7548776662466927i, 1)) for i in 1:20000]
  dm = SmoothingData(x)
  pm = KernelSmoothing._LaggedBins(dm, 1e-3, 30.0)
  @test !isempty(pm.rest)
  @test _psi(pm, 4, 0.1) ≈ _psi(dm, 4, 0.1, true) rtol = 1e-4
  # the sums with and without the responses in one pass agree with separate passes
  xs = [mod(0.6180339887498949i, 1) for i in 1:3000]
  dy = SmoothingData(xs, sin.(2π .* xs); weights = 1 .+ mod.(1:3000, 3))
  for k in (EpanechnikovKernel(), TriangularKernel(), GaussianKernel()), h in (0.003, 0.03, 0.3), p in (1, 2)
    S, T = KernelSmoothing._sumsyz(dy, k, h, p, SumsWorkspace())
    @test S ≈ kernelsums(dy, k, h; kmax = 2p) rtol = 1e-12
    @test T ≈ kernelsums(dy, k, h; kmax = p, z = dy.y) rtol = 1e-12
  end
end

@testset "cross-validation with isolated observations and binned Gaussian sums" begin
  k = EpanechnikovKernel()
  x = [mod(0.6180339887498949i, 1) for i in 1:1000]
  y = sin.(2π .* x) .+ 0.3 .* cos.(37.1 .* (1:1000))
  h0 = bandwidth(CrossValidation(), SmoothingData(x, y), k)
  # one isolated observation is left out of the criterion and does not set the floor
  s = selectbandwidth(CrossValidation(), SmoothingData([x; 10.0], [y; 0.0]), k)
  @test s.h ≈ h0 rtol = 0.02
  @test s.details.scored < 1001 && s.details.grid[1] < 0.05
  # with a floor that binds, the rule warns
  x2 = x[1:200]
  @test_logs (:warn, r"lower end of its grid") selectbandwidth(CrossValidation(), SmoothingData(x2, sin.(80π .* x2)), k)
  # Gaussian kernels: binned sums above `exact` observations, close to the exact ones
  g = GaussianKernel()
  xs = [mod(0.7548776662466927i, 1) for i in 1:3000]
  ys = sin.(2π .* xs) .+ 0.3 .* cos.(41.3 .* (1:3000))
  d = SmoothingData(xs, ys; weights = 1 .+ mod.(1:3000, 3))
  for h in (0.03, 0.1, 0.3)
    Sb, Tb = KernelSmoothing._binnedsumsyz(d, g, h, 1)
    Se = kernelsums(d, g, h; kmax = 2)
    Te = kernelsums(d, g, h; kmax = 1, z = d.y)
    @test all(abs.(Sb .- Se) .<= 5e-4 .* maximum(abs.(Se); dims = 2))
    @test all(abs.(Tb .- Te) .<= 5e-4 .* maximum(abs.(Te); dims = 2))
    @test _cvcriterion(d, g, h, 1, SumsWorkspace(), trues(3000), true) ≈ _cvcriterion(d, g, h, 1, SumsWorkspace()) rtol = 2e-4
  end
  @test bandwidth(CrossValidation(), d, g) ≈ bandwidth(CrossValidation(exact = 10^4), d, g) rtol = 5e-3
end

@testset "least-squares cross-validation on clusters, lognormal and rounded data" begin
  q(n) = [quantile(Normal(), mod(i * 0.6180339887498949, 1)) for i in 1:n]
  # compact kernels give local minimizers at the scale of the gaps between clusters,
  # far above the lowest local minimum; Sheather's largest local minimizer took them
  x2 = [q(100); 10 .+ reverse(q(100))]
  s2 = selectbandwidth(LSCV(), SmoothingData(x2), EpanechnikovKernel())
  @test s2.details.minimizers[end] > 10 && s2.h < 2
  x5 = vcat([10j .+ 0.1 .* q(60) for j in 1:5]...)
  @test bandwidth(LSCV(), SmoothingData(x5), EpanechnikovKernel()) < 0.2
  # lognormal data: the minimizer lies below the grid, which extends downward
  sl = @test_logs selectbandwidth(LSCV(), SmoothingData(exp.(3 .* q(2000))), GaussianKernel())
  @test !sl.details.atbound && length(sl.details.grid) > 60
  # rounded data: the criterion falls without bound below the rounding, at the lower end
  # of the grid, and the largest local minimizer stays
  sr = selectbandwidth(LSCV(), SmoothingData(round.(q(1000); digits = 1)), GaussianKernel())
  @test sr.h > 0.3 && length(sr.details.grid) == 60
end

end
