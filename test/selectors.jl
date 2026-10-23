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
  r = @test_logs (:warn,) KernelSmoothing._sparse(SheatherJones(), d, g, "a test")
  @test r.details.fallback && r.h == bandwidth(NormalReference(), d, g) && r.rule == SheatherJones()
  @test_throws ArgumentError SheatherJones(method = :plugin)
  @test_throws ArgumentError selectbandwidth(SheatherJones(), SmoothingData([1.0]), g)
  # the default bandwidth of a density estimate
  @test bandwidth(fit(KDE, Q)) == bandwidth(SheatherJones(), d, EpanechnikovKernel())
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
    crit = _lscvcriterion(d, k, 0.01, true)
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
  st = selectbandwidth(LSCV(), SmoothingData(repeat([0.0, 1.0, 2.0, 3.0], 50)), GaussianKernel())
  @test st.details.atbound && isempty(st.details.minimizers)
  @test st.h == st.details.grid[1]
  @test_throws ArgumentError selectbandwidth(LSCV(), SmoothingData(Z), BumpKernel())
  d = SmoothingData(Z)
  # the criterion is flat near its minimum, so binning moves the minimizer by about 1e-4
  @test bandwidth(LSCV(exact = 0), d, GaussianKernel()) ≈ bandwidth(LSCV(), d, GaussianKernel()) rtol = 5e-4
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
    # leave-one-out fits by their definition
    function cv(h)
      s = 0.0
      for i in 1:400
        keep = [j for j in 1:400 if j != i]
        X = [(x[j] - x[i])^l for j in keep, l in 0:p]
        D = ww[keep] .* k.((x[keep] .- x[i]) ./ h)
        G = X' * (D .* X)
        LinearAlgebra.rank(G) < p + 1 && return Inf   # no leave-one-out fit
        β = G \ (X' * (D .* y[keep]))
        s += ww[i] * (y[i] - β[1])^2
      end
      return s / sum(ww)
    end
    for h in (0.6, 0.9)   # above the smallest bandwidth with leave-one-out fits everywhere
      @test _cvcriterion(d, k, h, p, SumsWorkspace()) ≈ cv(h) rtol = 1e-9
    end
    @test _cvcriterion(d, k, 0.05, p, SumsWorkspace()) == Inf
    sel = selectbandwidth(CrossValidation(), d, k; degree = p)
    @test !sel.details.atbound
    # the criterion is continuous but wiggles as observations enter and leave windows
    @test cv(sel.h) <= min(cv(0.99 * sel.h), cv(1.01 * sel.h)) * (1 + 1e-4)
  end
  m = fit(LocalPolynomialRegression, x, y; bandwidth = CrossValidation(), points = [0.0])
  @test m.selection.rule == CrossValidation()
  @test_throws ArgumentError fit(LocalPolynomialRegression, x, y; derivative = 1, degree = 2,
                                 bandwidth = CrossValidation())
  @test_throws ArgumentError fit(LocalPolynomialRegression, x, y; bandwidth = SheatherJones())
  @test_throws ArgumentError fit(KDE, x; bandwidth = CrossValidation())
  @test_throws ArgumentError fit(KDE, x; bandwidth = RuleOfThumb())
end

@testset "root finding and minimization" begin
  r, evals = _findroot(h -> h^3 - 2, 0.0, 3.0)
  @test r ≈ cbrt(2) rtol = 1e-10
  @test evals < 60
  @test _findroot(h -> h - 1, 0.0, 2.0)[1] ≈ 1
  @test _goldenmin(h -> (h - 0.3)^2, 0.0, 1.0) ≈ 0.3 rtol = 1e-5
end

end
