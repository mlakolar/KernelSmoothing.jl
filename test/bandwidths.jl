module BandwidthsTest

using Test
using KernelSmoothing
using KernelSmoothing: PiecewisePolynomialKernel, _quantile, _std
using Random
using Statistics

const XR = [7 * sin(1.618i) + 0.3 * cos(2.7i) for i in 1:2000]
const XS = round.(7 .* sin.(1:500) .+ cos.(3 .* (1:500)); digits = 1)   # many ties
# heavy tails: the interquartile range is the smaller scale
const XT = [tan(π * (mod(0.618034i, 1.0) - 0.5)) for i in 1:1000]

@testset "weighted quantiles and standard deviation" begin
  for x in (XR, XS, XT, [3.0], [1.0, 2.0], [2.0, 2.0, 2.0, 5.0])
    d = SmoothingData(x)
    de = SmoothingData(x; weights = fill(2.5, length(x)))   # equal weights
    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.99, 1.0)
      @test _quantile(d, p) ≈ quantile(x, p) rtol = 1e-14   # R's type 7
      @test _quantile(de, p) ≈ quantile(x, p) rtol = 1e-14
    end
    length(x) > 1 && @test _std(d) ≈ std(x) rtol = 1e-13
    length(x) > 1 && @test _std(de) ≈ std(x) rtol = 1e-13
  end
  # reversing the sample reverses the weighted quantiles
  w = 1.0 .+ mod.(1:500, 4)
  d, dr = SmoothingData(XS; weights = w), SmoothingData(-XS; weights = w)
  @test all(_quantile(d, p) ≈ -_quantile(dr, 1 - p) for p in 0:0.05:1)
end

@testset "rules of thumb" begin
  for x in (XR, XT)
    d = SmoothingData(x)
    n = length(x)
    sd, iqr = std(x), quantile(x, 0.75) - quantile(x, 0.25)
    @test bandwidth(NormalReference(), d, GaussianKernel()) ≈
          (4 / 3)^(1 / 5) * min(sd, iqr / 1.349) * n^(-1 / 5) rtol = 1e-13
    @test bandwidth(NormalReference(robust = false), d, GaussianKernel()) ≈
          (4 / 3)^(1 / 5) * sd * n^(-1 / 5) rtol = 1e-13
    @test bandwidth(NormalReference(), d, EpanechnikovKernel()) ≈
          (40 * sqrt(π))^(1 / 5) * min(sd, iqr / 1.349) * n^(-1 / 5) rtol = 1e-13
    # R's bw.nrd0, and its scaling to other kernels in density()
    h0 = 0.9 * min(sd, iqr / 1.34) * n^(-1 / 5)
    @test bandwidth(SilvermanRule(), d, GaussianKernel()) ≈ h0 rtol = 1e-13
    @test bandwidth(SilvermanRule(), d, EpanechnikovKernel()) ≈ h0 * sqrt(5) rtol = 1e-13
    @test bandwidth(SilvermanRule(), d, UniformKernel()) ≈ h0 * sqrt(3) rtol = 1e-13
  end
  s = selectbandwidth(NormalReference(), SmoothingData(XT), EpanechnikovKernel())
  @test s.details.scale == s.details.iqr / 1.349 < s.details.sd   # heavy tails
  @test s.rule == NormalReference() && bandwidth(s) == s.h
  @test sprint(show, s) == "BandwidthSelection($(s.h), NormalReference(robust = true))"
  # weighted samples use the effective sample size
  w = 1.0 .+ mod.(1:2000, 3)
  dw = SmoothingData(XR; weights = w)
  sw = selectbandwidth(NormalReference(), dw, EpanechnikovKernel())
  @test sw.details.n == effective_nobs(dw) < 2000
  # location and scale equivariance
  for rule in (NormalReference(), NormalReference(robust = false), SilvermanRule()),
      k in (EpanechnikovKernel(), GaussianKernel(), TriangularKernel())
    h = bandwidth(rule, SmoothingData(XR), k)
    @test bandwidth(rule, SmoothingData(3.0 .+ 2.5 .* XR), k) ≈ 2.5h rtol = 1e-12
    @test bandwidth(rule, SmoothingData(-XR), k) ≈ h rtol = 1e-12
  end
  # half the data tied: the interquartile range is zero and the standard deviation is used
  xt = vcat(fill(1.0, 600), XR[1:400])
  @test bandwidth(NormalReference(), SmoothingData(xt), GaussianKernel()) ≈
        (4 / 3)^(1 / 5) * std(xt) * 1000^(-1 / 5) rtol = 1e-13
  @test_throws ArgumentError bandwidth(NormalReference(), SmoothingData(fill(2.0, 10)), EpanechnikovKernel())
  @test_throws ArgumentError bandwidth(SilvermanRule(), SmoothingData([2.0]), EpanechnikovKernel())
  k4 = PiecewisePolynomialKernel([-1, 1], [[9/8, 0, -15/8]])   # fourth order: moment(k, 2) = 0
  @test_throws ArgumentError bandwidth(NormalReference(), SmoothingData(XR), k4)
end

@testset "fit with a rule" begin
  f = fit(KDE, XR)
  @test bandwidth(f) == bandwidth(NormalReference(), SmoothingData(XR), EpanechnikovKernel())
  @test f.selection.rule == NormalReference()
  g = fit(KDE, XR; kernel = GaussianKernel(), bandwidth = SilvermanRule())
  @test bandwidth(g) == bandwidth(SilvermanRule(), SmoothingData(XR), GaussianKernel())
  @test fit(KDE, XR; bandwidth = 0.25).selection.rule === nothing
end

end
