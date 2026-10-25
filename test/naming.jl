module NamingTest
using Test
using KernelSmoothing

@testset "canonical names and compatibility" begin
  for (old, canonical) in ((:kernelsums, :kernel_sums), (:kernelsums!, :kernel_sums!),
                           (:selectbandwidth, :select_bandwidth), (:weightsum, :weight_sum),
                           (:worstcasebias, :worst_case_bias), (:localpolynomial, :local_polynomial),
                           (:smoothermatrix, :smoother_matrix), (:confband, :conf_band))
    # Binding identity preserves downstream methods defined using the old spelling.
    @test getproperty(KernelSmoothing, old) === getproperty(KernelSmoothing, canonical)
  end
  canonical = Honest(2; bias_corrected=true)
  legacy = Honest(2; biascorrected=true)
  @test legacy.bias_corrected == canonical.bias_corrected == true
  @test legacy.biascorrected == legacy.bias_corrected
  @test hasproperty(legacy, :biascorrected)
  @test_throws ArgumentError Honest(2; bias_corrected=true, biascorrected=false)

  x = collect(range(0, 1; length=21))
  d = SmoothingData(x, x .^ 2)
  fit = KernelSmoothing.fit(LocalPolynomialRegression, d; bandwidth=0.3, points=[0.5])
  @test worst_case_bias(fit, 2; bias_corrected=true) ==
        KernelSmoothing.worstcasebias(fit, 2; biascorrected=true)
  @test_throws ArgumentError worst_case_bias(fit, 2; bias_corrected=true, biascorrected=false)
end
end
