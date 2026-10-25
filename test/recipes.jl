module RecipesTest

using Test
using KernelSmoothing
using RecipesBase: RecipesBase

const I = 1:400
const X = round.(mod.(I .* 0.6180339887498949, 1), digits = 2)
const Y = sin.(2π .* X) .+ 0.3 .* sin.(7.3 .* I)

# a plotting package such as Plots says which attributes it supports
RecipesBase.is_key_supported(::Symbol) = true
recipe(m; kwargs...) = RecipesBase.apply_recipe(Dict{Symbol,Any}(kwargs...), m)

@testset "plot recipes" begin
  @test Base.get_extension(KernelSmoothing, :KernelSmoothingRecipesBaseExt) !== nothing
  m = fit(LocalPolynomialRegression, X, Y; bandwidth = 0.2)
  # the intervals as a shaded area, then the estimates
  band, line = recipe(m)
  ci = confint(m)
  @test band.args == (m.points, ci[:, 2])
  @test band.plotattributes[:fillrange] == ci[:, 1]
  @test band.plotattributes[:label] == "95% robust intervals"
  @test line.args == (m.points, coef(m))
  @test line.plotattributes[:label] == "estimate"
  # conventional intervals, the uniform band and the data
  _, band, _ = recipe(m; method = Conventional(), level = 0.9, uniform = true, data = true)
  @test band.plotattributes[:label] == "90% band"
  @test all(band.args[2] .>= confint(m; method = Conventional(), level = 0.9)[:, 2])
  scatter, _, _ = recipe(m; data = true)
  @test scatter.plotattributes[:seriestype] == :scatter
  @test sort(scatter.args[1]) == sort(X)
  @test recipe(m; method = Honest(10.0))[1].plotattributes[:label] == "95% honest intervals"
  @test recipe(m; level = 0.975)[1].plotattributes[:label] == "97.5% robust intervals"
  # densities: pointwise fits, and the estimate of a KDE as a curve
  for d in (fit(PointwiseDensity, X; points = 0.1:0.1:0.9), fit(LocalPolynomialDensity, X; points = 0.1:0.1:0.9))
    band, line = recipe(d; data = true)
    @test line.args == (d.points, coef(d))
    @test band.plotattributes[:fillrange] == confint(d)[:, 1]
  end
  f = fit(KDE, X)
  (curve,) = recipe(f; npoints = 101)
  @test length(curve.args[1]) == 101
  @test curve.args[2] == pdf.(f, curve.args[1])
  b = fit(BoundedKDE, X; support = (0, 1))
  (curve,) = recipe(b; npoints = 51)
  @test curve.args[1][1] == 0 && curve.args[1][end] == 1
  @test curve.args[2] == pdf.(b, curve.args[1])
  (curve,) = recipe(fit(BoundedKDE, X; support = (0, Inf)))
  @test curve.args[1][1] == 0 && isfinite(curve.args[1][end])
end

end
