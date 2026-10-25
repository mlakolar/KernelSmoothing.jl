# Plot recipes, loaded with RecipesBase (for example by Plots.jl)
module KernelSmoothingRecipesBaseExt

using KernelSmoothing
using KernelSmoothing: Honest
using RecipesBase: RecipesBase, @recipe, @series

const _Fit = Union{LocalPolynomialRegression,PointwiseDensity,LocalPolynomialDensity}

# A pointwise fit: the estimates at the points as a line, over the shaded confidence
# intervals, or the uniform band with `uniform = true`; for regression, `data = true`
# adds the observations.
# (the coverage of Julia 1.10 counts the definitions of the recipes as lines never run)
@recipe function f(m::_Fit; level = 0.95, method = RobustBiasCorrected(), uniform = false, data = false)  # COV_EXCL_LINE
  x = m.points
  lower, upper = if uniform
    b = conf_band(m; level, method)
    b.lower, b.upper
  else
    c = confint(m; level, method)
    c[:, 1], c[:, 2]
  end
  pct = string(round(100 * level; digits = 1))
  endswith(pct, ".0") && (pct = chop(pct; tail = 2))
  kind = method isa Conventional ? "" : method isa Honest ? " honest" : " robust"
  if data && m isa LocalPolynomialRegression
    @series begin
      seriestype := :scatter
      markersize --> 2
      markerstrokewidth --> 0
      markeralpha --> 0.4
      seriescolor --> :gray
      label --> "data"
      m.data.x, m.data.y
    end
  end
  @series begin
    seriestype := :path
    fillrange := lower
    fillalpha --> 0.25
    linealpha := 0
    seriescolor --> 1
    label --> (uniform ? "$pct%$kind band" : "$pct%$kind intervals")
    x, upper
  end
  seriestype --> :path
  seriescolor --> 1
  label --> "estimate"
  xguide --> "x"
  x, coef(m)
end

# A density estimate as a curve over the central 99.9% of its probability
@recipe function f(k::KDE; npoints = 401)  # COV_EXCL_LINE
  grid = range(quantile(k, 0.0005), quantile(k, 0.9995); length = npoints)
  seriestype --> :path
  label --> "density"
  xguide --> "x"
  grid, pdf.(k, grid)
end

# A density estimate in two dimensions as contours over the region of the data
# widened by three standard deviations of the kernel
@recipe function f(k::MultivariateKDE; npoints = 101)  # COV_EXCL_LINE
  length(k) == 2 || throw(ArgumentError("the plot recipe shows estimates in two dimensions, got $(length(k))"))
  X, H = k.data.x, bandwidth(k)
  g1, g2 = (range(minimum(view(X, a, :)) - 3 * sqrt(H[a, a]), maximum(view(X, a, :)) + 3 * sqrt(H[a, a]);
                  length = npoints) for a in 1:2)
  seriestype --> :contour
  xguide --> "x₁"
  yguide --> "x₂"
  # contour wants the values of the second coordinate along the rows
  g1, g2, permutedims(pdf(k, (g1, g2)))
end

end
