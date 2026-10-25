## A precompile workload: the first call of each common estimator would otherwise
## spend seconds compiling

@setup_workload begin
  x = [quantile(Distributions.Normal(), (i - 0.5) / 200) + 0.05 * sin(7i) for i in 1:200]
  y = sin.(x) .+ 0.1 .* cos.(13 .* (1:200))
  @compile_workload begin
    f = fit(KDE, x)
    pdf(f, 0.0); logpdf(f, 0.0); cdf(f, 0.0); quantile(f, 0.5); stderror(f, 0.0)
    pdf.(f, [-1.0, 0.0, 1.0]); cdf.(f, [-1.0, 0.0, 1.0])
    fit(KDE, x[1:50])   # exact sums for Sheather and Jones
    m = fit(LocalPolynomialRegression, x, y; points = [-1.0, 0.0, 1.0])
    confint(m); coeftable(m); vcov(m); conf_band(m; draws = 100)
    confint(fit(LocalPolynomialRegression, x, y; points = [0.0]))
    confint(fit(PointwiseDensity, x; points = [-1.0, 0.0, 1.0]))
    confint(fit(LocalPolynomialDensity, x .- minimum(x); points = [0.0, 1.0, 2.0]))
    k = fit(MultivariateKDE, [x[1:60]'; y[1:60]'])
    pdf(k, [0.0, 0.0]); pdf(k, [x[1:3]'; y[1:3]'])
  end
end
