# [Local polynomial regression](@id regression-page)

`fit(LocalPolynomialRegression, x, y)` fits local polynomials at a set of points
and estimates the regression function or one of its derivatives, with
conventional and robust bias-corrected standard errors and confidence intervals
(Calonico, Cattaneo and Farrell 2018), computed as in nprobust.

```julia
using KernelSmoothing

x = rand(2000); y = sin.(2π .* x) .+ 0.3 .* randn(2000)
m = fit(LocalPolynomialRegression, x, y; points = range(0, 1; length = 21))
coef(m)                                  # the estimates at the points
confint(m)                               # robust bias-corrected 95% intervals
confint(m; method = Conventional())      # intervals that ignore the bias
predict(m, [0.25, 0.5])                  # estimates at new points
coeftable(m)                             # everything, as a table of columns
fit(LocalPolynomialRegression, x, y; degree = 2, derivative = 1)   # the derivative
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["locpoly.jl"]
```

## Uniform bands

[`vcov`](@ref) gives the covariances of the estimates at the points, and
[`confband`](@ref) a sup-t band that covers the curve at all points at once, for
regression and density fits alike.

```julia
m = fit(LocalPolynomialRegression, x, y; points = range(0, 1; length = 50))
b = confband(m)              # robust bias-corrected 95% band
b.lower, b.upper, b.critical
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["bands.jl"]
```
