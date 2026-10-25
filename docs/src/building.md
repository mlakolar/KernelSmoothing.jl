# Building blocks

The estimators of this package are assembled from a few pieces that other packages
can use as well, such as the varying-coefficient models of
[CoordinateDescent.jl](https://github.com/mlakolar/CoordinateDescent.jl): the kernel
weights of the observations near a point, the default evaluation points and
bandwidth grids, the search for the minimizer of a criterion over a grid, the robust
variances of a local weighted least squares fit, and the critical value of uniform
bands. Their results agree with those inside the estimators.

```julia
d = SmoothingData(x, y)
k = EpanechnikovKernel()
grid = bandwidth_grid(d; mindistinct = 3)          # local linear fits are identified on it
h, details = grid_minimize(cvcriterion, grid)      # a criterion of your own, a function of h
r = kernel_weights(d, k, 0.5, h)
Z = [ones(length(r.window)) (d.x[r.window] .- 0.5) ./ h]
θ = (Z' * (r.weights .* Z)) \ (Z' * (r.weights .* d.y[r.window]))
V = sandwich(Z, r.weights, d.y[r.window] .- Z * θ; vce = HCVariance(3))
```

```@docs
kernel_weights
regression_points
bandwidth_grid
grid_minimize
sandwich
supt_critical
```
