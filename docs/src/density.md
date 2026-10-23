# [Density estimation](@id density-page)

`fit(KDE, x)` returns a kernel density estimate that is a continuous univariate
distribution of Distributions.jl. Its density and distribution function are
computed exactly from the observations, without a grid; broadcasting over many
points uses the engines of [`kernelsums`](@ref). Bandwidths are numbers or
[`BandwidthSelector`](@ref)s, by default [`SheatherJones`](@ref); see
[Bandwidth selection](@ref bandwidths-page).

```@autodocs
Modules = [KernelSmoothing]
Pages = ["kde.jl"]
```

## Inference

[`PointwiseDensity`](@ref) estimates the density at a set of points with
conventional and robust bias-corrected standard errors and intervals (Calonico,
Cattaneo and Farrell 2018), as nprobust's `kdrobust` does, but with the bias
constant of the paper; `compat = true` reproduces nprobust. A fitted `KDE` gives
intervals at its own bandwidth with `confint(f, points)`.

```julia
x = randn(1000)
m = fit(PointwiseDensity, x; points = [-1, 0, 1])   # plug-in bandwidths at each point
confint(m)                                          # robust bias-corrected 95% intervals
confint(fit(KDE, x), [-1, 0, 1])                    # at the Sheather-Jones bandwidth
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["kdeinference.jl"]
```

## At boundaries

[`LocalPolynomialDensity`](@ref) is the estimator of Cattaneo, Jansson and Ma
(2020), as R's lpdensity computes it: a local polynomial fit to the empirical
distribution function, whose slope estimates the density. It needs no knowledge of
where a boundary lies and keeps its rate there. By default, the bandwidth at each
point is lpdensity's plug-in estimate, [`MSEOptimal`](@ref).

```julia
x = randexp(5000)
m = fit(LocalPolynomialDensity, x; points = [0.0, 0.5, 1.0])   # consistent at the boundary 0
confint(m)
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["lpdensity.jl"]
```
