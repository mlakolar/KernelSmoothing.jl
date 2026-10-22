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
