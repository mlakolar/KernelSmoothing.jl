# [Bandwidth selection](@id bandwidths-page)

Estimators accept a number or a [`BandwidthSelector`](@ref) as bandwidth.
[`selectbandwidth`](@ref) applies a rule and returns a [`BandwidthSelection`](@ref)
that records what the rule computed, such as pilot bandwidths or a
cross-validation curve.

For densities, [`SheatherJones`](@ref) is the default; [`LSCV`](@ref),
[`NormalReference`](@ref) and [`SilvermanRule`](@ref) are alternatives. For
regression, [`RuleOfThumb`](@ref) is the default and [`CrossValidation`](@ref) an
alternative.

```julia
d = SmoothingData(randn(1000))
s = selectbandwidth(LSCV(), d, EpanechnikovKernel())
s.h, s.details.minimizers        # the bandwidth, and all local minimizers of the criterion
```

```@autodocs
Modules = [KernelSmoothing]
Pages = ["bandwidths.jl", "selectors.jl"]
```
