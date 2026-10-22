"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

So far the package provides the smoothing kernels, see [`AbstractKernel`](@ref),
samples prepared for smoothing, see [`SmoothingData`](@ref), and the kernel sums
that estimators are built from, see [`kernelsums`](@ref).
"""
module KernelSmoothing

import Distributions
import Distributions: cdf, support, convolve, moment
import StatsAPI: nobs

export
  AbstractKernel,
  UniformKernel, TriangularKernel, EpanechnikovKernel, BiweightKernel,
  TriweightKernel, GaussianKernel,
  derivative, roughness, smoothness,
  cdf, support, convolve, moment,
  SmoothingData, nobs, effective_nobs, nunique, window, knn_distance,
  kernelsums, kernelsums!, SumEngine, DirectSums, WindowSums, FastSums, BinnedSums, isexact

include("polynomials.jl")
include("kernels.jl")
include("data.jl")
include("engines.jl")

end
