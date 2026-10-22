"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

So far the package provides the smoothing kernels, see [`AbstractKernel`](@ref),
and samples prepared for smoothing, see [`SmoothingData`](@ref).
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
  SmoothingData, nobs, effective_nobs, nunique, window, knn_distance

include("polynomials.jl")
include("kernels.jl")
include("data.jl")

end
