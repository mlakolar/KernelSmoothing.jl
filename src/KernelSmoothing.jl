"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

So far the package provides the smoothing kernels; see [`AbstractKernel`](@ref).
"""
module KernelSmoothing

import Distributions
import Distributions: cdf, support, convolve, moment

export
  AbstractKernel,
  UniformKernel, TriangularKernel, EpanechnikovKernel, BiweightKernel,
  TriweightKernel, GaussianKernel,
  derivative, roughness, smoothness,
  cdf, support, convolve, moment

include("polynomials.jl")
include("kernels.jl")

end
