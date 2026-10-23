"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

The package provides smoothing kernels, see [`AbstractKernel`](@ref), samples
prepared for smoothing, see [`SmoothingData`](@ref), the kernel sums that
estimators are built from, see [`kernelsums`](@ref), and kernel density estimates,
see [`KDE`](@ref).
"""
module KernelSmoothing

import Distributions
import Distributions: cdf, support, convolve, moment, pdf, logpdf, quantile, insupport, mean, var
import LinearAlgebra
import StatsAPI: fit, nobs, stderror
using Random: AbstractRNG

export
  AbstractKernel,
  UniformKernel, TriangularKernel, EpanechnikovKernel, BiweightKernel,
  TriweightKernel, GaussianKernel,
  derivative, roughness, smoothness,
  cdf, support, convolve, moment,
  SmoothingData, nobs, effective_nobs, nunique, weightsum, window, knn_distance,
  kernelsums, kernelsums!, SumEngine, DirectSums, WindowSums, FastSums, BinnedSums, isexact,
  SumsWorkspace,
  BandwidthSelector, BandwidthSelection, selectbandwidth, bandwidth,
  NormalReference, SilvermanRule,
  KDE, fit, pdf, logpdf, quantile, stderror

include("polynomials.jl")
include("kernels.jl")
include("data.jl")
include("engines.jl")
include("bandwidths.jl")
include("kde.jl")

end
