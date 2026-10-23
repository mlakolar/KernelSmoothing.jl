"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

The package provides smoothing kernels, see [`AbstractKernel`](@ref), samples
prepared for smoothing, see [`SmoothingData`](@ref), the kernel sums that
estimators are built from, see [`kernelsums`](@ref), kernel density estimates, see
[`KDE`](@ref), pointwise density estimates with robust bias-corrected inference,
see [`PointwiseDensity`](@ref), and local polynomial regression, see
[`LocalPolynomialRegression`](@ref).
"""
module KernelSmoothing

import Distributions
import Distributions: cdf, support, convolve, moment, pdf, logpdf, quantile, insupport, mean, var
import LinearAlgebra
import StatsAPI
import StatsAPI: fit, nobs, stderror, coef, confint, coeftable, predict, fitted, residuals, vcov
import Random
using Random: AbstractRNG
using PrecompileTools: @setup_workload, @compile_workload

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
  NormalReference, SilvermanRule, SheatherJones, LSCV, CrossValidation,
  KDE, fit, pdf, logpdf, quantile, stderror,
  LocalPolynomialRegression, PointwiseDensity, LocalPolynomialDensity, RuleOfThumb, MSEOptimal, IMSEOptimal, CoverageOptimal,
  VarianceEstimator, NNVariance, HCVariance,
  InferenceMethod, Conventional, RobustBiasCorrected, Honest, FLCIOptimal, worstcasebias,
  coef, confint, coeftable, predict, fitted, residuals, vcov, confband

include("polynomials.jl")
include("kernels.jl")
include("data.jl")
include("engines.jl")
include("bandwidths.jl")
include("selectors.jl")
include("kde.jl")
include("locpoly.jl")
include("plugin.jl")
include("kdeinference.jl")
include("bands.jl")
include("honest.jl")
include("lpdensity.jl")
include("precompile.jl")

end
