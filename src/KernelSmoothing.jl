"""
    KernelSmoothing

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

The estimators are [`KDE`](@ref), a kernel density estimate that is a distribution
of Distributions.jl; [`PointwiseDensity`](@ref), density estimates at a set of
points with robust bias-corrected intervals; [`LocalPolynomialDensity`](@ref), the
boundary-adaptive density estimator of Cattaneo, Jansson and Ma; and
[`LocalPolynomialRegression`](@ref), local polynomial estimates of a regression
function or its derivatives with conventional, robust bias-corrected and
[`Honest`](@ref) intervals and uniform bands, see [`conf_band`](@ref); and
[`RegressionDiscontinuity`](@ref), estimates for regression discontinuity designs.
They are fitted
with `fit` and choose their bandwidths with a [`BandwidthSelector`](@ref).
Underneath are the smoothing kernels, see [`AbstractKernel`](@ref), samples
prepared for smoothing, see [`SmoothingData`](@ref), and the kernel sums that many
of the estimators and bandwidth selectors are built from, see [`kernel_sums`](@ref).
"""
module KernelSmoothing

import Distributions
import Distributions: cdf, ccdf, cquantile, support, convolve, moment, pdf, logpdf, quantile, insupport, mean, var,
  std, median
import LinearAlgebra
import StatsAPI
import StatsAPI: fit, nobs, stderror, coef, confint, coeftable, predict, fitted, residuals, vcov
import Random
using Random: AbstractRNG
using PrecompileTools: @setup_workload, @compile_workload

export
  AbstractKernel,
  UniformKernel, TriangularKernel, EpanechnikovKernel, BiweightKernel,
  TriweightKernel, GaussianKernel, PiecewisePolynomialKernel,
  derivative, roughness, smoothness,
  cdf, ccdf, support, insupport, convolve, moment,
  SmoothingData, nobs, effective_nobs, nunique, weight_sum, window, knn_distance,
  kernel_sums, kernel_sums!, SumEngine, DirectSums, WindowSums, FastSums, BinnedSums, isexact,
  SumsWorkspace,
  BandwidthSelector, BandwidthSelection, select_bandwidth, bandwidth,
  NormalReference, SilvermanRule, SheatherJones, LSCV, CrossValidation,
  KDE, fit, pdf, logpdf, quantile, cquantile, mean, var, std, median, stderror,
  LocalPolynomialRegression, PointwiseDensity, LocalPolynomialDensity, RuleOfThumb, MSEOptimal, IMSEOptimal, CoverageOptimal,
  VarianceEstimator, NNVariance, HCVariance, ClusterVariance,
  InferenceMethod, Conventional, RobustBiasCorrected, Honest, FLCIOptimal, worst_case_bias,
  RegressionDiscontinuity, RDOptimal, RDHonestOptimal, DensityDiscontinuity, RDDensityOptimal,
  coef, confint, coeftable, predict, fitted, residuals, vcov, conf_band,
  local_polynomial, smoother_matrix, PartiallyLinear, LocalLikelihood,
  kernel_weights, regression_points, bandwidth_grid, grid_minimize, sandwich, supt_critical

include("polynomials.jl")
include("kernels.jl")
include("data.jl")
include("engines.jl")
include("bandwidths.jl")
include("selectors.jl")
include("kde.jl")
include("locpoly.jl")
include("clusters.jl")
include("surface.jl")
include("plugin.jl")
include("kdeinference.jl")
include("bands.jl")
include("honest.jl")
include("lpdensity.jl")
include("rdbandwidth.jl")
include("rd.jl")
include("rdhonest.jl")
include("semiparametric.jl")
include("locallik.jl")
include("rddensity.jl")
include("building.jl")
include("precompile.jl")

end
