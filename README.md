# KernelSmoothing.jl

[![CI](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/KernelSmoothing.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/KernelSmoothing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/KernelSmoothing.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Kernel density estimation and local polynomial regression, with data-driven
bandwidth selection and robust bias-corrected inference.

The package is under development. So far it provides the pieces the estimators
will build on:

- smoothing kernels: the uniform, triangular, Epanechnikov, biweight, triweight
  and Gaussian kernels, with their distribution functions, moments, roughness,
  derivatives, convolutions and products;
- `SmoothingData`, a sample sorted once, with its weights and ties recorded, that
  answers local questions fast: which observations lie within a bandwidth of a
  point, how many distinct values they take, what they weigh, and how far away
  the `k`-th nearest observation is;
- `kernelsums`, the sums `∑ wᵢ zᵢ K(uᵢ) uᵢʳ` that density estimates, local
  polynomial fits and cross-validation are built from, with one bandwidth or one
  per point, exact direct, windowed and fast engines and an approximate binned
  one. They run in parallel when Julia is started with several threads
  (`julia --threads=auto`).

## Installation

Requires Julia 1.10 or later. The package is not registered; install it from
GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/mlakolar/KernelSmoothing.jl")
```
