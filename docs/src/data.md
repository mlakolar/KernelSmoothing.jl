# [Prepared data](@id data-page)

[`SmoothingData`](@ref) sorts a sample once and keeps, in sorted order, what the
estimators ask for repeatedly: the observations, the responses if any, the
weights, the distinct values and where each begins, and the cumulative weights.
Every estimator accepts a `SmoothingData` in place of the vectors, so that a
sample prepared once can be smoothed many times.

Weights are relative, so that only their ratios matter, and observations with zero
weight are dropped. The effective sample size of a weighted sample is Kish's,

```math
n_{\mathrm{eff}} = \frac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2},
```

which suits sampling weights and equals ``n`` for equal weights. Missing values,
non-finite values and vectors of different lengths raise errors, so that no
observation is dropped silently. Tied observations keep their input order.

Since the data are sorted, the questions that the estimators ask at every point
are cheap. [`window`](@ref) gives the observations within a bandwidth of a point,
by binary search, with the same floating-point test that a compactly supported
kernel applies, so that the window holds exactly the observations that the kernel
reaches. [`nunique`](@ref) and [`weightsum`](@ref) count the distinct values and the
weight of a range of observations in constant time, and [`knn_distance`](@ref) gives
the distance to the ``k``-th nearest observation or distinct value, the smallest
bandwidth whose window holds ``k`` of them. The floors that the estimators put on
their bandwidths, such as nprobust's 21 observations, are such distances.

```@autodocs
Modules = [KernelSmoothing]
Pages = ["data.jl"]
```
