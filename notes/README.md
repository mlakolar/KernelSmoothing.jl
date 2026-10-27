# Performance and coverage notes

These notes document how fast the procedures of the package are, how well their
intervals cover, and which settings we recommend (#59). Each note reads like the
simulation section of a paper. It states a question, describes the design, reports
the results, and ends with recommendations. The scripts that produce every table are
in [`../simulations`](../simulations) and, for the timings, in
[`../benchmark`](../benchmark); each simulation can be reproduced from its seeds.

In most tables, the parentheses give the Monte Carlo standard error of a coverage
rate; where a table averages over points, they give the smallest coverage, as its
caption says. With 2000 replications, the standard error is about half a percentage
point for a coverage near 95%, and up to one point for a coverage near 75%.

The [recommendations](recommendations.md) collect the advice of all notes on one page.

| note | estimators |
|---|---|
| [Estimators added in #60–#65](new-estimators.md) | `LocalLikelihood` (new families and intervals), `LocalQuantileRegression` (conventional and RBC intervals, #94), `rearrange` and `constrain`, `BoundedKDE`, `LocalRandomization` |
| [Local polynomial regression and density estimation](regression-density.md) | `LocalPolynomialRegression`, `PointwiseDensity`, `conf_band` |
| [Cluster-robust variances](clusters.md) | `ClusterVariance` |
| [Regression discontinuity and honest intervals](rd-honest.md) | `RegressionDiscontinuity`, `Honest`, `FLCIOptimal` |
| [Densities at a boundary and the manipulation test](boundary-manipulation.md) | `LocalPolynomialDensity`, `DensityDiscontinuity` |
| [Multivariate densities](multivariate.md) | `MultivariateKDE` and its bandwidth matrices |
| [The partially linear model](partially-linear.md) | `PartiallyLinear` |
| [Timings](timings.md) | the default call of every estimator, on one and eight threads, against other Julia packages and the R counterparts |
