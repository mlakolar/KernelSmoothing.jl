# Recommendations

This page collects the recommendations of the notes, with the evidence for each in the
note it cites. They describe what worked in our simulations, at the sample sizes and
designs of the notes; they are not guarantees.

## Inference

- **Report robust bias-corrected intervals**, the default of `confint` for local
  polynomial regression, densities, local polynomial densities, regression
  discontinuity and local likelihood. In regression and density estimation they
  cover close to the nominal level from n = 2000 on, where the conventional intervals
  undercover by up to 22 points (73% in the peak design with one bandwidth for all
  points); for regression discontinuity see below. They are 1.4 to 1.6 times as long
  in regression and density estimation, and 1.0 to 1.2 times in regression
  discontinuity.
  ([regression and densities](regression-density.md),
  [boundaries](boundary-manipulation.md), [estimators of #60–#65](new-estimators.md))
- **Use the robust bands of `conf_band`** for statements about the whole function.
  Expect 84% to 91% coverage at n = 500 and 94% to 95% by n = 10000.
  ([regression and densities](regression-density.md))
- **With clustered data, pass `cluster`** and use a cluster-robust variance, CR3 with
  fewer than about 50 clusters. Ignoring the clusters gives intervals that cover 65%
  to 78% of the time. ([clusters](clusters.md))
- **In regression discontinuity, use `RDOptimal(:cerrd)` for inference**. The
  MSE-optimal bandwidth is for the point estimate, and its robust interval can fall
  to 88% in a curved design. ([RD](rd-honest.md))
- **Use honest intervals with a credible bound** on the curvature. They cover at
  about the nominal level at n = 500 and conservatively at n = 2000, and they are 6%
  to 15% shorter than the robust intervals; a bound that is too small by half brings
  coverage down to 86% to 88%. ([RD](rd-honest.md))
- **The intervals of `LocalQuantileRegression` ignore the bias**. Where the quantile
  function curves, read them as intervals for the smoothed quantile, or use a smaller
  bandwidth; a bias correction is proposed in #94. ([estimators of #60–#65](new-estimators.md))
- **`LocalRandomization` needs a window in which the outcome is flat**. A slope in the
  window makes the tests reject far too often (#95).
  ([estimators of #60–#65](new-estimators.md))
- **In `PartiallyLinear`, pass a small bandwidth** when a covariate depends on u in a
  nonlinear way (#96). ([partially linear model](partially-linear.md))
- **A manipulation test that does not reject proves little** with a few thousand
  observations, since only large jumps in the density are detected reliably.
  ([manipulation test](boundary-manipulation.md))

## Estimation

- **Rearrange monotone fits.** `rearrange` reduces the error and can only improve the
  band. Impose a shape only when it is known.
  ([estimators of #60–#65](new-estimators.md))
- **On a bounded support**, use `BoundedKDE` with reflection when the density is
  positive at the boundary, and the plain `KDE` when it vanishes there.
  ([estimators of #60–#65](new-estimators.md))
- **In two dimensions, keep the default plug-in bandwidth matrix.** Use a full matrix
  for correlated coordinates. ([multivariate densities](multivariate.md))

## Speed

- **Bandwidth selection is most of the cost.** When fitting many curves with one
  bandwidth, choose it once and pass it. ([timings](timings.md))
- **Start Julia with threads** for regression, local likelihood and multivariate
  densities. ([timings](timings.md))
- **Prefer `RuleOfThumb()` for `LocalLikelihood` beyond a few thousand
  observations.** Cross-validation grows nearly quadratically (#93), and the rule of thumb
  covered as well in our studies. ([timings](timings.md),
  [estimators of #60–#65](new-estimators.md))
