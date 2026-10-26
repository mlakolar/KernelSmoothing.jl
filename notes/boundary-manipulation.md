# Densities at a boundary and the manipulation test

In this note, we ask two questions about the local polynomial density estimator of
Cattaneo, Jansson and Ma (2020). First, do the intervals of `LocalPolynomialDensity`
cover at and near a boundary of the support, where the kernel density estimate is
biased? Second, what are the size and the power of the manipulation test of
`DensityDiscontinuity`, which is built on the same estimator?

## Design

For the intervals, we draw exponential samples and samples of Beta(1, 3), whose
densities are 1 and 3 at the boundary 0, and estimate the density at 0, 0.05, 0.2 and
0.5 with the default bandwidths, for n = 500, 2000 and 10000. We compare the
conventional and the robust bias-corrected intervals of nominal level 95%.

For the test, x is standard normal or 2B - 1 with B ~ Beta(2, 4), the running
variable of the regression discontinuity designs, and the observations below the
cutoff 0 are kept with probability 1/(1 + δ), so that the density jumps by the ratio
1 + δ at the cutoff. δ = 0 gives the size, and δ = 0.1, 0.25 and 0.5 the power, for
n = 1000 and 5000. There are 2000 replications of every design
(`simulations/lpdboundary.jl` and `simulations/rddensity.jl`, run
`20261026T0215Z-ks-sim59c`).

## Results

Coverage of the 95% intervals of `LocalPolynomialDensity` (Monte Carlo standard
error), and the mean length of the robust interval:

| density | n | point | conventional | robust | robust length |
|---|---|---|---|---|---|
| exponential | 500 | 0 | 95.4 (0.5) | 96.4 (0.4) | 1.049 |
| exponential | 500 | 0.05 | 94.4 (0.5) | 95.8 (0.4) | 0.537 |
| exponential | 500 | 0.2 | 94.4 (0.5) | 94.3 (0.5) | 0.217 |
| exponential | 500 | 0.5 | 89.0 (0.7) | 95.0 (0.5) | 0.196 |
| exponential | 2000 | 0 | 95.6 (0.5) | 96.9 (0.4) | 0.551 |
| exponential | 2000 | 0.05 | 94.2 (0.5) | 95.7 (0.5) | 0.268 |
| exponential | 2000 | 0.2 | 92.9 (0.6) | 94.3 (0.5) | 0.113 |
| exponential | 2000 | 0.5 | 84.6 (0.8) | 94.2 (0.5) | 0.107 |
| exponential | 10000 | 0 | 94.6 (0.5) | 96.6 (0.4) | 0.264 |
| exponential | 10000 | 0.05 | 92.8 (0.6) | 95.0 (0.5) | 0.12 |
| exponential | 10000 | 0.2 | 93.7 (0.5) | 95.4 (0.5) | 0.063 |
| exponential | 10000 | 0.5 | 85.7 (0.8) | 95.4 (0.5) | 0.057 |
| Beta(1, 3) | 500 | 0 | 95.4 (0.5) | 96.4 (0.4) | 3.821 |
| Beta(1, 3) | 500 | 0.05 | 95.2 (0.5) | 94.6 (0.5) | 0.828 |
| Beta(1, 3) | 500 | 0.2 | 91.9 (0.6) | 96.0 (0.4) | 0.542 |
| Beta(1, 3) | 500 | 0.5 | 89.2 (0.7) | 93.8 (0.5) | 0.353 |
| Beta(1, 3) | 2000 | 0 | 94.6 (0.5) | 95.3 (0.5) | 2.014 |
| Beta(1, 3) | 2000 | 0.05 | 95.0 (0.5) | 94.9 (0.5) | 0.426 |
| Beta(1, 3) | 2000 | 0.2 | 88.8 (0.7) | 96.3 (0.4) | 0.275 |
| Beta(1, 3) | 2000 | 0.5 | 84.1 (0.8) | 94.7 (0.5) | 0.193 |
| Beta(1, 3) | 10000 | 0 | 95.4 (0.5) | 96.0 (0.4) | 0.969 |
| Beta(1, 3) | 10000 | 0.05 | 95.2 (0.5) | 94.8 (0.5) | 0.232 |
| Beta(1, 3) | 10000 | 0.2 | 77.3 (0.9) | 95.2 (0.5) | 0.134 |
| Beta(1, 3) | 10000 | 0.5 | 83.1 (0.8) | 95.4 (0.5) | 0.1 |

Rejection rates of the manipulation test at the 5% level:

| density | n | δ = 0 (size) | δ = 0.1 | δ = 0.25 | δ = 0.5 |
|---|---|---|---|---|---|
| normal | 1000 | 3.9 (0.4) | 5.1 (0.5) | 11.4 (0.7) | 29.1 (1.0) |
| normal | 5000 | 4.8 (0.5) | 8.6 (0.6) | 32.8 (1.0) | 77.0 (0.9) |
| 2 Beta(2, 4) - 1 | 1000 | 3.9 (0.4) | 5.1 (0.5) | 9.8 (0.7) | 20.8 (0.9) |
| 2 Beta(2, 4) - 1 | 5000 | 4.4 (0.5) | 9.4 (0.7) | 31.2 (1.0) | 78.6 (0.9) |

The robust intervals cover at 93.8% to 96.9% at every point, at the boundary as in
the interior, and the boundary costs only length: there, the robust interval is about
5 to 11 times as long as at the interior point 0.5. The conventional intervals cover at the
boundary and at 0.05, but undercover at 0.2 and 0.5, by up to 18 points at
n = 10000, presumably because the bandwidths there are large enough for the bias to
matter.

The manipulation test is slightly conservative, with a size of 3.9% to 4.8%. Its power
against a jump of 10% in the density is close to the size at n = 1000 and below 10% at
n = 5000; against a jump of 50% it rejects 21% to 29% of the time at n = 1000 and 77%
to 79% at n = 5000. Small manipulations are thus hard to detect at the sample sizes of
many applications.

## Recommendations

- Use the robust bias-corrected intervals of `LocalPolynomialDensity`, the default of
  `confint`, at boundary and interior points alike.
- Read a manipulation test that does not reject with its power in mind: with a few
  thousand observations, only large jumps in the density are detected reliably.
