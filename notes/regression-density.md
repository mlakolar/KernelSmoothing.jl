# Local polynomial regression and density estimation: coverage

In this note, we ask how well the pointwise intervals and the uniform bands of
`LocalPolynomialRegression` and `PointwiseDensity` cover, with the bandwidths that the
package offers. Calonico, Cattaneo and Farrell (2018) show that robust bias
correction, which subtracts an estimate of the bias and accounts for its variability
in the standard error, gives intervals whose coverage error vanishes faster than that
of the conventional intervals at the MSE-optimal bandwidth. The question is whether
this shows at the sample sizes of applications, and what the choice of the bandwidth
costs.

## Design

For regression, x is uniform on [0, 1] and y = m(x) + σ(x) ε with ε ~ N(0, 1), for
three functions: a sine, m(x) = sin(2πx) with σ = 0.5; the same with heteroskedastic
errors, σ(x) = 0.2 + 0.6x; and a peak, m(x) = exp(-32 (x - 0.5)²) + 0.5x with σ = 0.3.
To see the effect of a design density that thins out at the ends, two more designs
take x normal or t₃,
with m(x) = sin(2x) and σ = 0.5. The points are 0.1, 0.25, 0.5, 0.75 and 0.9, or
-1.5, -0.75, 0, 0.75 and 1.5 for the last two designs. For densities, the samples
are normal, a bimodal mixture of N(-1, 0.25) and N(1, 0.25), and t₅, at the points
-1.5, -0.5, 0, 0.5 and 1.5. The sample sizes are 500, 2000 and 10000, with 2000
replications each, so that the Monte Carlo standard error of a coverage rate near
95% is 0.5 percentage points.

We compare the conventional and the robust bias-corrected intervals of nominal level
95% at the MSE-optimal bandwidth of each point, at the IMSE-optimal bandwidth, at the
coverage-optimal bandwidth in regression (`CoverageOptimal()`) and, for densities,
at the bandwidth of Sheather and Jones. Note that in its default convention
`CoverageOptimal()` leaves the bandwidth of local linear regression unchanged and
rescales only the pilot bandwidth, which `fit` does not use unless `pilot = :selected`,
so that its intervals coincide with those at the MSE-optimal bandwidth here. The bands are the sup-t bands of `conf_band` on
a grid of 21 points, between the outer points for regression and on [-2, 2] for
densities, with the conventional or the robust standard errors, at the IMSE-optimal
bandwidth for regression and at Sheather and Jones's for densities, with 2000 draws for the
critical value. The scripts are `simulations/regression.jl` and
`simulations/density.jl`, and the results come from the run
`20261026T0140Z-ks-sim59b`.

## Results

The tables give the mean coverage over the five points, with the smallest coverage
in parentheses, in percent, and the coverage of the bands.

Regression:

| design | n | conventional, MSE | conventional, IMSE | robust, MSE | robust, IMSE | robust, CE | sup-t band, conventional | sup-t band, robust |
|---|---|---|---|---|---|---|---|---|
| sine | 500 | 90.5 (88.4) | 90.3 (85.8) | 93.6 (92.0) | 93.4 (91.9) | 93.6 (92.0) | 83.2 | 88.8 |
| sine | 2000 | 91.2 (89.6) | 91.4 (88.1) | 95.1 (94.5) | 94.9 (94.2) | 95.1 (94.5) | 85.6 | 94.6 |
| sine | 10000 | 91.7 (89.4) | 91.4 (87.4) | 95.0 (94.4) | 94.9 (94.2) | 95.0 (94.4) | 85.6 | 93.8 |
| sine-heteroskedastic | 500 | 90.2 (87.4) | 87.9 (77.0) | 93.9 (92.4) | 93.9 (93.2) | 93.9 (92.4) | 77.6 | 89.3 |
| sine-heteroskedastic | 2000 | 91.4 (90.1) | 89.2 (80.1) | 94.5 (94.2) | 94.4 (94.0) | 94.5 (94.2) | 80.6 | 92.2 |
| sine-heteroskedastic | 10000 | 91.8 (90.6) | 89.1 (80.8) | 94.9 (94.5) | 94.7 (94.0) | 94.9 (94.5) | 81.6 | 94.2 |
| peak | 500 | 90.1 (86.8) | 88.1 (73.2) | 93.8 (93.6) | 93.6 (93.2) | 93.8 (93.6) | 78.0 | 87.6 |
| peak | 2000 | 90.8 (88.4) | 89.0 (73.9) | 94.5 (94.1) | 94.6 (94.2) | 94.5 (94.1) | 82.0 | 93.0 |
| peak | 10000 | 91.7 (90.2) | 89.7 (75.7) | 95.2 (94.6) | 94.8 (94.4) | 95.2 (94.6) | 84.6 | 94.7 |
| normal-sine | 500 | 91.6 (88.6) | 91.4 (87.7) | 93.7 (93.0) | 93.5 (92.4) | 93.7 (93.0) | 84.2 | 86.9 |
| normal-sine | 2000 | 91.8 (90.1) | 92.0 (89.2) | 94.4 (94.0) | 94.3 (93.6) | 94.4 (94.0) | 87.6 | 91.9 |
| normal-sine | 10000 | 93.0 (90.5) | 92.3 (88.1) | 95.2 (94.8) | 95.0 (94.2) | 95.2 (94.8) | 86.6 | 94.2 |
| t3-sine | 500 | 91.6 (89.4) | 91.8 (90.1) | 93.0 (91.1) | 92.9 (90.9) | 93.0 (91.1) | 85.0 | 83.7 |
| t3-sine | 2000 | 92.8 (91.1) | 93.1 (91.5) | 94.8 (94.6) | 94.3 (94.0) | 94.8 (94.6) | 89.5 | 91.7 |
| t3-sine | 10000 | 93.1 (90.4) | 93.3 (90.8) | 95.0 (94.6) | 94.8 (94.4) | 95.0 (94.6) | 90.1 | 94.2 |

Densities:

| design | n | conventional, MSE | conventional, SJ | robust, MSE | robust, IMSE | robust, SJ | sup-t band, conventional | sup-t band, robust |
|---|---|---|---|---|---|---|---|---|
| normal | 500 | 85.0 (82.4) | 88.8 (82.4) | 94.5 (94.0) | 94.6 (94.2) | 94.5 (94.0) | 86.6 | 90.8 |
| normal | 2000 | 86.3 (84.2) | 90.2 (85.6) | 94.6 (93.7) | 94.8 (94.0) | 94.7 (94.2) | 88.5 | 93.6 |
| normal | 10000 | 88.3 (87.6) | 91.0 (88.1) | 94.9 (94.3) | 95.0 (94.4) | 95.0 (94.2) | 89.1 | 94.8 |
| bimodal | 500 | 92.8 (87.7) | 91.6 (80.0) | 95.1 (94.3) | 94.6 (94.2) | 94.6 (94.0) | 83.0 | 88.1 |
| bimodal | 2000 | 93.5 (89.0) | 92.9 (82.2) | 95.5 (94.7) | 95.1 (95.0) | 95.0 (94.8) | 87.2 | 93.2 |
| bimodal | 10000 | 93.1 (90.4) | 92.4 (82.8) | 95.1 (94.6) | 94.8 (94.6) | 94.7 (94.4) | 86.2 | 94.6 |
| t5 | 500 | 84.4 (79.5) | 87.0 (74.6) | 94.7 (94.0) | 95.0 (94.3) | 95.0 (94.4) | 84.3 | 91.0 |
| t5 | 2000 | 86.6 (85.4) | 89.5 (81.4) | 94.7 (93.9) | 94.9 (94.6) | 95.1 (94.8) | 87.4 | 95.0 |
| t5 | 10000 | 89.2 (87.8) | 91.0 (84.1) | 94.9 (94.2) | 94.9 (94.6) | 94.9 (94.6) | 88.7 | 93.9 |

The robust bias-corrected intervals cover close to the nominal level at every point
and with every bandwidth from n = 2000 on: the smallest coverage at any point is
93.6%. At n = 500 they lose one to two points on average in regression (up to four at
single points of the t₃ design), and at most one for densities. The conventional
intervals undercover throughout, and they improve only slowly with n, by one to five
points from n = 500 to 10000, since at the MSE-optimal bandwidth the bias is of the
same order as the standard error. Note that the loss is largest with one bandwidth
for all points: in the peak design, the conventional interval at the IMSE-optimal
bandwidth covers only 73% to 76% of the time at its worst point, presumably where the
common bandwidth is too large for the curvature. For densities, the smallest
conventional coverage over the points is 80% to 90% for the normal and the bimodal
densities, and 75% to 88% for t₅.

The robust bands are below the nominal level at n = 500, with coverage from 84% to
91%, and reach 92% to 95% at n = 2000 and 94% to 95% at n = 10000. The conventional
bands undercover by 5 to 17 points at every sample size. Furthermore, the t₃ design is the
hardest case for the robust band at n = 500 (84%). This agrees with the validation run `20261024T1047Z-ks-b7b17`,
which found the robust band 2 to 3 points below nominal for a normal covariate at
n = 2000; we see 92% here, and the gap closes at n = 10000.

## Recommendations

- Report the robust bias-corrected intervals, the default of `confint`. At the
  MSE-optimal bandwidth they are 1.4 times as long as the conventional ones in
  regression and 1.5 to 1.6 times for densities (the lengths are in the full tables
  that `summarize.jl` prints), and they cover where these do not.
- The choice between the MSE-, IMSE- and coverage-optimal bandwidths matters little
  for the robust intervals. For the conventional ones, a common bandwidth (IMSE)
  undercovers badly where the function curves; if conventional intervals are
  wanted, use the bandwidth of each point.
- Use the robust bands for uniform statements, and treat them as somewhat optimistic
  below n = 2000.
