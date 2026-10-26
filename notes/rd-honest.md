# Regression discontinuity and honest intervals: coverage

In this note, we ask how the intervals of `RegressionDiscontinuity` cover in the
designs that Calonico, Cattaneo and Titiunik (2014) used to make their case for robust
bias correction, and what the honest intervals of Armstrong and Kolesár (2018, 2020)
gain and cost, including when their bound on the curvature is wrong.

## Design

For regression discontinuity, we take Models 1 and 2 of Calonico, Cattaneo and
Titiunik (2014), which mimic the data of Lee (2008) and of Ludwig and Miller (2007):
x = 2 B - 1 with B ~ Beta(2, 4), fifth-degree polynomials on the two sides of the
cutoff 0, and errors N(0, 0.1295²). The jump is 0.04 in the first design and -3.45 in
the second. The sample sizes are 500 and 5000, with 2000 replications. We compare the
conventional and the robust bias-corrected intervals with rdrobust's MSE-optimal
bandwidth (`RDOptimal(:mserd)`) and its coverage-error-optimal one (`:cerrd`). The
package reproduces rdrobust to rounding, so that these coverage rates are also those
of rdrobust.

For honest intervals, x is uniform on [0, 1] and y = m(x) + σε for the sine and the
peak of the regression note, whose largest second derivatives are M = 4π² and M = 64,
for n = 500 and 2000 with 2000 replications.
At the points 0.1 to 0.9 we compare the robust bias-corrected interval at the
MSE-optimal bandwidth with the honest interval at its optimal bandwidth
(`FLCIOptimal`), with the true M, with M/2, and with the rule of thumb for M of
Armstrong and Kolesár (2020). The scripts are `simulations/rd.jl` and
`simulations/honest.jl`, and the results come from the run
`20261026T0140Z-ks-sim59b`.

## Results

Regression discontinuity, coverage of 95% intervals for the jump (Monte Carlo
standard error), mean bias of the estimate, mean length and median bandwidth:

| design | n | method | coverage | bias | length | median h |
|---|---|---|---|---|---|---|
| lee | 500 | conventional-cer | 91.3 (0.6) | 0.0146 | 0.243 | 0.14 |
| lee | 500 | conventional-mse | 89.6 (0.7) | 0.0212 | 0.208 | 0.19 |
| lee | 500 | rbc-cer | 92.2 (0.6) | 0.0111 | 0.266 | 0.14 |
| lee | 500 | rbc-mse | 91.4 (0.6) | 0.015 | 0.246 | 0.19 |
| lee | 5000 | conventional-cer | 90.8 (0.6) | 0.0095 | 0.086 | 0.108 |
| lee | 5000 | conventional-mse | 77.5 (0.9) | 0.0174 | 0.069 | 0.165 |
| lee | 5000 | rbc-cer | 92.8 (0.6) | 0.007 | 0.09 | 0.108 |
| lee | 5000 | rbc-mse | 88.4 (0.7) | 0.0115 | 0.08 | 0.165 |
| ludwig-miller | 500 | conventional-cer | 91.1 (0.6) | 0.0246 | 0.378 | 0.073 |
| ludwig-miller | 500 | conventional-mse | 86.6 (0.8) | 0.0449 | 0.314 | 0.099 |
| ludwig-miller | 500 | rbc-cer | 93.5 (0.6) | 0.003 | 0.393 | 0.073 |
| ludwig-miller | 500 | rbc-mse | 93.6 (0.5) | 0.006 | 0.343 | 0.099 |
| ludwig-miller | 5000 | conventional-cer | 92.4 (0.6) | 0.0067 | 0.144 | 0.038 |
| ludwig-miller | 5000 | conventional-mse | 87.4 (0.7) | 0.0173 | 0.116 | 0.059 |
| ludwig-miller | 5000 | rbc-cer | 93.6 (0.5) | 0.0002 | 0.147 | 0.038 |
| ludwig-miller | 5000 | rbc-mse | 93.2 (0.6) | 0.0021 | 0.123 | 0.059 |

Honest intervals: mean coverage over the five points (smallest in parentheses), in
percent, and mean length:

| design | n | robust, MSE | honest, true M | honest, M/2 | honest, rule of thumb |
|---|---|---|---|---|---|
| sine | 500 | 94.0 (93.2), 0.302 | 94.8 (93.4), 0.256 | 91.3 (86.4), 0.223 | 95.2 (94.2), 0.293 |
| sine | 2000 | 94.9 (94.6), 0.171 | 96.1 (95.2), 0.147 | 92.1 (87.7), 0.128 | 96.5 (96.0), 0.167 |
| peak | 500 | 93.7 (93.2), 0.204 | 95.0 (93.0), 0.187 | 93.6 (86.4), 0.163 | 94.6 (93.8), 0.200 |
| peak | 2000 | 94.9 (94.5), 0.115 | 96.5 (95.2), 0.108 | 94.9 (87.9), 0.094 | 96.9 (96.2), 0.116 |

In regression discontinuity, the robust bias-corrected interval at the coverage-error
optimal bandwidth covers 92% to 94%, and it is the only interval that stays close to
nominal at n = 5000 in the Lee design. At the MSE-optimal bandwidth, the robust
interval loses ground as n grows in that design (91% at n = 500, 88% at n = 5000),
because the bias of the bias-corrected estimate is still a sizable fraction of its
standard error there: from the table, about a quarter of the standard error at
n = 500 and a half at n = 5000. The conventional intervals undercover in both designs (77% to
92%).

Honest intervals with the true M cover at about the nominal level at n = 500 and
conservatively at n = 2000 (96% to 97%), which is the price of guaranteeing coverage
over the whole class of functions. Note that they are nevertheless 6% to 15% shorter
than the robust intervals, presumably because they use the bound on M instead of
estimating the bias. With M halved, they fail where the function curves most, at the
peak and at the extremes of the sine (86% to 88%), while they remain valid elsewhere.
The rule of thumb for M gives intervals that cover at 94% to 97%. The robust intervals
cover at 93% to 95% without a bound on M.

## Recommendations

- In regression discontinuity, report the robust bias-corrected interval, and prefer
  the coverage-error optimal bandwidth `RDOptimal(:cerrd)` for inference; the
  MSE-optimal bandwidth is for the point estimate.
- Use honest intervals when a credible bound on the curvature is available, and
  expect them to be conservative. A bound that is too small by half brings coverage
  down to 86% to 88% where the function curves, up to nine points below nominal.
