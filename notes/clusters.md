# Cluster-robust variances: coverage

In this note, we ask how the robust bias-corrected intervals of
`LocalPolynomialRegression` cover when the observations are clustered, with the
cluster-robust variances `ClusterVariance(1)`, `(2)` and `(3)` (CR1, CR2, CR3) of #46
and with a heteroskedasticity-consistent variance that ignores the clusters, and how
their coverage falls with the number of clusters.

## Design

There are G = 20, 50 or 200 clusters, of 25 observations each (balanced) or of sizes
drawn uniformly from 5 to 45 (unbalanced). Both the covariate and the errors are
correlated within clusters: x = 0.5 u_g + 0.5 v with u_g and v uniform on [0, 1], and
y = sin(2πx) + 0.5 (√0.5 a_g + √0.5 e) with a_g and e standard normal, so that half of
the error variance is common to the cluster. The bandwidth is the MSE-optimal one of
each point with CR1, and all four variances use the same bandwidth, so that only the
standard errors differ. The points are 0.3, 0.5 and 0.7, and there are 2000
replications of each design. The script is `simulations/cluster.jl`, and the results
come from the run `20261026T0140Z-ks-sim59b`.

## Results

Mean coverage over the three points, with the smallest in parentheses, in percent,
of the robust bias-corrected intervals of nominal level 95%:

| design | CR1 | CR2 | CR3 | HC1, ignoring clusters |
|---|---|---|---|---|
| balanced-20 | 91.8 (90.8) | 91.8 (90.9) | 92.2 (91.4) | 69.5 (68.3) |
| balanced-50 | 94.0 (92.6) | 94.0 (92.7) | 94.1 (92.8) | 73.7 (72.4) |
| balanced-200 | 94.4 (93.8) | 94.4 (93.8) | 94.4 (93.8) | 78.3 (77.6) |
| unbalanced-20 | 90.6 (90.2) | 90.9 (90.5) | 91.7 (91.5) | 66.1 (65.2) |
| unbalanced-50 | 93.2 (92.1) | 93.4 (92.6) | 93.6 (93.0) | 69.4 (68.9) |
| unbalanced-200 | 94.4 (94.2) | 94.4 (94.2) | 94.5 (94.3) | 74.5 (74.2) |

With 200 clusters, the cluster-robust intervals cover at the nominal level. With 50
clusters they lose up to three points at the outer points, and with 20 clusters three
to five points, more in the unbalanced designs. The three variances differ little:
CR3, which inflates the residuals of influential clusters most, is the closest to
nominal with few clusters, by about one point. An interval that ignores the clusters
covers only 65% to 78% of the time on average, and it improves only slowly with the
number of clusters (from 70% to 78% in the balanced designs), presumably because the bandwidth shrinks and each
window holds fewer observations of a cluster.

## Recommendations

- With clustered data, pass `cluster` and use a cluster-robust variance; ignoring the
  clusters understates the standard errors badly even with 200 clusters.
- Prefer CR3 when there are fewer than about 50 clusters, and expect coverage a few
  points below nominal with 20 clusters. The choice matters little with many
  clusters.
