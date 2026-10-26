# Multivariate densities: bandwidth matrices

In this note, we ask which bandwidth matrix of `MultivariateKDE` (#60) to use in two
dimensions: the normal scale matrix, the plug-in matrix of Wand and Jones (1994) and
Duong and Hazelton (2003), smoothed cross-validation (Duong and Hazelton 2005) or
least-squares cross-validation, and whether a full matrix is worth its extra
parameter.

## Design

We draw samples of n = 200 and 1000 from three bivariate targets, in the spirit of
those of Wand and Jones (1993): a normal with correlation 0.7; a bimodal mixture of
two normals with standard deviation 0.5 centred at (±1, 0); and a skewed mixture with
weights 0.2, 0.2 and 0.6 of normals centred at (0, 0), (0.5, 0.5) and (1, 1) with
standard deviations 1, 0.4 and 0.2. For each of the eight rules (four rules, full and diagonal), we
compute the integrated squared error on a 101 × 101 grid over [-3.5, 3.5]², with 500
replications at n = 200 and 200 at n = 1000 (`simulations/mvkde.jl`, run
`20261026T0215Z-ks-sim59c`).

## Results

Mean integrated squared error × 1000:

| target | n | full normal scale | full plug-in | full SCV | full LSCV | diagonal normal scale | diagonal plug-in | diagonal SCV | diagonal LSCV |
|---|---|---|---|---|---|---|---|---|---|
| correlated | 200 | 4.01 | 4.358 | 4.104 | 6.814 | 5.434 | 5.829 | 5.375 | 6.492 |
| correlated | 1000 | 1.487 | 1.534 | 1.519 | 1.944 | 1.99 | 2.033 | 2.007 | 2.244 |
| bimodal | 200 | 14.772 | 10.495 | 11.578 | 13.048 | 14.686 | 10.408 | 11.471 | 11.054 |
| bimodal | 1000 | 6.649 | 3.741 | 3.744 | 4.077 | 6.638 | 3.723 | 3.727 | 3.746 |
| skewed | 200 | 208.763 | 64.22 | 76.175 | 72.898 | 187.75 | 59.663 | 69.425 | 62.062 |
| skewed | 1000 | 121.573 | 21.496 | 23.32 | 23.281 | 103.524 | 20.581 | 21.988 | 21.376 |

Mean time of the selection and the fit, in seconds, averaged over the targets:

| n | normal scale | plug-in | SCV | LSCV |
|---|---|---|---|---|
| 200 | 0.0259 | 0.0031 | 0.174 | 0.065 |
| 1000 | 0.0656 | 0.059 | 1.02 | 0.656 |

The plug-in matrix is within 10% of the best rule for every target, and it is
fast. The normal scale matrix is the best for the normal target, as it should be, but
its error for the mixtures is 1.4 to 1.8 times that of the plug-in matrix for the
bimodal target and 3 to 6 times for the skewed one. The two cross-validation rules
come close to the plug-in matrix at n = 1000, except full least-squares
cross-validation on the correlated target, whose error is 27% larger; at n = 200,
least-squares cross-validation is the least accurate for the correlated target. The
full matrices pay off for the correlated target, by about a quarter of the error with
the plug-in and SCV rules. For the two mixtures, whose components are spherical, the
diagonal matrices are as good or better, by up to 15%. Note that the time of the
normal scale matrix at n = 200 includes compilation, since it is the first rule each
worker runs.

## Recommendations

- Use the default, the plug-in matrix. Use the normal scale matrix only for data that
  are close to normal.
- Use a full matrix when the coordinates are correlated, and consider a diagonal one
  for a mixture of spherical components.
