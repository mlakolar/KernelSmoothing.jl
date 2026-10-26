# The partially linear model: coverage

In this note, we ask whether the sandwich intervals for the coefficients of
`PartiallyLinear` (Robinson 1988) cover, with the default cross-validated bandwidth,
when the covariates depend on the variable of the smooth part.

## Design

y = x₁ - 0.5 x₂ + g(u) + σ ε with u uniform on [0, 1] and standard normal ε, where
x₁ = u + v₁ and x₂ = sin(2πu) + v₂ with standard normal v₁ and v₂, so that the
conditional mean of x₁ given u is linear and that of x₂ is a sine. The smooth part is
g(u) = sin(2πu) with σ = 0.5 or with heteroskedastic errors σ = 0.2 + 0.6u, or a
steep logistic g(u) = 2/(1 + exp(-(u - 0.5)/0.05)) with σ = 0.5. The intervals use the
HC0 and HC1 sandwich variances at the bandwidth of `CrossValidation()`, for n = 200
and 1000, with 2000 replications (`simulations/partiallylinear.jl`; run in the
development allocation of `20261025T2000Z-ks-overnight`).

## Results

Coverage of 95% intervals (Monte Carlo standard error) and mean length, with HC0;
HC1 differs by at most 0.3 points (not shown):

| g | σ | n | coefficient of x₁ | coefficient of x₂ | length | median h |
|---|---|---|---|---|---|---|
| sine | 0.5 | 200 | 93.2 (0.6) | 73.0 (1.0) | 0.148, 0.143 | 0.25 |
| sine | 0.5 | 1000 | 96.1 (0.4) | 93.0 (0.6) | 0.063, 0.062 | 0.17 |
| sine | 0.2 + 0.6u | 200 | 94.7 (0.5) | 70.9 (1.0) | 0.157, 0.150 | 0.26 |
| sine | 0.2 + 0.6u | 1000 | 94.7 (0.5) | 92.2 (0.6) | 0.066, 0.066 | 0.17 |
| logistic | 0.5 | 200 | 94.1 (0.5) | 93.4 (0.6) | 0.139, 0.139 | 0.16 |
| logistic | 0.5 | 1000 | 94.2 (0.5) | 94.3 (0.5) | 0.062, 0.062 | 0.11 |

The coefficient of x₁, whose conditional mean is linear, is covered at 93% to 96%
in every design. However, when g is a sine, the coefficient of x₂ is covered only 71%
to 73% of the time at n = 200, and 92% to 93% at n = 1000. This is likely due to the bandwidth:
cross-validation chooses it for the smooth of y in u (0.25 at n = 200), and the same
bandwidth smooths x₂, whose conditional mean sin(2πu) needs a much smaller one. The
residualized x₂ then keeps part of its conditional mean, and with the bias of the
smooth of g this biases β̂₂. With the steep logistic g, cross-validation chooses a
small bandwidth (0.16), which also suits x₂, and both coefficients are covered.

## Recommendations

- When a covariate depends on u in a nonlinear way, consider a bandwidth small
  enough for its conditional mean, for example the cross-validated bandwidth of the
  regression of that covariate on u, rather than the default; we have not yet
  studied this choice.
- A bandwidth chosen from the covariates as well is proposed in #96.
