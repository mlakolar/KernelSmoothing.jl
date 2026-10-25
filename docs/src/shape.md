# [Shape constraints](@id shape-page)

Many applications know the sign of a derivative of the function they estimate: a
dose response that increases, a hazard that is convex, a distribution function. A
kernel estimate need not respect it, and the wiggles that violate the shape are
noise that the bandwidth alone cannot remove without adding bias. Two operations
impose the shape on the estimates of any fit at its points.

## Rearrangement

The increasing rearrangement of Chernozhukov, Fernández-Val and Galichon (2009)
sorts the estimates ``\hat f(t_1), \dots, \hat f(t_m)`` at sorted points in increasing
order. For an increasing function ``f``, the rearranged estimate ``\hat f^*`` is never
farther from it than ``\hat f``,

```math
\Big(\sum_k |\hat f^*(t_k) - f(t_k)|^p\Big)^{1/p} \le \Big(\sum_k |\hat f(t_k) - f(t_k)|^p\Big)^{1/p}
\quad \text{for every } p \ge 1,
```

with strict improvement wherever ``\hat f`` is not monotone. The rearrangement is
with respect to the counting measure on the points, so that for a function on an
interval the points should be equally spaced. Rearranging the lower and the upper
ends of a confidence band separately gives a band that covers an increasing
function whenever the original band does, and is never wider (their
Proposition 2). `rearrange(m)` works for every fit with points and estimates, and
`confint` and `conf_band` of the result rearrange the intervals and bands of the
fit.

## Projection onto a cone of shapes

`constrain(m, shape)` projects the estimates onto the sequences of a shape in the
weighted least-squares norm,

```math
\hat\theta = \arg\min_{\theta \in \mathcal{C}} \sum_{k=1}^m v_k (\theta_k - \hat f(t_k))^2,
```

where ``\mathcal{C}`` is the cone of increasing or decreasing sequences, of convex or
concave ones (nonnegative or nonpositive second divided differences), or of
monotone convex or concave ones: smoothing followed by an isotonic or convex
regression, as Mammen (1991) proposed for monotone regression. Since ``\mathcal{C}``
is a closed convex cone, the projection is never farther from a sequence of the
shape than the estimates. Every sequence of a shape is a linear part plus a
nonnegative combination of generators, such as the steps ``1\{t \ge t_j\}`` for
increasing sequences and the hinges ``(t - t_j)_+`` for convex ones, so that the
projection is a nonnegative least-squares problem, which the active-set method of
Lawson and Hanson (1974) solves exactly. `shape_projection(values, points; shape,
weights)` is the projection itself.

## Example

```julia
using KernelSmoothing

x = rand(500); y = x .^ 2 .+ 0.2 .* randn(500)
m = fit(LocalPolynomialRegression, x, y)
r = rearrange(m)                     # increasing estimates
coef(r), conf_band(r)                # the rearranged band
c = constrain(m, :increasing_convex) # the projection onto increasing convex sequences
coef(c)
```

## Reference

```@docs
rearrange
Rearranged
constrain
ShapeConstrained
shape_projection
```
