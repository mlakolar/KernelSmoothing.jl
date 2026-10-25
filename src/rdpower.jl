## Power calculations for regression discontinuity designs

"""
    rd_power(m::RegressionDiscontinuity; tau, alpha = 0.05, n = nobs(m))

The power of the robust bias-corrected test of no effect at level `alpha` against
the effect `tau` (Cattaneo, Titiunik and Vazquez-Bare 2019), as rdpower's
`power.rbc`:

    1 - Φ(z - τ/se) + Φ(-z - τ/se),   z = Φ⁻¹(1 - α/2),

where `se` is the robust standard error of the fit `m`, the standard deviation of
the bias-corrected estimate. With `n`, the power for a sample of `n` observations
spread as in the data, analysed with the same bandwidths, whose standard error is
`se √(nobs(m)/n)`.
"""
function rd_power(m::RegressionDiscontinuity; tau::Real, alpha::Real = 0.05, n::Real = nobs(m))
  0 < alpha < 1 || throw(ArgumentError("alpha must be between 0 and 1, got $alpha"))
  n > 0 || throw(ArgumentError("the sample size must be positive, got $n"))
  isfinite(m.se_rbc) || return NaN
  z = Distributions.quantile(Distributions.Normal(), 1 - alpha / 2)
  t = tau / (m.se_rbc * sqrt(nobs(m) / n))
  N = Distributions.Normal()
  return Distributions.ccdf(N, z - t) + Distributions.cdf(N, -z - t)
end

"""
    rd_samplesize(m::RegressionDiscontinuity; tau, power = 0.8, alpha = 0.05)

The smallest number of observations, spread as in the data and analysed with the
same bandwidths, for which the robust bias-corrected test of no effect at level
`alpha` has the given `power` against the effect `tau`, by [`rd_power`](@ref): the
total sample size of rdpower's `rdsampsi`.
"""
function rd_samplesize(m::RegressionDiscontinuity; tau::Real, power::Real = 0.8, alpha::Real = 0.05)
  0 < power < 1 || throw(ArgumentError("power must be between 0 and 1, got $power"))
  0 < alpha < 1 || throw(ArgumentError("alpha must be between 0 and 1, got $alpha"))
  tau != 0 || throw(ArgumentError("the effect tau must not be zero"))
  power > alpha || throw(ArgumentError("the power must exceed the level alpha"))
  isfinite(m.se_rbc) || throw(ArgumentError("the fit has no robust standard error"))
  # the power increases with n; start from the approximation that ignores the far tail
  N = Distributions.Normal()
  z = Distributions.quantile(N, 1 - alpha / 2) + Distributions.quantile(N, power)
  n = max(1, ceil(Int, nobs(m) * (z * m.se_rbc / tau)^2))
  while n > 1 && rd_power(m; tau, alpha, n = n - 1) >= power
    n -= 1
  end
  while rd_power(m; tau, alpha, n) < power
    n += 1
  end
  return n
end
