# The time of the default call of each estimator of the package, at sample sizes from
# 10³ to 10⁶, on one thread: the minimum of `reps` runs after a warm-up, with the time
# of preparing the data (`SmoothingData`, which sorts it) for comparison. Writes a CSV
# to the file given as the first argument; the second, if given, caps the sample size.
#
#     julia --project=. -t 1 timings.jl timings.csv [NMAX]
using KernelSmoothing
using Distributions
using Random

const NS = [1_000, 10_000, 100_000, 1_000_000]

# the time of f(), the minimum over runs taking at least `budget` seconds in all
function timeit(f; budget = 2.0, maxreps = 20)
  f()
  best, total, reps = Inf, 0.0, 0
  while reps < maxreps && (reps < 3 || total < budget)
    t = @elapsed f()
    best = min(best, t)
    total += t
    reps += 1
    (t > 20 || total > 60) && break
  end
  return best, reps
end

# name, the largest n, and a function of (x, y, rng) that runs the call
function cases()
  pts = collect(range(0.1, 0.9; length = 30))
  return [
    ("SmoothingData(x, y)", 10^6, (x, y, b, X2) -> SmoothingData(x, y)),
    ("fit(KDE, x)", 10^6, (x, y, b, X2) -> fit(KDE, x)),
    ("fit(KDE, x; bandwidth = SheatherJones())", 10^6, (x, y, b, X2) -> fit(KDE, x; bandwidth = SheatherJones())),
    ("fit(KDE, x; bandwidth = LSCV())", 10^5, (x, y, b, X2) -> fit(KDE, x; bandwidth = LSCV())),
    ("fit(PointwiseDensity, x)", 10^6, (x, y, b, X2) -> fit(PointwiseDensity, x)),
    ("fit(LocalPolynomialDensity, x)", 10^5, (x, y, b, X2) -> fit(LocalPolynomialDensity, x)),
    ("fit(LocalPolynomialRegression, x, y)", 10^6, (x, y, b, X2) -> fit(LocalPolynomialRegression, x, y)),
    ("fit(LocalPolynomialRegression, x, y; bandwidth = IMSEOptimal())", 10^6,
     (x, y, b, X2) -> fit(LocalPolynomialRegression, x, y; bandwidth = IMSEOptimal())),
    ("fit(LocalPolynomialRegression, x, y; bandwidth = CrossValidation())", 10^5,
     (x, y, b, X2) -> fit(LocalPolynomialRegression, x, y; bandwidth = CrossValidation())),
    ("conf_band(fit(LocalPolynomialRegression, x, y))", 10^6,
     (x, y, b, X2) -> conf_band(fit(LocalPolynomialRegression, x, y); rng = Xoshiro(1))),
    ("fit(RegressionDiscontinuity, x .- 0.5, y)", 10^6, (x, y, b, X2) -> fit(RegressionDiscontinuity, x .- 0.5, y)),
    ("fit(DensityDiscontinuity, x .- 0.5)", 10^5, (x, y, b, X2) -> fit(DensityDiscontinuity, x .- 0.5)),
    ("fit(LocalLikelihood, x, b)", 10^4, (x, y, b, X2) -> fit(LocalLikelihood, x, b)),
    ("fit(LocalLikelihood, x, b; bandwidth = RuleOfThumb())", 10^6,
     (x, y, b, X2) -> fit(LocalLikelihood, x, b; bandwidth = RuleOfThumb())),
    ("fit(LocalLikelihoodDensity, x)", 10^5, (x, y, b, X2) -> fit(LocalLikelihoodDensity, x)),
    ("fit(LocalQuantileRegression, x, y)", 10^5, (x, y, b, X2) -> fit(LocalQuantileRegression, x, y)),
    ("fit(ConditionalDensity, x, y)", 10^4, (x, y, b, X2) -> fit(ConditionalDensity, x, y)),
    ("fit(BoundedKDE, x; support = (0, 1))", 10^6, (x, y, b, X2) -> fit(BoundedKDE, x; support = (0, 1))),
    ("fit(BoundedKDE, x; support = (0, 1), method = BetaKernels())", 10^4,
     (x, y, b, X2) -> fit(BoundedKDE, x; support = (0, 1), method = BetaKernels())),
    ("fit(MultivariateKDE, X2)", 10^5, (x, y, b, X2) -> fit(MultivariateKDE, X2)),
    ("fit(MultivariateKDE, X2; bandwidth = PlugInMatrix(binned = false))", 10^4,
     (x, y, b, X2) -> fit(MultivariateKDE, X2; bandwidth = PlugInMatrix(binned = false))),
    ("fit(MultivariateKDE, X2; bandwidth = PlugInMatrix())", 10^4,
     (x, y, b, X2) -> fit(MultivariateKDE, X2; bandwidth = PlugInMatrix())),
    ("fit(MultivariateKDE, X2; bandwidth = SCVMatrix())", 10^4,
     (x, y, b, X2) -> fit(MultivariateKDE, X2; bandwidth = SCVMatrix())),
    ("fit(MultivariateKDE, X2; bandwidth = SCVMatrix(binned = false))", 10^4,
     (x, y, b, X2) -> fit(MultivariateKDE, X2; bandwidth = SCVMatrix(binned = false))),
    ("fit(PartiallyLinear, x, z, y)", 10^5, (x, y, b, X2) -> fit(PartiallyLinear, x, reshape(X2[2, :], :, 1), y)),
  ]
end

function main()
  out = ARGS[1]
  nmax = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10^6
  open(out, "w") do io
    println(io, "call,n,seconds,runs,threads")
    for n in NS
      n > nmax && continue
      rng = Xoshiro(n)
      x = rand(rng, Beta(2, 2), n)
      y = sin.(2π .* x) .+ 0.5 .* randn(rng, n)
      b = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-2 .* sin.(2π .* x))))
      X2 = permutedims(hcat(x, 0.5 .* x .+ 0.3 .* randn(rng, n)))   # observations as columns
      for (name, cap, f) in cases()
        n > cap && continue
        t, reps = try
          timeit(() -> f(x, y, b, X2))
        catch err
          @warn "failed" name n err
          (NaN, 0)
        end
        println(io, "\"", name, "\",", n, ",", t, ",", reps, ",", Threads.nthreads())
        flush(io)
        println(rpad(name, 70), lpad(n, 8), "  ", round(t; sigdigits = 3), " s")
      end
    end
  end
end

main()
