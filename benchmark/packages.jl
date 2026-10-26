# The time of KernelSmoothing.jl against the Julia packages that overlap with it, on
# the data of timings.jl, on one thread: KernelDensity.jl's kde (binned, with an FFT),
# and Loess.jl against local linear regression at a given bandwidth and with the
# default plug-in bandwidth. Writes a CSV to the file given as the first argument.
#
#     julia --project=ENV -t 1 packages.jl packages.csv
using KernelSmoothing, KernelDensity, Loess, Random, Distributions

best(f; reps = 5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

open(ARGS[1], "w") do io
  println(io, "call,n,seconds")
  for n in (1_000, 10_000, 100_000, 1_000_000)
    rng = Xoshiro(n)
    x = rand(rng, Beta(2, 2), n)
    y = sin.(2π .* x) .+ 0.5 .* randn(rng, n)
    g = collect(range(0, 1; length = 512))
    h = KernelDensity.default_bandwidth(x)
    cases = Any[
      ("KernelDensity.kde(x), Silverman, then pdf on 512 points", () -> pdf(kde(x), g)),
      ("KernelSmoothing fit(KDE, x; bandwidth = SilvermanRule()), pdf on 512 points",
       () -> pdf(fit(KDE, x; bandwidth = SilvermanRule()), g)),
      ("KernelSmoothing fit(KDE, x; bandwidth = h, kernel = GaussianKernel()), pdf on 512 points",
       () -> pdf(fit(KDE, x; bandwidth = h, kernel = GaussianKernel()), g)),
      ("KernelSmoothing fit(KDE, x; bandwidth = h, kernel = GaussianKernel(), engine = BinnedSums()), pdf on 512 points",
       () -> pdf(fit(KDE, x; bandwidth = h, kernel = GaussianKernel(), engine = BinnedSums()), g)),
      ("KernelSmoothing fit(KDE, x) (Sheather-Jones), pdf on 512 points", () -> pdf(fit(KDE, x), g)),
    ]
    if n <= 100_000
      pts = collect(range(0.05, 0.95; length = 30))
      push!(cases, ("Loess.loess(x, y; span = 0.2), predict at 30 points", () -> predict(loess(x, y; span = 0.2), pts)))
      push!(cases, ("KernelSmoothing local_polynomial(x, y, points; bandwidth = 0.1)",
                    () -> local_polynomial(x, y, pts; bandwidth = 0.1)))
      push!(cases, ("KernelSmoothing fit(LocalPolynomialRegression, x, y; bandwidth = 0.1, points)",
                    () -> fit(LocalPolynomialRegression, x, y; bandwidth = 0.1, points = pts)))
    end
    for (name, f) in cases
      t = try
        best(f)
      catch err
        @warn "failed" name n err
        NaN
      end
      println(io, "\"", name, "\",", n, ",", t)
      println(rpad(name, 90), lpad(n, 8), "  ", round(t; sigdigits = 3), " s")
    end
  end
end
