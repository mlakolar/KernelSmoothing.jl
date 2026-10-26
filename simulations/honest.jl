# Coverage and length of the honest intervals of Armstrong and Kolesár (2018, 2020) for
# local linear regression, with the true bound M on |m''|, a bound half as large, and
# the rule of thumb for M, against robust bias-corrected intervals.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["sine", "peak"]
const SETUP = Dict("sine" => (x -> sin(2π * x), 0.5, 4π^2), "peak" => (x -> exp(-32 * (x - 0.5)^2) + 0.5x, 0.3, 64.0))
const POINTS = [0.1, 0.25, 0.5, 0.75, 0.9]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  m, σ, M = SETUP[a.design]
  truth = m.(POINTS)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(9, code, a.n, rep))
    x = rand(rng, a.n)
    y = m.(x) .+ σ .* randn(rng, a.n)
    d = SmoothingData(x, y)
    record(name, f, ci) = for j in eachindex(POINTS)
      push!(rows, (a.design, a.n, rep, name, POINTS[j], truth[j], ci[j, 1], ci[j, 2], f.h[j]))
    end
    f = fit(LocalPolynomialRegression, d; points = POINTS, bandwidth = MSEOptimal())
    record("rbc-mse", f, confint(f))
    for (name, Mb) in (("true-M", M), ("half-M", M / 2), ("rot-M", :rot))
      fl = fit(LocalPolynomialRegression, d; points = POINTS, bandwidth = FLCIOptimal(Mb))
      record("honest-" * name, fl, confint(fl; method = Honest(Mb)))
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "lower", "upper", "h"), rows)
end

main()
