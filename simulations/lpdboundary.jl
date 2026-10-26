# Coverage of the intervals of LocalPolynomialDensity (Cattaneo, Jansson and Ma 2020)
# at and near a boundary of the support, where the kernel density estimate is biased:
# exponential and Beta(1, 3) samples, at the boundary 0 and at interior points, with
# the default (robust bias-corrected) and the conventional intervals.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["exponential", "beta13"]
const POINTS = [0.0, 0.05, 0.2, 0.5]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  D = a.design == "exponential" ? Exponential() : Beta(1, 3)
  truth = pdf.(D, POINTS)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(12, code, a.n, rep))
    m = fit(LocalPolynomialDensity, rand(rng, D, a.n); points = POINTS)
    for (name, method) in (("conventional", Conventional()), ("rbc", RobustBiasCorrected()))
      est, se = coef(m; method), stderror(m; method)
      for j in eachindex(POINTS)
        push!(rows, (a.design, a.n, rep, name, POINTS[j], truth[j], est[j], se[j], m.h[j]))
      end
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "estimate", "se", "h"), rows)
end

main()
