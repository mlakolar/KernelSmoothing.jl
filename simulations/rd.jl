# Coverage of the conventional and robust bias-corrected intervals of sharp regression
# discontinuity with rdrobust's MSE- and coverage-optimal bandwidths, for the designs
# of Calonico, Cattaneo and Titiunik (2014, Models 1 and 2), which mimic Lee (2008)
# and Ludwig and Miller (2007): x ~ 2 Beta(2, 4) - 1 and fifth-degree polynomials on
# the two sides of the cutoff 0, with ε ~ N(0, 0.1295²).
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["lee", "ludwig-miller"]
const COEFS = Dict(
  "lee" => ([0.48, 1.27, 7.18, 20.21, 21.54, 7.33], [0.52, 0.84, -3.00, 7.99, -9.01, 3.56]),
  "ludwig-miller" => ([3.71, 2.30, 3.28, 1.45, 0.23, 0.03], [0.26, 18.49, -54.81, 74.30, -45.02, 9.83]))

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  cl, cr = COEFS[a.design]
  τ = cr[1] - cl[1]
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(10, code, a.n, rep))
    x = 2 .* rand(rng, Beta(2, 4), a.n) .- 1
    y = [xi < 0 ? evalpoly(xi, cl) : evalpoly(xi, cr) for xi in x] .+ 0.1295 .* randn(rng, a.n)
    for (name, rule) in (("mse", RDOptimal(:mserd)), ("cer", RDOptimal(:cerrd)))
      f = fit(RegressionDiscontinuity, x, y; bandwidth = rule)
      push!(rows, (a.design, a.n, rep, "conventional-" * name, τ, f.estimate, f.se, f.h[1], f.h[2]))
      push!(rows, (a.design, a.n, rep, "rbc-" * name, τ, f.estimate_bc, f.se_rbc, f.h[1], f.h[2]))
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "truth", "estimate", "se", "hleft", "hright"), rows)
end

main()
