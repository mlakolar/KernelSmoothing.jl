# Coverage of the conventional and robust bias-corrected pointwise intervals and the
# uniform bands of kernel density estimates at interior points, with the MSE- and
# IMSE-optimal bandwidths and the bandwidth of Sheather and Jones.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["normal", "bimodal", "t5"]
const DISTRIBUTIONS = Dict("normal" => Normal(), "bimodal" => MixtureModel([Normal(-1, 0.5), Normal(1, 0.5)]),
                           "t5" => TDist(5))
const POINTS = [-1.5, -0.5, 0.0, 0.5, 1.5]
const GRID = collect(range(-2, 2; length = 21))

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  D = DISTRIBUTIONS[a.design]
  truth = pdf.(D, POINTS)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(7, code, a.n, rep))
    d = SmoothingData(rand(rng, D, a.n))
    hsj = bandwidth(SheatherJones(), d, EpanechnikovKernel())
    for (name, rule, kw) in (("mse", MSEOptimal(), (;)), ("imse", IMSEOptimal(), (;)), ("sj", hsj, (minobs = 0,)))
      f = fit(PointwiseDensity, d; points = POINTS, bandwidth = rule, kw...)
      for j in eachindex(POINTS)
        push!(rows, (a.design, a.n, rep, "conventional-" * name, POINTS[j], truth[j], f.estimate[j], f.se[j], f.h[j]))
        push!(rows, (a.design, a.n, rep, "rbc-" * name, POINTS[j], truth[j], f.estimate_bc[j], f.se_rbc[j], f.h[j]))
      end
    end
    g = fit(PointwiseDensity, d; points = GRID, bandwidth = hsj, minobs = 0)
    tg = pdf.(D, GRID)
    for (name, method) in (("band-rbc", RobustBiasCorrected()), ("band-conventional", Conventional()))
      b = conf_band(g; method, draws = 2000, rng)
      push!(rows, (a.design, a.n, rep, name, NaN, Float64(all(b.lower .<= tg .<= b.upper)),
                   mean(b.upper .- b.lower), b.critical, hsj))
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "estimate", "se", "h"), rows)
end

main()
