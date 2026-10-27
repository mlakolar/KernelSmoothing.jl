# Integrated squared error of bivariate kernel density estimates with the bandwidth
# matrices of MultivariateKDE (#60): normal scale, plug-in, smoothed cross-validation
# and least-squares cross-validation, full and diagonal, for a correlated normal, a
# bimodal mixture and a skewed mixture, in the spirit of the bivariate targets of Wand
# and Jones (1993). The integral is the trapezoid rule on a 101 × 101 grid. Keep
# binned = false so that the historical study uses exact selection above the new
# binning thresholds as well.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["correlated", "bimodal", "skewed"]
mv(μ, Σ) = MvNormal(μ, Σ)
const TARGETS = Dict(
  "correlated" => MixtureModel([mv([0.0, 0.0], [1.0 0.7; 0.7 1.0])], [1.0]),
  "bimodal" => MixtureModel([mv([-1.0, 0.0], [0.25 0.0; 0.0 0.25]), mv([1.0, 0.0], [0.25 0.0; 0.0 0.25])], [0.5, 0.5]),
  "skewed" => MixtureModel([mv([0.0, 0.0], [1.0 0.0; 0.0 1.0]), mv([0.5, 0.5], [0.16 0.0; 0.0 0.16]),
                            mv([1.0, 1.0], [0.04 0.0; 0.0 0.04])], [0.2, 0.2, 0.6]))
const RULES = [("normal-scale", NormalScaleMatrix()), ("plugin", PlugInMatrix(binned = false)), ("scv", SCVMatrix(binned = false)),
               ("lscv", LSCVMatrix(binned = false)), ("normal-scale-diag", NormalScaleMatrix(diagonal = true)),
               ("plugin-diag", PlugInMatrix(diagonal = true, binned = false)), ("scv-diag", SCVMatrix(diagonal = true, binned = false)),
               ("lscv-diag", LSCVMatrix(diagonal = true, binned = false))]
const G = collect(range(-3.5, 3.5; length = 101))

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  D = TARGETS[a.design]
  truth = [pdf(D, [u, v]) for u in G, v in G]
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(13, code, a.n, rep))
    X = rand(rng, D, a.n)
    for (name, rule) in RULES
      t = @elapsed f = fit(MultivariateKDE, X; bandwidth = rule)
      e = (pdf(f, (G, G)) .- truth) .^ 2
      w = step(range(-3.5, 3.5; length = 101))
      ise = w^2 * (sum(e) - (sum(e[1, :]) + sum(e[end, :]) + sum(e[:, 1]) + sum(e[:, end])) / 2 +
                   (e[1, 1] + e[1, end] + e[end, 1] + e[end, end]) / 4)
      H = f.H
      push!(rows, (a.design, a.n, rep, name, ise, H[1, 1], H[1, 2], H[2, 2], t))
    end
  end
  writerows(a.out, ("design", "n", "rep", "rule", "ise", "h11", "h12", "h22", "seconds"), rows)
end

main()
