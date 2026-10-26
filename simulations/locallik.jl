# Coverage of the conventional and the robust bias-corrected intervals of local
# likelihood (#65), with cross-validated and rule-of-thumb bandwidths, for the logit,
# Poisson, gamma and negative binomial families: x ~ U(0, 1) and a linear predictor
# η(x) of period one, in the spirit of Fan, Heckman and Wand (1995).
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["logit-sine", "poisson-sine", "gamma-sine", "negbin-sine"]
const POINTS = [0.1, 0.25, 0.5, 0.75, 0.9]

function draw(rng, design, n)
  x = rand(rng, n)
  if design == "logit-sine"
    η = 2 .* sin.(2π .* x)
    return x, Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-η))), :logit, x0 -> 2sin(2π * x0)
  elseif design == "poisson-sine"
    η = 1 .+ sin.(2π .* x)
    return x, Float64.(rand.(rng, Poisson.(exp.(η)))), :poisson, x0 -> 1 + sin(2π * x0)
  elseif design == "gamma-sine"
    μ = exp.(0.5 .+ sin.(2π .* x))
    return x, rand.(rng, Gamma.(2, μ ./ 2)), :gamma, x0 -> 0.5 + sin(2π * x0)
  else   # negative binomial with θ = 2: variance μ + μ²/2
    μ = exp.(1 .+ sin.(2π .* x))
    return x, Float64.(rand.(rng, NegativeBinomial.(2, 2 ./ (2 .+ μ)))), :negative_binomial, x0 -> 1 + sin(2π * x0)
  end
end

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(1, code, a.n, rep))
    x, y, family, η = draw(rng, a.design, a.n)
    theta = family === :negative_binomial ? 2.0 : nothing
    for (bw, rule) in (("cv", CrossValidation()), ("rot", RuleOfThumb()))
      t = @elapsed m = fit(LocalLikelihood, x, y; family, theta, bandwidth = rule, points = POINTS)
      for (j, x0) in enumerate(POINTS)
        push!(rows, (a.design, a.n, rep, bw, x0, η(x0), m.eta[j], m.se[j], m.eta_bc[j], m.se_rbc[j], m.h[j], t))
      end
    end
  end
  writerows(a.out, ("design", "n", "rep", "bandwidth", "point", "truth", "eta", "se", "eta_bc", "se_rbc", "h", "seconds"), rows)
end

main()
