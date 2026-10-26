# Size and power of the randomization tests of LocalRandomization (#64) in the window
# (-0.1, 0.1): x ~ U(-1, 1) and y = 0.5 + τ 1{x ≥ 0} + β x + ε with ε ~ N(0, 1), under
# the null hypothesis τ = 0 and against τ = 0.5, with a flat regression function
# (β = 0), for which local randomization holds, and with a slope (β = 2), for which it
# holds only approximately, in the spirit of Cattaneo, Frandsen and Titiunik (2015).
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["null-flat", "null-slope", "alt-flat", "alt-slope"]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  τ = startswith(a.design, "alt") ? 0.5 : 0.0
  β = endswith(a.design, "slope") ? 2.0 : 0.0
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(5, code, a.n, rep))
    x = 2 .* rand(rng, a.n) .- 1
    y = 0.5 .+ τ .* (x .>= 0) .+ β .* x .+ randn(rng, a.n)
    m = fit(LocalRandomization, x, y; window = 0.1, statistic = :all, reps = 1000, rng)
    for (k, s) in enumerate(m.statistic)
      push!(rows, (a.design, a.n, rep, s, m.ncontrol, m.ntreated, m.observed[k], m.pvalue[k], m.asymptotic[k]))
    end
  end
  writerows(a.out, ("design", "n", "rep", "statistic", "ncontrol", "ntreated", "observed", "pvalue", "asymptotic"), rows)
end

main()
