# Size and power of the manipulation test of DensityDiscontinuity (Cattaneo, Jansson
# and Ma 2020) at the cutoff 0: x normal, or 2 Beta(2, 4) - 1 as in the regression
# discontinuity designs, with the density below the cutoff thinned by the factor
# 1/(1 + δ), so that it jumps by the ratio 1 + δ; δ = 0 gives the size.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = [b * "-" * d for b in ("normal", "beta") for d in ("0", "0.1", "0.25", "0.5")]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  base, δs = rsplit(a.design, "-"; limit = 2)
  δ = parse(Float64, δs)
  D = base == "normal" ? Normal() : LocationScale(-1.0, 2.0, Beta(2, 4))
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(11, code, a.n, rep))
    x = Float64[]
    while length(x) < a.n
      t = rand(rng, D)
      (t >= 0 || rand(rng) < 1 / (1 + δ)) && push!(x, t)
    end
    m = fit(DensityDiscontinuity, x)
    push!(rows, (a.design, a.n, rep, m.statistic, m.pvalue, m.h[1], m.h[2]))
  end
  writerows(a.out, ("design", "n", "rep", "statistic", "pvalue", "hleft", "hright"), rows)
end

main()
