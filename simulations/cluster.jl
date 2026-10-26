# Coverage of robust bias-corrected intervals of local linear regression with clustered
# data: G clusters of 25 observations (balanced) or of sizes from 5 to 45 (unbalanced),
# with x = 0.5 u_g + 0.5 v (u_g, v ~ U(0, 1)) and errors 0.5 (√ρ a_g + √(1 - ρ) e),
# ρ = 0.5, so that both the covariate and the errors are correlated within clusters.
# The variances are CR1, CR2 and CR3, and HC1, which ignores the clusters, at the same
# bandwidth.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["balanced-20", "balanced-50", "balanced-200", "unbalanced-20", "unbalanced-50", "unbalanced-200"]
const POINTS = [0.3, 0.5, 0.7]

function main()
  a = studyargs()   # here n is unused; the sample size follows from the clusters
  code = designcode(DESIGNS, a.design)
  kind, Gs = split(a.design, "-")
  G = parse(Int, Gs)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(8, code, a.n, rep))
    sizes = kind == "balanced" ? fill(25, G) : rand(rng, 5:45, G)
    g = vcat([fill(k, s) for (k, s) in enumerate(sizes)]...)
    u, ag = rand(rng, G), randn(rng, G)
    x = 0.5 .* u[g] .+ 0.5 .* rand(rng, length(g))
    y = sin.(2π .* x) .+ 0.5 .* (sqrt(0.5) .* ag[g] .+ sqrt(0.5) .* randn(rng, length(g)))
    h = fit(LocalPolynomialRegression, x, y; cluster = g, vce = ClusterVariance(1), points = POINTS).h
    for (name, vce) in (("cr1", ClusterVariance(1)), ("cr2", ClusterVariance(2)), ("cr3", ClusterVariance(3)),
                        ("hc1", HCVariance(1)))
      f = vce isa ClusterVariance ? fit(LocalPolynomialRegression, x, y; cluster = g, vce, points = POINTS, bandwidth = h) :
          fit(LocalPolynomialRegression, x, y; vce, points = POINTS, bandwidth = h)
      for j in eachindex(POINTS)
        push!(rows, (a.design, length(g), rep, "rbc-" * name, POINTS[j], sin(2π * POINTS[j]), f.estimate_bc[j],
                     f.se_rbc[j], f.h[j]))
      end
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "estimate", "se", "h"), rows)
end

main()
