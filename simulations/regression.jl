# Coverage of the conventional and robust bias-corrected pointwise intervals and the
# uniform bands of local linear regression, with the MSE-, IMSE- and coverage-optimal
# bandwidths: y = m(x) + σ(x) ε with ε ~ N(0, 1), for uniform covariates and for normal
# and t₃ ones, whose tails thin the data at the ends.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["sine", "sine-heteroskedastic", "peak", "normal-sine", "t3-sine"]
const SETUP = Dict(
  "sine" => (Uniform(), x -> sin(2π * x), x -> 0.5, [0.1, 0.25, 0.5, 0.75, 0.9]),
  "sine-heteroskedastic" => (Uniform(), x -> sin(2π * x), x -> 0.2 + 0.6x, [0.1, 0.25, 0.5, 0.75, 0.9]),
  "peak" => (Uniform(), x -> exp(-32 * (x - 0.5)^2) + 0.5x, x -> 0.3, [0.1, 0.25, 0.5, 0.75, 0.9]),
  "normal-sine" => (Normal(), x -> sin(2x), x -> 0.5, [-1.5, -0.75, 0.0, 0.75, 1.5]),
  "t3-sine" => (TDist(3), x -> sin(2x), x -> 0.5, [-1.5, -0.75, 0.0, 0.75, 1.5]))

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  xdist, m, σ, pts = SETUP[a.design]
  grid = collect(range(pts[1], pts[end]; length = 21))
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(6, code, a.n, rep))
    x = rand(rng, xdist, a.n)
    y = m.(x) .+ σ.(x) .* randn(rng, a.n)
    d = SmoothingData(x, y)
    truth = m.(pts)
    for (name, rule) in (("mse", MSEOptimal()), ("imse", IMSEOptimal()), ("ce", CoverageOptimal()))
      f = fit(LocalPolynomialRegression, d; points = pts, bandwidth = rule)
      for j in eachindex(pts)
        push!(rows, (a.design, a.n, rep, "conventional-" * name, pts[j], truth[j], f.estimate[j], f.se[j], f.h[j]))
        push!(rows, (a.design, a.n, rep, "rbc-" * name, pts[j], truth[j], f.estimate_bc[j], f.se_rbc[j], f.h[j]))
      end
    end
    # uniform bands on a grid at the IMSE-optimal bandwidth: the coverage of the whole
    # function, with the critical value in the column of the standard error
    g = fit(LocalPolynomialRegression, d; points = grid, bandwidth = IMSEOptimal())
    tg = m.(grid)
    for (name, method) in (("band-rbc", RobustBiasCorrected()), ("band-conventional", Conventional()))
      b = conf_band(g; method, draws = 2000, rng)
      push!(rows, (a.design, a.n, rep, name, NaN, Float64(all(b.lower .<= tg .<= b.upper)),
                   mean(b.upper .- b.lower), b.critical, g.h[1]))
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "estimate", "se", "h"), rows)
end

main()
