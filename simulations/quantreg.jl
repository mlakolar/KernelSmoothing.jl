# Coverage of conventional and robust bias-corrected pointwise intervals of local
# polynomial quantile regression (#62, #94), with the bandwidth of Yu and Jones (1998):
# x ~ U(0, 1) and y = sin(2πx) + σ(x) ε with heteroskedastic normal errors or with
# t₃ errors, for the quartiles and the median.
# The qy2 design adapts Qu and Yoon (2019, Section 8, equation (21), Model 2),
# with x ~ U(-1, 1) and normal errors. It is used here for pointwise checks only.
include(joinpath(@__DIR__, "common.jl"))

# Keep the sine designs first: their positions determine the seeds.
const DESIGNS = vcat([d * "-" * t for d in ("sine-het", "sine-t3") for t in ("0.25", "0.5", "0.75")],
                     ["qy2-" * t for t in ("0.2", "0.5", "0.8")])
const POINTS = [0.1, 0.25, 0.5, 0.75, 0.9]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  base, τs = rsplit(a.design, "-"; limit = 2)
  τ = parse(Float64, τs)
  points = base == "qy2" ? collect(-0.8:0.4:0.8) : POINTS
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(2, code, a.n, rep))
    x = rand(rng, a.n)
    if base == "sine-het"
      y = sin.(2π .* x) .+ (0.5 .+ x) .* randn(rng, a.n)
      q = x0 -> sin(2π * x0) + (0.5 + x0) * quantile(Normal(), τ)
    elseif base == "sine-t3"
      y = sin.(2π .* x) .+ 0.5 .* rand(rng, TDist(3), a.n)
      q = x0 -> sin(2π * x0) + 0.5 * quantile(TDist(3), τ)
    else
      x = 2 .* x .- 1
      y = 0.5 .+ x .+ x.^2 .+ sin.(π .* x .- 1) .+ (x .+ 1.25) .* randn(rng, a.n)
      q = x0 -> 0.5 + x0 + x0^2 + sin(π * x0 - 1) + (x0 + 1.25) * quantile(Normal(), τ)
    end
    t = @elapsed begin
      m = fit(LocalQuantileRegression, x, y; tau = τ, points = points)
      est, se = coef(m; method = Conventional()), stderror(m; method = Conventional())
      est_bc = coef(m; method = RobustBiasCorrected())
      se_rbc = stderror(m; method = RobustBiasCorrected())
    end
    for (j, x0) in enumerate(points)
      push!(rows, (a.design, a.n, rep, x0, q(x0), est[j], se[j], est_bc[j], se_rbc[j], bandwidth(m), m.b, t))
    end
  end
  writerows(a.out, ("design", "n", "rep", "point", "truth", "estimate", "se", "estimate_bc", "se_rbc", "h", "b", "seconds"), rows)
end

main()
