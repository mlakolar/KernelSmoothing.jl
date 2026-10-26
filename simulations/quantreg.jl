# Coverage of the pointwise intervals of local polynomial quantile regression (#62),
# which ignore the bias, with the default bandwidth of Yu and Jones (1998):
# x ~ U(0, 1) and y = sin(2πx) + σ(x) ε with heteroskedastic normal errors or with
# t₃ errors, for the quartiles and the median.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = [d * "-" * t for d in ("sine-het", "sine-t3") for t in ("0.25", "0.5", "0.75")]
const POINTS = [0.1, 0.25, 0.5, 0.75, 0.9]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  base, τs = rsplit(a.design, "-"; limit = 2)
  τ = parse(Float64, τs)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(2, code, a.n, rep))
    x = rand(rng, a.n)
    if base == "sine-het"
      y = sin.(2π .* x) .+ (0.5 .+ x) .* randn(rng, a.n)
      q = x0 -> sin(2π * x0) + (0.5 + x0) * quantile(Normal(), τ)
    else
      y = sin.(2π .* x) .+ 0.5 .* rand(rng, TDist(3), a.n)
      q = x0 -> sin(2π * x0) + 0.5 * quantile(TDist(3), τ)
    end
    t = @elapsed m = fit(LocalQuantileRegression, x, y; tau = τ, points = POINTS)
    est, se = coef(m), stderror(m)
    for (j, x0) in enumerate(POINTS)
      push!(rows, (a.design, a.n, rep, x0, q(x0), est[j], se[j], bandwidth(m), t))
    end
  end
  writerows(a.out, ("design", "n", "rep", "point", "truth", "estimate", "se", "h", "seconds"), rows)
end

main()
