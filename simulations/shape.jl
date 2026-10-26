# Integrated squared error and coverage of local polynomial regression with and without
# the rearrangement and the projection onto increasing sequences (#63), for two
# increasing regression functions and one that is not monotone: x ~ U(0, 1) and
# y = m(x) + 0.3 ε, at the default points of the fit, as in Chernozhukov,
# Fernández-Val and Galichon (2009).
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["logistic", "flat", "sine"]
const FUNCTIONS = Dict("logistic" => x -> 1 / (1 + exp(-(x - 0.5) / 0.08)),   # steep in the middle
                       "flat" => x -> 0.2x,                                  # nearly constant
                       "sine" => x -> 0.5 * sin(2π * x))                     # not monotone

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  f = FUNCTIONS[a.design]
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(3, code, a.n, rep))
    x = rand(rng, a.n)
    y = f.(x) .+ 0.3 .* randn(rng, a.n)
    m = fit(LocalPolynomialRegression, x, y)
    truth = f.(m.points)
    r = rearrange(m)
    c = constrain(m, :increasing)
    ise(est) = mean(abs2, est .- truth)
    ci, cr = confint(m), confint(r)
    s = rand(rng, UInt64)
    b = conf_band(m; draws = 2000, rng = Xoshiro(s))
    br = conf_band(r; draws = 2000, rng = Xoshiro(s))
    covered(lo, hi) = mean(lo .<= truth .<= hi)
    push!(rows, (a.design, a.n, rep, ise(coef(m)), ise(coef(r)), ise(coef(c)),
                 covered(ci[:, 1], ci[:, 2]), covered(cr[:, 1], cr[:, 2]),
                 all(b.lower .<= truth .<= b.upper), all(br.lower .<= truth .<= br.upper),
                 mean(b.upper .- b.lower), mean(br.upper .- br.lower), issorted(coef(m))))
  end
  writerows(a.out, ("design", "n", "rep", "ise_fit", "ise_rearranged", "ise_constrained", "pointwise_fit",
                    "pointwise_rearranged", "band_fit", "band_rearranged", "width_fit", "width_rearranged",
                    "fit_monotone"), rows)
end

main()
