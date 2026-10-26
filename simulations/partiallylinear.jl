# Coverage of the confidence intervals of the coefficients of PartiallyLinear (Robinson
# 1988): y = 1.0 x₁ - 0.5 x₂ + g(u) + σ ε with u uniform on [0, 1], x₁ = u + v₁ and
# x₂ = sin(2πu) + v₂, so that both covariates depend on u, and g(u) = sin(2πu) or a
# steep logistic; homoskedastic errors (σ = 0.5) or heteroskedastic ones
# (σ = 0.2 + 0.6u), with the HC0 and HC1 sandwich variances.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["sine", "sine-heteroskedastic", "logistic"]
const β = [1.0, -0.5]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  g = a.design == "logistic" ? (u -> 2 / (1 + exp(-(u - 0.5) / 0.05))) : (u -> sin(2π * u))
  σ = a.design == "sine-heteroskedastic" ? (u -> 0.2 + 0.6u) : (u -> 0.5)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(14, code, a.n, rep))
    u = rand(rng, a.n)
    X = hcat(u .+ randn(rng, a.n), sin.(2π .* u) .+ randn(rng, a.n))
    y = X * β .+ g.(u) .+ σ.(u) .* randn(rng, a.n)
    m0 = fit(PartiallyLinear, u, X, y)
    m1 = fit(PartiallyLinear, u, X, y; bandwidth = m0.h, vce = HCVariance(1))
    for (name, m) in (("hc0", m0), ("hc1", m1))
      b, se = coef(m), stderror(m)
      for j in 1:2
        push!(rows, (a.design, a.n, rep, name, j, β[j], b[j], se[j], m.h))
      end
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "point", "truth", "estimate", "se", "h"), rows)
end

main()
