# Coverage of the confidence intervals of the coefficients of PartiallyLinear (Robinson
# 1988): y = 1.0 x₁ - 0.5 x₂ + g(u) + σ ε with u uniform on [0, 1], x₁ = u + v₁ and
# x₂ = sin(2πu) + v₂, with independent standard normal v₁, v₂ and ε. The original
# designs have g(u) = sin(2πu) or a steep logistic, with σ = 0.5 or σ = 0.2 + 0.6u.
# The appended designs are quadratic (E[x₂ | u] = u²), cancellation (g = 0.5 sin(2πu),
# so E[y | u] = u), g-zero (g = 0), and linear-means (E[x₂ | u] = -u). Otherwise
# g = sin(2πu), and the appended designs have σ = 0.5. These are custom designs,
# not a replication of a published study. Compare three CV bandwidth rules and
# residualization with the true conditional means, with HC0 and HC1 variances.
include(joinpath(@__DIR__, "common.jl"))

# Keep the original positions: they determine the seeds.
const DESIGNS = ["sine", "sine-heteroskedastic", "logistic", "quadratic", "cancellation", "g-zero", "linear-means"]
const β = [1.0, -0.5]

function design_data(rng, design, n)
  g = design == "logistic" ? (u -> 2 / (1 + exp(-(u - 0.5) / 0.05))) :
      design == "cancellation" ? (u -> 0.5sin(2π * u)) :
      design == "g-zero" ? (u -> 0.0) : (u -> sin(2π * u))
  σ = design == "sine-heteroskedastic" ? (u -> 0.2 + 0.6u) : (u -> 0.5)
  u = rand(rng, n)
  mean_x2 = design == "quadratic" ? u.^2 : design == "linear-means" ? -u : sin.(2π .* u)
  X = hcat(u .+ randn(rng, n), mean_x2 .+ randn(rng, n))
  y = X * β .+ g.(u) .+ σ.(u) .* randn(rng, n)
  mean_X = hcat(u, mean_x2)
  mean_y = mean_X * β .+ g.(u)
  return u, X, y, mean_X, mean_y
end

# The unweighted sandwich of the residual regression, as in PartiallyLinear:
# G = X̃'X̃, M = X̃' diag(e²) X̃, V₀ = G⁻¹ M G⁻¹, V₁ = n/(n-q) V₀.
function oracle_fit(X, y, mean_X, mean_y)
  X̃, ỹ = X .- mean_X, y .- mean_y
  n, q = size(X̃)
  n > q || throw(ArgumentError("the oracle needs n > q"))
  b = qr(X̃, ColumnNorm()) \ ỹ
  e = ỹ .- X̃ * b
  G = X̃' * X̃
  meat = X̃' * (e.^2 .* X̃)
  V = G \ (G \ meat)'
  V0 = (V + V') / 2
  return (estimate = b, vcov0 = V0, vcov1 = n / (n - q) .* V0)
end

# writerows uses unquoted fields, so keep exception messages on one CSV line.
error_message(err) = replace(sprint(showerror, err), ',' => ';', '\n' => ' ', '\r' => ' ', '"' => '\'')

function result_rows!(rows, a, rep, method, vce, b, se, h, boundary, error = "")
  for j in eachindex(β)
    message = isempty(error) && !(isfinite(b[j]) && isfinite(se[j]) && se[j] >= 0) ?
              "non-finite estimate or invalid standard error" : error
    push!(rows, (a.design, a.n, rep, method, vce, j, β[j], b[j], se[j], h..., boundary..., message))
  end
end

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(14, code, a.n, rep))
    u, X, y, mean_X, mean_y = design_data(rng, a.design, a.n)
    for mode in (:response, :smallest, :each)
      h, boundary = fill(NaN, 3), fill(NaN, 3)
      try
        m0 = fit(PartiallyLinear, u, X, y; bandwidth = CrossValidation(), bandwidth_mode = mode, vce = HCVariance(0))
        h = copy(m0.bandwidths)
        # Check every returned selection, before a common bandwidth is imposed.
        boundary = m0.selections === nothing ? fill(NaN, 3) : [s.details.atbound for s in m0.selections]
        result_rows!(rows, a, rep, string(mode), "hc0", coef(m0), stderror(m0), h, boundary)
      catch err
        for vce in ("hc0", "hc1")
          result_rows!(rows, a, rep, string(mode), vce, fill(NaN, 2), fill(NaN, 2), h, boundary, error_message(err))
        end
        continue
      end
      try
        m1 = fit(PartiallyLinear, u, X, y; bandwidth = h, bandwidth_mode = mode, vce = HCVariance(1))
        result_rows!(rows, a, rep, string(mode), "hc1", coef(m1), stderror(m1), h, boundary)
      catch err
        result_rows!(rows, a, rep, string(mode), "hc1", fill(NaN, 2), fill(NaN, 2), h, boundary, error_message(err))
      end
    end
    try
      oracle = oracle_fit(X, y, mean_X, mean_y)
      for (vce, V) in (("hc0", oracle.vcov0), ("hc1", oracle.vcov1))
        result_rows!(rows, a, rep, "oracle", vce, oracle.estimate, sqrt.(max.(diag(V), 0.0)), fill(NaN, 3), fill(NaN, 3))
      end
    catch err
      for vce in ("hc0", "hc1")
        result_rows!(rows, a, rep, "oracle", vce, fill(NaN, 2), fill(NaN, 2), fill(NaN, 3), fill(NaN, 3), error_message(err))
      end
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "vce", "coefficient", "truth", "estimate", "se",
                   "h_y", "h_1", "h_2", "cv_boundary_y", "cv_boundary_1", "cv_boundary_2", "error"), rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
