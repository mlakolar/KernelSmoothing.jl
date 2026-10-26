# The error of density estimates on [0, 1] (#61): the kernel density estimate that
# ignores the boundary, reflection, the beta kernels of Chen (1999) and their modified
# version, the logit transformation, and LocalPolynomialDensity, each with its default
# bandwidth, for densities that are positive with a slope (Beta(1, 3)), zero
# (Beta(2, 2)) or flat (uniform) at the boundary.
include(joinpath(@__DIR__, "common.jl"))

const DESIGNS = ["beta13", "beta22", "uniform"]
const DISTRIBUTIONS = Dict("beta13" => Beta(1, 3), "beta22" => Beta(2, 2), "uniform" => Beta(1, 1))
const GRID = collect(0.005:0.01:0.995)       # midpoints: the integral is their mean
const AT = [0.001, 0.05, 0.5, 0.999]          # the errors reported at single points
const EVAL = vcat(GRID, AT)

# the estimates at EVAL
function estimates(method, x)
  if method == "kde"
    f = fit(KDE, x)
    return pdf.(Ref(f), EVAL)
  elseif method == "lpdensity"
    return coef(fit(LocalPolynomialDensity, x; points = EVAL))
  end
  bm = method == "reflection" ? Reflection() : method == "beta" ? BetaKernels() :
       method == "beta-modified" ? BetaKernels(modified = true) : Transformed(transform = :logit)
  f = fit(BoundedKDE, x; support = (0, 1), method = bm)
  return pdf.(Ref(f), EVAL)
end

const METHODS = ["kde", "reflection", "beta", "beta-modified", "logit", "lpdensity"]

function main()
  a = studyargs()
  code = designcode(DESIGNS, a.design)
  D = DISTRIBUTIONS[a.design]
  truth = pdf.(D, EVAL)
  G = length(GRID)
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    rng = Xoshiro(seed(4, code, a.n, rep))
    x = rand(rng, D, a.n)
    for method in METHODS
      t = @elapsed est = try
        estimates(method, x)
      catch err
        err isa ArgumentError || rethrow()
        fill(NaN, length(EVAL))
      end
      errs = est[G+1:end] .- truth[G+1:end]
      push!(rows, (a.design, a.n, rep, method, mean(abs2, est[1:G] .- truth[1:G]), errs..., t))
    end
  end
  writerows(a.out, ("design", "n", "rep", "method", "ise", "error_0.001", "error_0.05", "error_0.5", "error_0.999",
                    "seconds"), rows)
end

main()
