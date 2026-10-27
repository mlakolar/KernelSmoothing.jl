# Binning error of PI, SCV and LSCV bandwidth matrices, full and diagonal (#86).
#
#     julia --project=. mvbinning.jl DESIGN N FIRSTREP NREPS OUT
#
# References are exact through n = 10⁴ and binned at 301 nodes (2-D) or 61 (3-D)
# at n = 10⁵. ISE uses Gaussian overlaps, including the self-pairs, through n = 10⁴.
# The optional ise_selfcheck() compares this identity with a fine integration grid
# at n = 5; it is not run by main(). Binning is forced even at n = 1000 in 3-D,
# where automatic selection would use exact sums under the initial thresholds.
include(joinpath(@__DIR__, "common.jl"))

# Study code 15 follows partiallylinear (14). Keep these positions fixed. Sweep
# designs reuse their base target's seed, so all grids see the same observations.
const BASE_DESIGNS = ["correlated", "bimodal", "skewed", "trivariate"]
const SWEEP_DESIGNS = ["bimodal-g51", "bimodal-g101", "bimodal-g151", "bimodal-g301",
                       "trivariate-g21", "trivariate-g31", "trivariate-g51"]
const DESIGNS = vcat(BASE_DESIGNS, SWEEP_DESIGNS)
mv(μ, Σ) = MvNormal(μ, Σ)
const TARGETS = Dict(
  "correlated" => MixtureModel([mv([0.0, 0.0], [1.0 0.7; 0.7 1.0])], [1.0]),
  "bimodal" => MixtureModel([mv([-1.0, 0.0], [0.25 0.0; 0.0 0.25]), mv([1.0, 0.0], [0.25 0.0; 0.0 0.25])], [0.5, 0.5]),
  "skewed" => MixtureModel([mv([0.0, 0.0], [1.0 0.0; 0.0 1.0]), mv([0.5, 0.5], [0.16 0.0; 0.0 0.16]),
                            mv([1.0, 1.0], [0.04 0.0; 0.0 0.04])], [0.2, 0.2, 0.6]),
  "trivariate" => MixtureModel([mv([-1.0, 0.0, 0.0], [0.25 0.0 0.0; 0.0 0.25 0.0; 0.0 0.0 0.25]),
                               mv([1.0, 0.0, 0.0], [1.0 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 1.0])], [0.5, 0.5]))
const RULES = [("PI", PlugInMatrix), ("SCV", SCVMatrix), ("LSCV", LSCVMatrix)]
# Failed-row fallback only, mirroring the package's opt-in grid defaults.
const DEFAULT_BIN_GRIDSIZES = Dict(2 => 151, 3 => 51)

function design_settings(design, n)
  designcode(DESIGNS, design)
  parts = split(design, "-g"; limit = 2)
  length(parts) == 2 && n != 10_000 && error("grid sweeps use n = 10000")
  return (target = String(parts[1]), gridsize = length(parts) == 2 ? parse(Int, parts[2]) : nothing)
end

# Unordered pairs of whitened observations. The striped split balances the
# triangle; each part and their final sum have an order independent of the threads.
function ise_pairs(Y, t, ntasks)
  d, n = size(Y)
  s = 0.0
  @inbounds for i in t:ntasks:n-1
    for j in i+1:n
      q = 0.0
      for a in 1:d
        q += (Y[a, i] - Y[a, j])^2
      end
      s += exp(-q / 2)
    end
  end
  return s
end

# ∫ f̂² = (n φ_{2H}(0) + 2 ∑ᵢ<ⱼ φ_{2H}(xᵢ - xⱼ)) / n².
function kde_overlap(X, H)
  d, n = size(X)
  F = cholesky(Symmetric(2 .* H))
  Y = F.L \ X
  φ0 = (2π)^(-d / 2) / prod(F.L[a, a] for a in 1:d)
  ntasks = min(n, 256)
  parts = zeros(ntasks)
  Threads.@threads for t in 1:ntasks
    parts[t] = ise_pairs(Y, t, ntasks)
  end
  return φ0 * (n + 2sum(parts)) / n^2
end

# ∫ f̂f and ∫ f² use φ_{H+Σₖ}(xᵢ - μₖ) and
# φ_{Σₖ+Σₗ}(μₖ - μₗ), respectively, for the Gaussian mixture target.
function exact_ise(X, H, D)
  n = size(X, 2)
  components = Distributions.components(D)
  weights = Distributions.probs(D)
  cross, truth = 0.0, 0.0
  for (k, C) in enumerate(components)
    overlap = MvNormal(mean(C), H + cov(C))
    cross += weights[k] * sum(pdf(overlap, view(X, :, i)) for i in 1:n) / n
    for (l, B) in enumerate(components)
      truth += weights[k] * weights[l] * pdf(MvNormal(mean(B), cov(C) + cov(B)), mean(C))
    end
  end
  return kde_overlap(X, H) - 2cross + truth
end

function ise_selfcheck()
  D = TARGETS["skewed"]
  X = rand(Xoshiro(86_015), D, 5)
  H = [0.3 0.08; 0.08 0.2]
  kernels = [MvNormal(X[:, i], H) for i in 1:5]
  G = collect(-10.0:0.05:10.0)
  point = zeros(2)
  integral = 0.0
  for i in eachindex(G), j in eachindex(G)
    point[1], point[2] = G[i], G[j]
    estimate = sum(pdf(C, point) for C in kernels) / 5
    weight = (i == 1 || i == length(G) ? 0.5 : 1.0) * (j == 1 || j == length(G) ? 0.5 : 1.0)
    integral += weight * (estimate - pdf(D, point))^2
  end
  integral *= 0.05^2
  identity = exact_ise(X, H, D)
  @assert isapprox(identity, integral; atol = 2e-7, rtol = 2e-6)
  return (identity = identity, quadrature = integral, absolute_error = abs(identity - integral))
end

# Exceptions and invalid matrices stay in the output as failed attempts; a failure
# on one side does not prevent the other selection or the remaining rules.
function selection(constructor, X; diagonal, binned, gridsize = nothing)
  s = nothing
  seconds = @elapsed try
    s = select_bandwidth(constructor(; diagonal, binned, gridsize), X)
  catch err
    @warn "selection failed" constructor diagonal binned gridsize exception = (err, catch_backtrace())
  end
  s === nothing && return (H = nothing, seconds = seconds, iterations = NaN, converged = false,
                            finite = false, gridsize = gridsize)
  details = s.details
  valid = all(isfinite, s.h) && isfinite(details.criterion) && isposdef(Symmetric(s.h))
  return (H = s.h, seconds = seconds, iterations = details.iterations, converged = details.converged,
          finite = valid, gridsize = details.gridsize)
end

function ise_or_nan(X, result, D)
  result.finite || return NaN
  return try
    exact_ise(X, result.H, D)
  catch err
    @warn "ISE failed" exception = (err, catch_backtrace())
    NaN
  end
end

function main()
  a = studyargs()
  a.n in (1_000, 10_000, 100_000) || error("n must be 1000, 10000 or 100000")
  settings = design_settings(a.design, a.n)
  code = designcode(BASE_DESIGNS, settings.target)
  D = TARGETS[settings.target]
  d = length(D)
  reference = a.n <= 10_000 ? "exact" : "fine-grid"
  refgrid = a.n <= 10_000 ? nothing : d == 2 ? 301 : 61
  rows = Any[]
  for rep in a.first:a.first+a.nreps-1
    X = rand(Xoshiro(seed(15, code, a.n, rep)), D, a.n)
    for (name, constructor) in RULES, diagonal in (false, true)
      bin = selection(constructor, X; diagonal, binned = true, gridsize = settings.gridsize)
      ref = selection(constructor, X; diagonal, binned = a.n > 10_000, gridsize = refgrid)
      error = if bin.finite && ref.finite
        try
          maximum(abs.(eigvals(Symmetric(bin.H), Symmetric(ref.H)) .- 1))
        catch err
          @warn "generalized eigenvalues failed" name diagonal rep exception = (err, catch_backtrace())
          NaN
        end
      else
        NaN
      end
      ise_bin = a.n <= 10_000 ? ise_or_nan(X, bin, D) : NaN
      ise_ref = a.n <= 10_000 ? ise_or_nan(X, ref, D) : NaN
      ratio = isfinite(ise_bin) && isfinite(ise_ref) && ise_ref > 0 ? ise_bin / ise_ref : NaN
      # If default selection throws before returning details, retain its planned
      # grid in the failed row. Successful rows always use the reported grid.
      grid = something(bin.gridsize, settings.gridsize, DEFAULT_BIN_GRIDSIZES[d])
      push!(rows, (a.design, d, a.n, rep, name, diagonal, grid, reference, something(ref.gridsize, NaN),
                   bin.seconds, ref.seconds, bin.iterations, ref.iterations, bin.converged, ref.converged,
                   bin.finite, ref.finite, error, ise_bin, ise_ref, ratio))
    end
  end
  writerows(a.out, ("design", "d", "n", "rep", "rule", "diagonal", "gridsize", "reference", "reference_gridsize",
                   "seconds_bin", "seconds_ref", "iterations_bin", "iterations_ref", "converged_bin", "converged_ref",
                   "finite_bin", "finite_ref", "max_eigen_error", "ise_bin", "ise_ref", "ise_ratio"), rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
