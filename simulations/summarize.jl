# Tables of the results of the simulation campaign, as Markdown: run from the run
# directory, which holds results-STUDY.csv, as
#
#     julia --project=src/simulations src/simulations/summarize.jl [STUDY ...]
#
# Coverage is reported with its Monte Carlo standard error.
using Statistics
using Distributions: Normal, quantile

const Z = quantile(Normal(), 0.975)

# a CSV file of numbers and strings as a vector of named rows
function readcsv(path)
  lines = readlines(path)
  header = Symbol.(split(lines[1], ","))
  rows = map(lines[2:end]) do l
    vals = map(split(l, ",")) do v
      t = tryparse(Float64, v)
      t === nothing ? (v == "true" ? 1.0 : v == "false" ? 0.0 : String(v)) : t
    end
    NamedTuple{Tuple(header)}(Tuple(vals))
  end
  return rows
end

groupby(rows, keys) = (g = Dict{Any,Vector{eltype(rows)}}(); for r in rows; push!(get!(g, Tuple(r[k] for k in keys), eltype(rows)[]), r); end; g)
fmt(x; d = 3) = isnan(x) ? "NaN" : string(round(x; digits = d))
pct(p, R) = string(fmt(100p; d = 1), " (", fmt(100sqrt(p * (1 - p) / R); d = 1), ")")
finite(v) = filter(isfinite, v)

cell(v) = v isa AbstractFloat && isfinite(v) && isinteger(v) ? string(Int(v)) : string(v)

function table(header, rows)
  println("| ", join(header, " | "), " |")
  println("|", join(fill("---", length(header)), "|"), "|")
  for r in rows
    println("| ", join(cell.(r), " | "), " |")
  end
  println()
end

function locallik(rows)
  println("### Local likelihood: coverage of 95% intervals for the linear predictor (Monte Carlo standard error)\n")
  out = []
  for k in sort(collect(keys(groupby(rows, (:design, :n, :bandwidth, :point)))))
    g = groupby(rows, (:design, :n, :bandwidth, :point))[k]
    ok = filter(r -> isfinite(r.eta) && isfinite(r.se), g)
    okb = filter(r -> isfinite(r.eta_bc) && isfinite(r.se_rbc), g)
    cc = mean(abs(r.eta - r.truth) <= Z * r.se for r in ok)
    cr = mean(abs(r.eta_bc - r.truth) <= Z * r.se_rbc for r in okb)
    push!(out, (k..., pct(cc, length(ok)), pct(cr, length(okb)), fmt(mean(2Z * r.se for r in ok)),
                fmt(mean(2Z * r.se_rbc for r in okb)), fmt(median(r.h for r in g)), length(g) - length(okb)))
  end
  table(("design", "n", "bandwidth", "point", "conventional", "robust", "length", "robust length", "median h", "NaN"), out)
end

function quantreg(rows)
  println("### Local quantile regression: 95% pointwise intervals\n")
  println("Coverage, calibration and length comparisons use the common valid set.\n")
  println("Coverage is in percent; differences are in percentage points (Monte Carlo SE).\n")
  println("Bandwidth medians use all attempted replications.\n")
  println("Length ratios are RBC / conventional. Undefined ratios are NaN.\n")
  println("SDs and the paired Monte Carlo SE need at least two valid replications.\n")
  coverage, calibration = [], []
  valid(r) = isfinite(r.estimate) && isfinite(r.se)
  valid_bc(r) = isfinite(r.estimate_bc) && isfinite(r.se_rbc)
  avg(v) = isempty(v) ? NaN : mean(v)
  sd(v) = length(v) < 2 ? NaN : std(v)
  G = groupby(rows, (:design, :n, :point))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> valid(r) && valid_bc(r), g)
    R = length(ok)
    cc = [abs(r.estimate - r.truth) <= Z * r.se for r in ok]
    cr = [abs(r.estimate_bc - r.truth) <= Z * r.se_rbc for r in ok]
    delta = Int.(cr) .- Int.(cc)
    paired = string(fmt(100avg(delta); d = 1), " (", fmt(100sd(delta) / sqrt(R); d = 1), ")")
    se = [r.se for r in ok]
    se_rbc = [r.se_rbc for r in ok]
    sd_est = sd([r.estimate for r in ok])
    sd_bc = sd([r.estimate_bc for r in ok])
    bias = avg([r.estimate - r.truth for r in ok])
    bias_bc = avg([r.estimate_bc - r.truth for r in ok])
    bias_sd = sd_est == 0 ? NaN : bias / sd_est
    bias_sd_bc = sd_bc == 0 ? NaN : bias_bc / sd_bc
    ratio = R == 0 || avg(se) == 0 ? NaN : avg(se_rbc) / avg(se)
    ratios = R == 0 || any(iszero, se) ? NaN : avg(se_rbc ./ se)
    design, τs = rsplit(k[1], "-"; limit = 2)
    key = (design, k[2], parse(Float64, τs), k[3])
    push!(coverage, (key..., length(g), count(r -> !valid(r), g), count(r -> !valid_bc(r), g), R,
                     pct(avg(cc), R), pct(avg(cr), R), paired,
                     fmt(median(r.h for r in g)), fmt(median(r.b for r in g))))
    push!(calibration, (key..., fmt(bias), fmt(bias_sd), fmt(bias_bc), fmt(bias_sd_bc),
                        fmt(sd_est), fmt(avg(se)), fmt(sqrt(avg(se.^2))),
                        fmt(sd_bc), fmt(avg(se_rbc)), fmt(sqrt(avg(se_rbc.^2))),
                        fmt(ratio), fmt(ratios)))
  end
  println("Coverage\n")
  table(("design", "n", "τ", "point", "attempted", "failed conventional", "failed RBC", "common valid",
         "conventional coverage (MC SE)", "RBC coverage (MC SE)", "RBC − conventional (MC SE)",
         "median h", "median b"), coverage)
  println("Calibration and length\n")
  table(("design", "n", "τ", "point", "conventional bias", "conventional bias/SD", "RBC bias", "RBC bias/SD",
         "conventional SD", "conventional mean se", "conventional RMS se", "RBC SD", "RBC mean se", "RBC RMS se",
         "ratio of mean lengths", "mean paired length ratio"), calibration)
end

function shape(rows)
  println("### Shape constraints: mean squared error at the points, and coverage (Monte Carlo standard error)\n")
  out = []
  G = groupby(rows, (:design, :n))
  for k in sort(collect(keys(G)))
    g = G[k]
    R = length(g)
    m(f) = mean(getfield(r, f) for r in g)
    push!(out, (k..., fmt(1000m(:ise_fit)), fmt(1000m(:ise_rearranged)), fmt(1000m(:ise_constrained)),
                pct(m(:pointwise_fit), R), pct(m(:pointwise_rearranged), R), pct(m(:band_fit), R),
                pct(m(:band_rearranged), R), fmt(m(:width_rearranged) / m(:width_fit)), pct(m(:fit_monotone), R)))
  end
  table(("design", "n", "MSE × 1000 fit", "rearranged", "constrained", "pointwise fit", "pointwise rearranged",
         "band fit", "band rearranged", "width ratio", "fit monotone"), out)
end

function bounded(rows)
  println("### Densities on [0, 1]: integrated squared error and errors at single points (root mean square)\n")
  out = []
  G = groupby(rows, (:design, :n, :method))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> isfinite(r.ise), g)
    rms(f) = fmt(sqrt(mean(getfield(r, f)^2 for r in ok)))
    push!(out, (k..., fmt(1000mean(r.ise for r in ok)), rms(Symbol("error_0.001")), rms(Symbol("error_0.05")),
                rms(Symbol("error_0.5")), rms(Symbol("error_0.999")), fmt(mean(r.seconds for r in g); d = 4),
                length(g) - length(ok)))
  end
  table(("design", "n", "method", "ISE × 1000", "at 0.001", "at 0.05", "at 0.5", "at 0.999", "seconds", "failed"), out)
end

function rdlocrand(rows)
  println("### Local randomization: rejection rates at the 5% level (Monte Carlo standard error)\n")
  out = []
  G = groupby(rows, (:design, :n, :statistic))
  for k in sort(collect(keys(G)))
    g = G[k]
    R = length(g)
    push!(out, (k..., pct(mean(r.pvalue <= 0.05 for r in g), R), pct(mean(r.asymptotic <= 0.05 for r in g), R),
                fmt(mean(r.ncontrol for r in g); d = 1), fmt(mean(r.ntreated for r in g); d = 1)))
  end
  table(("design", "n", "statistic", "randomization", "asymptotic", "controls", "treated"), out)
end

# the studies with rows (design, n, rep, method, point, truth, estimate, se, h), and
# the bands, whose rows carry the coverage, the mean width and the critical value in
# the columns truth, estimate and se
function intervals(title; bysize = true)
  return function (rows)
    # the sample size of the unbalanced cluster designs varies between replications
    bysize || (rows = [merge(r, (n = 0.0,)) for r in rows])
    println("### ", title, ": coverage of 95% intervals (Monte Carlo standard error)\n")
    out = []
    pw = filter(r -> !startswith(r.method, "band"), rows)
    G = groupby(pw, (:design, :n, :method, :point))
    for k in sort(collect(keys(G)))
      g = G[k]
      ok = filter(r -> isfinite(r.estimate) && isfinite(r.se), g)
      c = mean(abs(r.estimate - r.truth) <= Z * r.se for r in ok)
      push!(out, (k..., pct(c, length(ok)), fmt(mean(2Z * r.se for r in ok)), fmt(median(r.h for r in g)),
                  length(g) - length(ok)))
    end
    table(("design", "n", "method", "point", "coverage", "length", "median h", "NaN"), out)
    bands = filter(r -> startswith(r.method, "band"), rows)
    isempty(bands) && return
    println("Uniform bands on a grid of 21 points\n")
    out = []
    G = groupby(bands, (:design, :n, :method))
    for k in sort(collect(keys(G)))
      g = G[k]
      push!(out, (k..., pct(mean(r.truth for r in g), length(g)), fmt(mean(r.estimate for r in g)),
                  fmt(mean(r.se for r in g))))
    end
    table(("design", "n", "method", "coverage", "mean width", "critical value"), out)
  end
end

function honest(rows)
  println("### Honest intervals: coverage of 95% intervals (Monte Carlo standard error)\n")
  out = []
  G = groupby(rows, (:design, :n, :method, :point))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> isfinite(r.lower) && isfinite(r.upper), g)
    push!(out, (k..., pct(mean(r.lower <= r.truth <= r.upper for r in ok), length(ok)),
                fmt(mean(r.upper - r.lower for r in ok)), fmt(median(r.h for r in g)), length(g) - length(ok)))
  end
  table(("design", "n", "method", "point", "coverage", "length", "median h", "NaN"), out)
end

function rd(rows)
  println("### Regression discontinuity: coverage of 95% intervals for the jump (Monte Carlo standard error)\n")
  out = []
  G = groupby(rows, (:design, :n, :method))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> isfinite(r.estimate) && isfinite(r.se), g)
    push!(out, (k..., pct(mean(abs(r.estimate - r.truth) <= Z * r.se for r in ok), length(ok)),
                fmt(mean(r.estimate - r.truth for r in ok); d = 4), fmt(mean(2Z * r.se for r in ok)),
                fmt(median(r.hleft for r in g)), fmt(median(r.hright for r in g)), length(g) - length(ok)))
  end
  table(("design", "n", "method", "coverage", "bias", "length", "median h left", "median h right", "NaN"), out)
end

function rddensity(rows)
  println("### Manipulation test: rejection rates at the 5% level (Monte Carlo standard error)\n")
  out = []
  G = groupby(rows, (:design, :n))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> isfinite(r.pvalue), g)
    push!(out, (k..., pct(mean(r.pvalue <= 0.05 for r in ok), length(ok)), fmt(median(r.hleft for r in g)),
                fmt(median(r.hright for r in g)), length(g) - length(ok)))
  end
  table(("design", "n", "rejection", "median h left", "median h right", "NaN"), out)
end

function mvkde(rows)
  println("### Bivariate densities: integrated squared error of the bandwidth matrices\n")
  out = []
  G = groupby(rows, (:design, :n, :rule))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(r -> isfinite(r.ise), g)
    push!(out, (k..., fmt(1000mean(r.ise for r in ok)), fmt(1000median(r.ise for r in ok)),
                fmt(mean(r.seconds for r in g); d = 4), length(g) - length(ok)))
  end
  table(("design", "n", "rule", "mean ISE × 1000", "median ISE × 1000", "seconds", "failed"), out)
end

function mvbinning(rows)
  println("### Binning error of bandwidth matrices\n")
  println("Metrics use finite, converged pairs; time medians include all finite attempt times.\n")
  println("ISE is omitted at n = 100000, whose reference is a finer binned grid.\n")
  stats(v) = isempty(v) ? (NaN, NaN, NaN) : (median(v), quantile(v, 0.9), maximum(v))
  med(v) = isempty(v) ? NaN : median(v)
  valid(r) = r.finite_bin == 1 && r.finite_ref == 1 && r.converged_bin == 1 && r.converged_ref == 1 &&
             all(isfinite, (r.seconds_bin, r.seconds_ref, r.iterations_bin, r.iterations_ref, r.max_eigen_error)) &&
             (r.n > 10_000 || (all(isfinite, (r.ise_bin, r.ise_ref, r.ise_ratio)) && r.ise_bin >= 0 && r.ise_ref > 0))
  matrices, errors = [], []
  G = groupby(rows, (:design, :d, :n, :gridsize, :rule, :diagonal, :reference))
  for k in sort(collect(keys(G)))
    g = G[k]
    ok = filter(valid, g)
    key = (k[1:5]..., k[6] == 1, k[7])
    eigen = stats([r.max_eigen_error for r in ok])
    push!(matrices, (key..., length(g), length(g) - length(ok), length(ok), fmt.(eigen; d = 6)...,
                     fmt(med(finite([r.seconds_bin for r in g])); d = 4),
                     fmt(med(finite([r.seconds_ref for r in g])); d = 4)))
    ise = filter(r -> r.n <= 10_000, ok)
    ratios = [r.ise_ratio for r in ise]
    push!(errors, (key..., length(ise), fmt.(stats(ratios); d = 6)...,
                   fmt.(stats(abs.(ratios .- 1)); d = 6)...))
  end
  println("Matrix errors and selection times\n")
  table(("design", "d", "n", "grid", "rule", "diagonal", "reference", "attempted", "failed", "valid pairs",
         "eigen median", "eigen p90", "eigen max", "bin seconds", "ref seconds"), matrices)
  println("Paired ISE ratios\n")
  table(("design", "d", "n", "grid", "rule", "diagonal", "reference", "ISE pairs",
         "ratio median", "ratio p90", "ratio max", "abs(ratio − 1) median", "abs(ratio − 1) p90",
         "abs(ratio − 1) max"), errors)
end

function partiallylinear(rows)
  println("### Partially linear model: 95% coefficient intervals\n")
  println("Coverage, calibration and lengths use the common valid replications across response, smallest, each and oracle, per design, n, VCE and coefficient.\n")
  println("Coverage is in percent; paired differences against response are in percentage points (Monte Carlo SE).\n")
  println("Failures include error messages, non-finite values and negative standard errors. Missing methods have zero attempts and no common valid replications.\n")
  println("Bandwidth medians use all finite attempted values. Boundary percentages use attempts with known CV flags for all three smooths; oracle flags and bandwidths are NaN.\n")
  println("SDs and paired Monte Carlo SEs need at least two common valid replications.\n")
  methods = ("response", "smallest", "each", "oracle")
  valid(r) = r.error == "" && all(isfinite, (r.truth, r.estimate, r.se)) && r.se >= 0
  covered(r) = abs(r.estimate - r.truth) <= Z * r.se
  avg(v) = isempty(v) ? NaN : mean(v)
  sd(v) = length(v) < 2 ? NaN : std(v)
  med(v) = isempty(v) ? NaN : median(v)
  coverage, calibration = [], []
  G = groupby(rows, (:design, :n, :vce, :coefficient))
  for k in sort(collect(keys(G)))
    groups = [filter(r -> r.method == method, G[k]) for method in methods]
    indexed = [Dict(r.rep => r for r in g) for g in groups]
    all(length(g) == length(index) for (g, index) in zip(groups, indexed)) ||
      throw(ArgumentError("duplicate partiallylinear replication in $k"))
    common = sort(collect(intersect((Set(r.rep for r in g if valid(r)) for g in groups)...)))
    R = length(common)
    reference = [covered(indexed[1][rep]) for rep in common]
    for (method, g, index) in zip(methods, groups, indexed)
      ok = [index[rep] for rep in common]
      cc = [covered(r) for r in ok]
      delta = Int.(cc) .- Int.(reference)
      paired = string(fmt(100avg(delta); d = 1), " (", fmt(100sd(delta) / sqrt(R); d = 1), ")")
      key = (k[1], k[2], method, k[3], k[4])
      push!(coverage, (key..., length(g), count(r -> !valid(r), g), R, pct(avg(cc), R), paired))
      se = [r.se for r in ok]
      boundary = filter(r -> all(isfinite, (r.cv_boundary_y, r.cv_boundary_1, r.cv_boundary_2)), g)
      push!(calibration, (key..., fmt(avg([r.estimate - r.truth for r in ok])),
                          fmt(sd([r.estimate for r in ok])), fmt(avg(se)), fmt(sqrt(avg(se.^2))),
                          fmt(2Z * avg(se)), fmt(med(finite([r.h_y for r in g]))),
                          fmt(med(finite([r.h_1 for r in g]))), fmt(med(finite([r.h_2 for r in g]))),
                          length(boundary),
                          (fmt(100avg([r[f] for r in boundary]); d = 1) for f in (:cv_boundary_y, :cv_boundary_1, :cv_boundary_2))...))
    end
  end
  println("Coverage\n")
  table(("design", "n", "method", "VCE", "coefficient", "attempted", "failed", "common valid",
         "coverage (MC SE)", "coverage − response (MC SE)"), coverage)
  println("Calibration, length and bandwidths\n")
  table(("design", "n", "method", "VCE", "coefficient", "bias", "SD", "mean se", "RMS se", "mean length",
         "median h_y", "median h_1", "median h_2", "CV known", "boundary y %", "boundary 1 %", "boundary 2 %"), calibration)
end

const STUDIES = Dict("locallik" => locallik, "quantreg" => quantreg, "shape" => shape, "bounded" => bounded,
                     "rdlocrand" => rdlocrand, "regression" => intervals("Local linear regression"),
                     "density" => intervals("Kernel density estimation"),
                     "cluster" => intervals("Cluster-robust variances"; bysize = false), "honest" => honest, "rd" => rd,
                     "rddensity" => rddensity, "lpdboundary" => intervals("Local polynomial density at a boundary"),
                     "mvkde" => mvkde, "mvbinning" => mvbinning,
                     "partiallylinear" => partiallylinear)

for s in (isempty(ARGS) ? sort(collect(keys(STUDIES))) : ARGS)
  if startswith(s, "partiallylinear=")
    partiallylinear(readcsv(split(s, "="; limit = 2)[2]))
    continue
  end
  f = "results-$s.csv"
  isfile(f) || continue
  STUDIES[s](readcsv(f))
end
