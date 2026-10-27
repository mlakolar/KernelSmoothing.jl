# Calibration of LSCVMatrix tie warnings and collapse diagnostics (#102).
#
#     julia --project=benchmark benchmark/lscv_ties.jl manifest
#     julia --project=benchmark benchmark/lscv_ties.jl run FIRST LAST out.csv
#
# Cases are numbered from 1. Exact cases iterate over d, n, form, seed, design
# in the orders below; binned cases follow, over d, n, seed, design. All runs
# require one Julia thread and use one BLAS thread. Times include compilation
# on the first selection and exclude sample generation and independent checks.
# Matrices use the lower triangle by columns (11, 21, 31, 22, 32, 33), padded
# with NaN in 2-D. S and A are computed independently, H0 is the starting matrix,
# and Hy = A * H * A'. Failures remain as rows with all available diagnostics.
using KernelSmoothing
using LinearAlgebra
using Logging
using Random
using Statistics

const DESIGNS = ("normal", "corr09", "bimodal6", "bimodal10", "lattice", "lognormal",
                 "round1", "round2", "round3", "dup5", "dup10", "dup25")
const VECH = ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3))
const HEADER = vcat(
  ["case_id", "design", "d", "n", "form", "binned", "seed", "status", "error", "seconds",
   "converged", "collapsed", "eigenratio_min", "eigenratio_max", "duplicates"],
  ["ties$k" for k in 1:3], ["H$a$b" for (a, b) in VECH],
  ["warn_tie", "warn_collapse", "warn_coarse", "warn_other", "dup_indep"],
  ["ties_indep$k" for k in 1:3],
  ["$prefix$a$b" for prefix in ("S", "A", "H0", "Hy") for (a, b) in VECH])

csv_cell(x) = x isa AbstractString ? "\"" * replace(x, '"' => "\"\"", '\n' => ' ', '\r' => ' ') * "\"" : string(x)
csv_row(io, row) = println(io, join(csv_cell.(row), ","))

function manifest()
  cases = NamedTuple[]
  for d in (2, 3), n in (100, 400, 2000), form in ("full", "diagonal"), seed in 1:30, design in DESIGNS
    push!(cases, (; case_id = length(cases) + 1, design, d, n, form, binned = false, seed))
  end
  for d in (2, 3), n in (10_000, 100_000), seed in 1:5, design in ("normal", "round2")
    push!(cases, (; case_id = length(cases) + 1, design, d, n, form = "full", binned = true, seed))
  end
  return cases
end

# Family codes 1:6 follow DESIGNS; rounded and duplicated samples use family 1.
# Decimal fields do not overlap on the manifest: family, d, n, and replicate.
function sample(c)
  family = findfirst(==(c.design), DESIGNS)
  family === nothing && throw(ArgumentError("Unknown design $(c.design); use a manifest design."))
  family = family > 6 ? 1 : family
  rng = Xoshiro(102_000_000_000 + 10_000_000_000family + 1_000_000_000c.d + 100c.n + c.seed)
  X = randn(rng, c.d, c.n)
  if c.design == "corr09"
    S = fill(0.9, c.d, c.d)
    S[diagind(S)] .= 1.0
    X = cholesky(Symmetric(S)).L * X
  elseif c.design in ("bimodal6", "bimodal10")
    separation = c.design == "bimodal6" ? 6.0 : 10.0
    X[1, :] .+= separation .* rand(rng, Bool, c.n)
  elseif c.design == "lattice"
    X .*= 0.05
    width = c.d == 2 ? 5 : 3
    for i in 1:c.n
      node = rand(rng, 0:width^c.d-1)
      for k in 1:c.d
        X[k, i] += node % width
        node ÷= width
      end
    end
  elseif c.design == "lognormal"
    X = exp.(X)
  elseif startswith(c.design, "round")
    X = round.(X; digits = parse(Int, c.design[6:end]))
  elseif startswith(c.design, "dup")
    fraction = parse(Int, c.design[4:end]) / 100
    k = round(Int, fraction * c.n)
    X[:, end-k+1:end] .= X[:, 1:k]
  end
  return X
end

"""Count repeated columns and coordinate fractions `1 - distinct / n`, treating signed zeros equally."""
function independent_ties(X)
  d, n = size(X)
  value(x) = iszero(x) ? 0.0 : x
  columns = sort!([ntuple(k -> value(X[k, i]), d) for i in 1:n])
  duplicates = count(i -> columns[i] == columns[i - 1], 2:n)
  ties = [1 - length(unique(value.(X[k, :]))) / n for k in 1:d]
  return duplicates, ties
end

"""Compute `S = cov(X; dims = 2)`, `A`, and `H0 = (4 / ((d + 2)n))^(2 / (d + 4)) * Σy`."""
function independent_start(X, diagonal)
  d, n = size(X)
  S = cov(X; dims = 2, corrected = true)
  A = if diagonal
    Matrix(Diagonal(1 ./ sqrt.(diag(S))))
  else
    E = eigen(Symmetric(S))
    E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors'
  end
  Σy = A * S * A'
  H0 = (4 / ((d + 2) * n))^(2 / (d + 4)) .* (diagonal ? Matrix(Diagonal(diag(Σy))) : Σy)
  return S, A, H0
end

mutable struct WarningLogger <: AbstractLogger
  tie::Bool
  collapse::Bool
  coarse::Bool
  other::Bool
end
WarningLogger() = WarningLogger(false, false, false, false)
Logging.min_enabled_level(::WarningLogger) = Logging.Warn
Logging.shouldlog(::WarningLogger, level, _module, group, id) = level >= Logging.Warn
Logging.catch_exceptions(::WarningLogger) = false

function Logging.handle_message(logger::WarningLogger, level, message, _module, group, id, file, line; kwargs...)
  level == Logging.Warn || return
  text = string(message)
  tie = occursin("tied or duplicated", text)
  collapse = occursin("degenerate bandwidth matrix", text)
  coarse = occursin("coarse", text)
  logger.tie |= tie
  logger.collapse |= collapse
  logger.coarse |= coarse
  logger.other |= !(tie || collapse || coarse)
  return nothing
end

function matrix_cells!(row, prefix, H)
  d = size(H, 1)
  for (a, b) in VECH
    row["$prefix$a$b"] = a <= d && b <= d ? H[a, b] : NaN
  end
end

function tie_cells!(row, prefix, ties)
  for k in 1:3
    row["$prefix$k"] = k <= length(ties) ? ties[k] : NaN
  end
end

function study_row(c)
  row = Dict{String,Any}(name => NaN for name in HEADER)
  for key in keys(c)
    row[string(key)] = getproperty(c, key)
  end
  row["status"], row["error"] = "ok", ""
  logger = WarningLogger()
  try
    X = sample(c)
    duplicates, ties = independent_ties(X)
    row["dup_indep"] = duplicates
    tie_cells!(row, "ties_indep", ties)
    S, A, H0 = independent_start(X, c.form == "diagonal")
    matrix_cells!(row, "S", S)
    matrix_cells!(row, "A", A)
    matrix_cells!(row, "H0", H0)
    rule = LSCVMatrix(; diagonal = c.form == "diagonal", binned = c.binned)
    selection = with_logger(logger) do
      start = time_ns()
      try
        select_bandwidth(rule, X)
      finally
        row["seconds"] = (time_ns() - start) / 1e9
      end
    end
    H = bandwidth(selection)
    matrix_cells!(row, "H", H)
    matrix_cells!(row, "Hy", A * H * A')
    details = selection.details
    missing_fields = String[]
    for key in (:converged, :collapsed, :duplicates, :ties, :eigenratio)
      if !hasproperty(details, key)
        push!(missing_fields, string(key))
      elseif key == :ties
        tie_cells!(row, "ties", details.ties)
      elseif key == :eigenratio
        row["eigenratio_min"], row["eigenratio_max"] = details.eigenratio
      else
        row[string(key)] = getproperty(details, key)
      end
    end
    isempty(missing_fields) || error("Selection details are missing fields: " * join(missing_fields, ", ") * ".")
  catch err
    err isa InterruptException && rethrow()
    row["status"], row["error"] = "error", sprint(showerror, err)
  end
  row["warn_tie"], row["warn_collapse"] = logger.tie, logger.collapse
  row["warn_coarse"], row["warn_other"] = logger.coarse, logger.other
  return [row[name] for name in HEADER]
end

function run_cases(first, last, out)
  cases = manifest()
  1 <= first <= last <= length(cases) || throw(ArgumentError(
    "Choose an inclusive case range within 1:$(length(cases))."))
  Threads.nthreads() == 1 || throw(ArgumentError("Run this study with one Julia thread (julia -t 1)."))
  BLAS.set_num_threads(1)
  open(out, "w") do io
    csv_row(io, HEADER)
    flush(io)
    for c in cases[first:last]
      csv_row(io, study_row(c))
      flush(io)
    end
  end
end

function main(args = ARGS)
  if args == ["manifest"]
    println(length(manifest()))
  elseif length(args) == 4 && args[1] == "run"
    first, last = tryparse(Int, args[2]), tryparse(Int, args[3])
    (first === nothing || last === nothing) && throw(ArgumentError("FIRST and LAST must be integer case ids."))
    run_cases(first, last, args[4])
  else
    throw(ArgumentError("Usage: lscv_ties.jl manifest | lscv_ties.jl run FIRST LAST out.csv."))
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
