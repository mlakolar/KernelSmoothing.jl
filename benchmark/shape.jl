# Shape projections (#91), with strictly shaped and noisy smooth inputs. Run the
# same script in separate processes for the two revisions, with distinct outputs:
#
#     julia --project=benchmark -t 1 benchmark/shape.jl OUT.csv [RESULTS_DIR]
#     julia --project=benchmark -t 1 benchmark/shape.jl --quick OUT.csv [RESULTS_DIR]
#
# Full sizes are 200, 400 and 800 for strict inputs, and 1000 for smooth inputs;
# --quick uses 20 and 40 for both. The default result directory is OUT-results
# (without .csv). It holds every fitted vector, its input and its weights, so the
# relative vector differences and weighted objectives can be checked afterwards.
# Timings exclude diagnostics and CSV output. If the warm-up takes less than
# 0.1 seconds, take at least 30 timed runs without a budget cutoff, also in --quick
# mode. From 0.1 to under 1 second, take at least 10 runs without a budget cutoff.
# At 1 second or more, take at most five runs within a two-second budget. --quick
# takes one run for warm-ups of 0.1 seconds or more. Record the minimum, median
# and spread. Run GC.gc() once before each case's timed runs, outside the timings.
# For warm-ups of 1 second or more, stop repeating when a
# call takes 60 seconds; a slow warm-up is retained as the single run, including
# compilation. Allocation counts use one separate warmed call, except after a
# slow call, when the warm-up count is retained and allocations_warmed is false.
# Package loading is measured after loading the standard libraries.
# first_call_seconds is per case, including
# compilation still needed at that point in the process.
using Random
using LinearAlgebra
using Logging

const PACKAGE_LOAD_SECONDS = @elapsed @eval using KernelSmoothing
const SHAPES = (:increasing, :decreasing, :convex, :concave, :increasing_convex,
                :increasing_concave, :decreasing_convex, :decreasing_concave)
LinearAlgebra.BLAS.set_num_threads(1)

mutable struct WarningLogger <: AbstractLogger
  messages::Vector{String}
end

Logging.min_enabled_level(::WarningLogger) = Logging.Warn
Logging.shouldlog(::WarningLogger, level, _module, group, id) = level >= Logging.Warn
Logging.catch_exceptions(::WarningLogger) = false
function Logging.handle_message(logger::WarningLogger, level, message, _module, group, id, file, line; kwargs...)
  push!(logger.messages, string(message))
end

csv_cell(x) = x isa AbstractString ? "\"" * replace(x, '"' => "\"\"", '\n' => ' ', '\r' => ' ') * "\"" : string(x)
csv_row(io, row) = println(io, join(csv_cell.(row), ","))

function strict_values(shape, t)
  m = length(t)
  increments = 0.5 .+ rand(Xoshiro(91_000 + m), m - 1)
  steps = vcat(0.0, cumsum(increments))
  curve = vcat(0.0, cumsum(cumsum(increments)))
  # Positive increments give positive step coefficients for the monotone shapes.
  # Double cumulative sums give positive initial slopes and second differences:
  # all right-hinge coefficients are positive. Reversing the curve and subtracting
  # it from its endpoint gives positive slopes that decrease, hence positive
  # left-hinge coefficients. Negation supplies the decreasing analogues.
  values = if shape === :increasing
    steps
  elseif shape === :decreasing
    -steps
  elseif shape === :convex
    curve .- 0.5 .* curve[end] .* t
  elseif shape === :concave
    -curve .+ 0.5 .* curve[end] .* t
  elseif shape === :increasing_convex
    curve
  elseif shape === :decreasing_concave
    -curve
  elseif shape === :increasing_concave
    curve[end] .- reverse(curve)
  else  # :decreasing_convex
    reverse(curve) .- curve[end]
  end
  # A common positive scale preserves strictness and every coefficient's sign.
  return values ./ maximum(abs, values)
end

function smooth_values(shape, t)
  # Smooth shaped trends with a sinusoidal perturbation, then independent N(0, σ²)
  # noise with σ = 0.1. Curved trends retain the indicated curvature before noise;
  # linear terms give the monotone variants slopes of the required sign. Negation
  # supplies the other directions. Each weighted/unweighted pair shares its noise.
  ripple = sin.(2π .* t)
  trend = if shape in (:increasing, :decreasing)
    t .+ 0.1 .* ripple
  elseif shape in (:convex, :concave)
    (t .- 0.5).^2 .+ 0.02 .* ripple
  elseif shape in (:increasing_convex, :decreasing_concave)
    t .+ t.^2 .+ 0.02 .* ripple
  else  # :increasing_concave, :decreasing_convex
    3 .* t .- t.^2 .+ 0.02 .* ripple
  end
  shape in (:decreasing, :concave, :decreasing_concave, :decreasing_convex) && (trend = -trend)
  return trend .+ 0.1 .* randn(Xoshiro(92_000 + length(t)), length(t))
end

function timeit(f; budget = 2.0, maxreps = 5)
  first_seconds = @elapsed warm_allocations = @allocated result = f()
  if first_seconds >= 60
    return (; result, seconds = first_seconds, median_seconds = first_seconds, spread = 0.0, runs = 1,
            first_seconds, allocations = warm_allocations, allocations_warmed = false)
  end
  GC.gc()
  fast = first_seconds < 0.1
  # maxreps = 1 requests quick mode for warm-ups of 0.1 seconds or more.
  unbudgeted = fast || (first_seconds < 1 && maxreps > 1)
  repetitions = fast ? max(30, maxreps) : unbudgeted ? max(10, maxreps) : maxreps
  times = Float64[]
  total = 0.0
  while length(times) < repetitions && (unbudgeted || isempty(times) || total < budget)
    seconds = @elapsed result = f()
    push!(times, seconds)
    total += seconds
    !unbudgeted && seconds >= 60 && break
  end
  allocations_warmed = maximum(times) < 60
  allocations = allocations_warmed ? (@allocated f()) : warm_allocations
  sort!(times)
  runs = length(times)
  median_seconds = (times[(runs + 1) ÷ 2] + times[runs ÷ 2 + 1]) / 2
  return (; result, seconds = first(times), median_seconds, spread = last(times) - first(times),
          runs, first_seconds, allocations, allocations_warmed)
end

function cone_support(values, t, weights, shape)
  for name in (:_conegenerators, :_nnls)
    isdefined(KernelSmoothing, name) || return (NaN, NaN)
  end
  # Reproduce the weighted residualization in the base _coneproject. These shared
  # helpers also work when the public monotone projection bypasses the cone solver.
  # Normalize the values in both revisions before the diagnostic solve.
  F, E = KernelSmoothing._conegenerators(shape, t)
  sw = sqrt.(weights)
  Fw, Ew = sw .* F, sw .* E
  yw = sw .* (values ./ max(maximum(abs, values), eps()))
  Q = Matrix(qr(Fw).Q)[:, 1:size(F, 2)]
  Er = Ew .- Q * (Q' * Ew)
  yr = yw .- Q * (Q' * yw)
  c = KernelSmoothing._nnls(Er, yr)
  return count(>(0), c), size(E, 2)
end

function write_result(path, t, values, weights, estimate)
  open(path, "w") do io
    println(io, "index,point,value,weight,estimate")
    for i in eachindex(t)
      csv_row(io, (i, t[i], values[i], weights[i], estimate[i]))
    end
  end
end

function benchmark_case(io, dir, family, shape, m, weighted; quick = false)
  t = collect(range(0.0, 1.0; length = m))
  values = family === :strict ? strict_values(shape, t) : smooth_values(shape, t)
  weights = weighted ? 0.5 .+ rand(Xoshiro(93_000 + m), m) : ones(m)
  kind = weighted ? "weighted" : "unweighted"
  logger = WarningLogger(String[])
  measurement, failure = nothing, ""
  try
    measurement = with_logger(logger) do
      timeit(() -> shape_projection(values, t; shape, weights = weighted ? weights : nothing);
             budget = quick ? 0.0 : 2.0, maxreps = quick ? 1 : 5)
    end
  catch err
    err isa InterruptException && rethrow()
    failure = sprint(showerror, err)
  end
  support, generators, support_error = NaN, NaN, ""
  diagnostic_logger = WarningLogger(String[])
  try
    support, generators = with_logger(diagnostic_logger) do
      cone_support(values, t, weights, shape)
    end
  catch err
    err isa InterruptException && rethrow()
    support_error = sprint(showerror, err)
  end
  result_file = ""
  result_sum, result_norm, result_sumsq, objective = NaN, NaN, NaN, NaN
  if measurement !== nothing
    estimate = measurement.result
    result_sum, result_norm, result_sumsq = sum(estimate), norm(estimate), sum(abs2, estimate)
    objective = sum(weights .* abs2.(estimate .- values))
    result_file = joinpath(dir, "$family-$shape-$kind-m$m.csv")
    write_result(result_file, t, values, weights, estimate)
  end
  # There is no counter API shared by the base and new solvers. NaN means unavailable,
  # not zero updates or rebuilds; support diagnostics have their own warning fields.
  csv_row(io, (family, shape, m, kind, support, generators, support == generators,
               measurement === nothing ? NaN : measurement.seconds,
               measurement === nothing ? NaN : measurement.median_seconds,
               measurement === nothing ? NaN : measurement.spread,
               measurement === nothing ? 0 : measurement.runs,
               measurement === nothing ? NaN : measurement.first_seconds,
               measurement === nothing ? NaN : measurement.allocations,
               measurement === nothing ? false : measurement.allocations_warmed,
               !isempty(logger.messages), join(unique(logger.messages), " | "),
               !isempty(diagnostic_logger.messages), join(unique(diagnostic_logger.messages), " | "),
               NaN, NaN, result_sum, result_norm, result_sumsq, objective,
               sum(values), norm(values), sum(weights), norm(weights), result_file,
               support_error, failure, PACKAGE_LOAD_SECONDS, Threads.nthreads(), BLAS.get_num_threads(), VERSION))
  flush(io)
  println(family, " ", shape, " m=", m, " ", kind, ": ",
          measurement === nothing ? failure : string(measurement.seconds, " s"),
          "; support=", support, "/", generators)
  return isempty(failure) && isempty(support_error)
end

function main()
  quick = "--quick" in ARGS
  args = filter(!=("--quick"), ARGS)
  1 <= length(args) <= 2 || error("usage: shape.jl [--quick] OUT.csv [RESULTS_DIR]")
  out = args[1]
  dir = length(args) == 2 ? args[2] : splitext(out)[1] * "-results"
  mkpath(dirname(abspath(out)))
  mkpath(dir)
  ok = true
  open(out, "w") do io
    csv_row(io, ("case", "shape", "m", "weights", "support", "generators", "full_support",
                 "seconds", "median_seconds", "spread_seconds", "runs", "first_call_seconds", "allocations",
                 "allocations_warmed", "warning", "warning_messages", "support_warning",
                 "support_warning_messages", "updates", "rebuilds", "result_sum", "result_norm",
                 "result_sumsq", "weighted_objective", "input_sum", "input_norm", "weights_sum",
                 "weights_norm", "result_file", "support_error", "error", "package_load_seconds",
                 "threads", "blas_threads", "julia_version"))
    for family in (:strict, :smooth)
      sizes = quick ? (20, 40) : family === :strict ? (200, 400, 800) : (1000,)
      for m in sizes, shape in SHAPES, weighted in (false, true)
        ok = benchmark_case(io, dir, family, shape, m, weighted; quick) && ok
      end
    end
  end
  ok || error("some benchmark cases failed; see $out")
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
