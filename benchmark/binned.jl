# Shared samples and selection times for the binned bandwidth matrices (#86).
# Generate the samples once, then time them with the thread count set by the caller:
#
#     julia --project=benchmark benchmark/binned.jl datagen samples
#     julia --project=benchmark -t 1 benchmark/binned.jl timings samples julia-1.csv
#     julia --project=benchmark -t 4 benchmark/binned.jl timings samples julia-4.csv
#
# Times exclude CSV input. Fast calls are warmed up, then timed at least three
# times within a budget; a first call taking 60 seconds is retained without a rerun
# (and includes compilation). Matrix entries are the lower triangle, by columns,
# padded with NaN for d = 2. Failures remain in the CSV, with their error messages.
using KernelSmoothing
using Distributions
using Random

const NS = [10_000, 100_000]
const RULES = [("PI", PlugInMatrix), ("SCV", SCVMatrix), ("LSCV", LSCVMatrix)]
const VECH = [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)]

sample_path(dir, d, n) = joinpath(dir, "sample-d$d-n$n.csv")
csv_cell(x) = x isa AbstractString ? "\"" * replace(x, '"' => "\"\"", '\n' => ' ', '\r' => ' ') * "\"" : string(x)
csv_row(io, row) = println(io, join(csv_cell.(row), ","))

function datagen(dir)
  mkpath(dir)
  # X2 in timings.jl is not Gaussian; use a normal with correlation 0.7 instead.
  targets = [MvNormal([0.0, 0.0], [1.0 0.7; 0.7 1.0]),
             MixtureModel([MvNormal([-1.0, 0.0, 0.0], [0.25 0.0 0.0; 0.0 0.25 0.0; 0.0 0.0 0.25]),
                           MvNormal([1.0, 0.0, 0.0], [1.0 0.5 0.5; 0.5 1.0 0.5; 0.5 0.5 1.0])], [0.5, 0.5])]
  for (k, D) in enumerate(targets), n in NS
    d = k + 1
    X = rand(Xoshiro(86_000_000 + 10n + d), D, n)
    open(sample_path(dir, d, n), "w") do io
      println(io, join(("x$a" for a in 1:d), ","))
      for i in 1:n
        csv_row(io, X[:, i])
      end
    end
    println(sample_path(dir, d, n))
  end
end

function read_sample(path, d, n)
  X = Matrix{Float64}(undef, d, n)
  open(path) do io
    split(readline(io), ",") == ["x$a" for a in 1:d] || error("unexpected header in $path")
    for i in 1:n
      eof(io) && error("too few observations in $path")
      values = split(readline(io), ",")
      length(values) == d || error("unexpected dimension in $path")
      X[:, i] .= parse.(Float64, values)
    end
    eof(io) || error("too many observations in $path")
  end
  return X
end

# The minimum over runs taking at least budget seconds in all, with its selection.
function timeit(f; budget = 2.0, maxreps = 20)
  warmtime = @elapsed warm = f()
  warmtime >= 60 && return warm, warmtime, 1
  best, total, reps = Inf, 0.0, 0
  selection = warm
  while reps < maxreps && (reps < 3 || total < budget)
    t = @elapsed s = f()
    if t < best
      best, selection = t, s
    end
    total += t
    reps += 1
    (t >= 60 || total >= 60) && break
  end
  return selection, best, reps
end

function timings(dir, out)
  open(out, "w") do io
    csv_row(io, ("rule", "d", "n", "settings", "binned", "gridsize", "threads", "seconds", "runs",
                "iterations", "converged", "criterion", "h11", "h21", "h31", "h22", "h32", "h33", "error"))
    for d in (2, 3), n in NS
      X = read_sample(sample_path(dir, d, n), d, n)
      for (name, constructor) in RULES, exact in (false, true)
        exact && n > 10_000 && continue
        settings = exact ? "exact" : "default"
        row = try
          rule = exact ? constructor(binned = false) : constructor()
          s, seconds, runs = timeit(() -> select_bandwidth(rule, X))
          details = s.details
          H = s.h
          size(H) == (d, d) && all(isfinite, H) && isfinite(details.criterion) ||
            error("non-finite criterion or invalid bandwidth matrix")
          entries = [a <= d && b <= d ? H[a, b] : NaN for (a, b) in VECH]
          (name, d, n, settings, details.binned, something(details.gridsize, NaN), Threads.nthreads(),
           seconds, runs, details.iterations, details.converged, details.criterion, entries..., "")
        catch err
          message = sprint(showerror, err)
          @warn "selection failed" name d n settings message
          (name, d, n, settings, !exact, NaN, Threads.nthreads(), NaN, 0, NaN, false, NaN,
           fill(NaN, length(VECH))..., message)
        end
        csv_row(io, row)
        flush(io)
        println(name, " d=", d, " n=", n, " ", settings, ": ", row[8], " s")
      end
    end
  end
end

function main()
  if length(ARGS) == 2 && ARGS[1] == "datagen"
    datagen(ARGS[2])
  elseif length(ARGS) == 3 && ARGS[1] == "timings"
    timings(ARGS[2], ARGS[3])
  else
    error("usage: binned.jl datagen SAMPLEDIR | binned.jl timings SAMPLEDIR OUT.csv")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
