# Shared definitions of the simulation studies of KernelSmoothing.jl (see README.md).
# Every study script is run as
#
#     julia --project=. STUDY.jl DESIGN N FIRSTREP NREPS OUT
#
# and writes the results of the replications FIRSTREP, …, FIRSTREP + NREPS - 1 of one
# design and sample size to the CSV file OUT. Replication r of a design draws its data
# from Xoshiro(seed(study, design, n, r)), so that any replication can be redone alone.
using KernelSmoothing
using Distributions
using Random
using Statistics
using LinearAlgebra

const Z95 = quantile(Normal(), 0.975)

# the arguments of a study script
function studyargs()
  length(ARGS) == 5 || error("usage: julia STUDY.jl DESIGN N FIRSTREP NREPS OUT")
  return (design = ARGS[1], n = parse(Int, ARGS[2]), first = parse(Int, ARGS[3]), nreps = parse(Int, ARGS[4]),
          out = ARGS[5])
end

# A seed from the study, the design, the sample size and the replication, stable across
# Julia versions: the studies and designs are numbered by their position in fixed lists.
function seed(studycode::Integer, designcode::Integer, n::Integer, rep::Integer)
  @assert 0 < studycode < 100 && 0 < designcode < 1000 && 0 < n < 10^7 && 0 < rep < 10^5
  return ((studycode * 1000 + designcode) * 10^7 + n) * 10^5 + rep
end

designcode(names, design) = something(findfirst(==(design), names), 0) |>
  (k -> k > 0 ? k : error("unknown design $design; the designs are $(join(names, ", "))"))

# rows of results as CSV with a header; NaN and missing values are written as NaN
function writerows(path, header, rows)
  open(path, "w") do io
    println(io, join(header, ","))
    for r in rows
      println(io, join((v isa AbstractFloat ? repr(v) : string(v) for v in r), ","))
    end
  end
end

# the trapezoid rule on a grid
trapezoid(x, y) = sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2 for i in 1:length(x)-1)
