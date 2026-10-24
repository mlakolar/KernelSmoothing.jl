using Test

tests = [
  "kernels",
  "data",
  "engines",
  "bandwidths",
  "selectors",
  "kde",
  "locpoly",
  "plugin",
  "kdeinference",
  "bands",
  "honest",
  "clusters",
  "rd",
  "lpdensity",
  "recipes",
  "aqua"
]

for t in tests
  f = "$t.jl"
  println("* running $f ...")
  t = @elapsed include(f)
  println("done (took $t seconds).")
end
