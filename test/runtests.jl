using Test

tests = [
  "kernels",
  "data",
  "engines",
  "bandwidths",
  "selectors",
  "kde",
  "locpoly",
  "smoother",
  "surface",
  "plugin",
  "kdeinference",
  "bands",
  "honest",
  "clusters",
  "rd",
  "rddensity",
  "rdhonest",
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
