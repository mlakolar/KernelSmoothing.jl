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
  "semiparametric",
  "locallik",
  "plugin",
  "kdeinference",
  "bands",
  "honest",
  "clusters",
  "rd",
  "rddensity",
  "rdhonest",
  "lpdensity",
  "mvkde",
  "boundedkde",
  "recipes",
  "building",
  "aqua"
]

for t in tests
  f = "$t.jl"
  println("* running $f ...")
  t = @elapsed include(f)
  println("done (took $t seconds).")
end
