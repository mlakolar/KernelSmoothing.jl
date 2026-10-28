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
  "rdmulti",
  "rdpower",
  "rdlocrand",
  "lpdensity",
  "shape",
  "nnls",
  "binning",
  "mvkde",
  "mvkde_ties",
  "boundedkde",
  "conditional",
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
