using Documenter
using KernelSmoothing

makedocs(;
  sitename = "KernelSmoothing.jl",
  modules = [KernelSmoothing],
  format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
  pages = [
    "Home" => "index.md",
    "Density estimation" => "density.md",
    "Multivariate density estimation" => "multivariate.md",
    "Densities on a bounded support" => "bounded.md",
    "Local polynomial regression" => "regression.md",
    "Regression discontinuity" => "rd.md",
    "Bandwidth selection" => "bandwidths.md",
    "Kernels" => "kernels.md",
    "Prepared data" => "data.md",
    "Kernel sums" => "sums.md",
    "Building blocks" => "building.md",
    "Comparison with R" => "r.md",
  ],
)

deploydocs(; repo = "github.com/mlakolar/KernelSmoothing.jl.git", devbranch = "master")
