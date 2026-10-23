using Documenter
using KernelSmoothing

makedocs(;
  sitename = "KernelSmoothing.jl",
  modules = [KernelSmoothing],
  format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
  pages = [
    "Home" => "index.md",
    "Density estimation" => "density.md",
    "Kernels" => "kernels.md",
    "Prepared data" => "data.md",
    "Kernel sums" => "sums.md",
  ],
)

deploydocs(; repo = "github.com/mlakolar/KernelSmoothing.jl.git", devbranch = "master")
