# Reference values from lpdensity 3.0.1 for one outlier, for the tests of KernelSmoothing.jl:
# the rules standardize by the standard deviation, and the plug-in rules reject the
# ill-conditioned pilot fits and fall back on their floors.
RLIB <- commandArgs(trailingOnly = TRUE)[1]
suppressPackageStartupMessages(library(lpdensity, lib.loc = RLIB))
i <- 1:2000
x <- c(qnorm((i * 0.6180339887498949) %% 1), 1e4)
for (bs in c("mse-dpi", "imse-dpi", "mse-rot")) {
  r <- lpdensity(data = x, grid = c(-1, 0, 1), bwselect = bs)
  cat(bs, ": bw = ", paste(sprintf("%.17g", r$Estimate[, "bw"]), collapse = ", "),
      "; f_p = ", paste(sprintf("%.17g", r$Estimate[, "f_p"]), collapse = ", "), "\n", sep = "")
}
