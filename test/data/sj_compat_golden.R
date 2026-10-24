# Reference paths around stats::bw.SJ's binning threshold and its sparse failures.
cat("# Generated from R", as.character(getRversion()), "stats::bw.SJ by sj_compat_golden.R.\n")
cat("const SJ_COMPAT = [\n")
for (n in c(500L, 501L)) {
  i <- seq_len(n)
  x <- qnorm((i * 0.6180339887498949) %% 1) + 0.3 * sin(2.71 * i)
  for (variant in c("plain", "reflected", "shifted")) {
    v <- switch(variant, plain = x, reflected = -x, shifted = x + 13.25)
    cat(sprintf("  (n = %d, variant = :%s, ste = %.17g, dpi = %.17g),\n",
                n, variant, bw.SJ(v), bw.SJ(v, method = "dpi")))
  }
}
cat("]\n")
# Both methods raise errors for these samples; Julia checks the same failures.
for (v in list(rep(2, 50), c(rep(0, 49), 1))) {
  for (method in c("ste", "dpi")) {
    stopifnot(inherits(try(bw.SJ(v, method = method), silent = TRUE), "try-error"))
  }
}
