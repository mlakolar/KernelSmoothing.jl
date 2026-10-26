# The time of the R counterparts of the calls in timings.jl, on data from the same
# distributions (evmix's beta kernels are left out: fbckden with bcmethod = "beta1"
# did not finish in 45 minutes at n = 1000): the minimum of up to 5 runs (one run once a call takes more than 20
# seconds). Usage: Rscript timings.R OUT.csv [NMAX], with the libraries on the path.
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
nmax <- if (length(args) >= 2) as.numeric(args[2]) else 1e6
pkgs <- c("nprobust", "lpdensity", "rdrobust", "rddensity", "ks", "np", "quantreg", "locfit", "evmix", "KernSmooth")
for (p in pkgs) suppressPackageStartupMessages(requireNamespace(p, quietly = TRUE))
options(np.messages = FALSE)
timeit <- function(f) {
  f()
  best <- Inf; total <- 0
  for (r in 1:5) {
    t <- system.time(f())[["elapsed"]]
    best <- min(best, t); total <- total + t
    if (t > 20 || total > 60) break
  }
  best
}
pts <- seq(0.1, 0.9, length.out = 30)
cases <- list(
  list("stats::density(x, bw = 'SJ')", 1e6, function(d) density(d$x, bw = "SJ")),
  list("stats::bw.ucv(x)", 1e5, function(d) bw.ucv(d$x)),
  list("nprobust::kdrobust(x)", 1e6, function(d) nprobust::kdrobust(d$x, eval = pts)),
  list("lpdensity::lpdensity(x)", 1e5, function(d) lpdensity::lpdensity(d$x, grid = pts)),
  list("nprobust::lprobust(y, x)", 1e6, function(d) nprobust::lprobust(d$y, d$x, eval = pts)),
  list("nprobust::lprobust(y, x, bwselect = 'imse-dpi')", 1e6, function(d) nprobust::lprobust(d$y, d$x, eval = pts, bwselect = "imse-dpi")),
  list("rdrobust::rdrobust(y, x - 0.5)", 1e6, function(d) rdrobust::rdrobust(d$y, d$x - 0.5)),
  list("rddensity::rddensity(x - 0.5)", 1e5, function(d) rddensity::rddensity(d$x - 0.5)),
  list("glm by point, local logit at fixed h", 1e6, function(d) for (x0 in pts) {
    u <- (d$x - x0) / 0.2; w <- pmax(0.75 * (1 - u^2), 0); k <- w > 0
    suppressWarnings(glm.fit(cbind(1, u[k]), d$b[k], weights = w[k], family = binomial()))
  }),
  list("locfit::locfit.raw(x), density", 1e5, function(d) locfit::locfit.raw(locfit::lp(d$x, nn = 0.7), kern = "epan", ev = pts)),
  list("quantreg::lprq(x, y, h)", 1e5, function(d) quantreg::lprq(d$x, d$y, h = 0.1, m = 30)),
  list("np::npcdens(y ~ x), cv.ml", 1e4, function(d) np::npcdens(np::npcdensbw(ydat = data.frame(y = d$y), xdat = data.frame(x = d$x), bwmethod = "cv.ml", nmulti = 1))),
  list("evmix::fbckden(x, bcmethod = 'reflect')", 1e3, function(d) evmix::fbckden(d$x, linit = 0.05, bcmethod = "reflect")),
  list("ks::Hpi(X2)", 1e4, function(d) ks::Hpi(d$X2, binned = FALSE)),
  list("ks::Hpi(X2), binned", 1e6, function(d) ks::Hpi(d$X2)),
  list("ks::Hscv(X2)", 1e4, function(d) ks::Hscv(d$X2, binned = FALSE)),
  list("ks::Hscv(X2), binned", 1e6, function(d) ks::Hscv(d$X2))
)
con <- file(out, "w")
writeLines("call,n,seconds", con)
for (n in c(1e3, 1e4, 1e5, 1e6)) {
  if (n > nmax) next
  set.seed(n)
  x <- rbeta(n, 2, 2)
  d <- list(x = x, y = sin(2 * pi * x) + 0.5 * rnorm(n), b = as.numeric(runif(n) < 1 / (1 + exp(-2 * sin(2 * pi * x)))),
            X2 = cbind(x, 0.5 * x + 0.3 * rnorm(n)))
  for (cs in cases) {
    if (n > cs[[2]]) next
    t <- tryCatch(timeit(function() cs[[3]](d)), error = function(e) { message(cs[[1]], ": ", conditionMessage(e)); NaN })
    writeLines(sprintf("\"%s\",%d,%.6g", cs[[1]], as.integer(n), t), con); flush(con)
    message(sprintf("%-60s %8d %10.4g s", cs[[1]], as.integer(n), t))
  }
}
close(con)
