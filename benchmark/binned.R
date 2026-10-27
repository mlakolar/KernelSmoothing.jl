# Selection times for ks 1.15.3 on the CSV samples written by binned.jl:
#
#     julia --project=benchmark benchmark/binned.jl datagen samples
#     Rscript benchmark/binned.R samples ks.csv
#
# Fast calls are warmed up, then timed with system.time's elapsed time; retain a
# first call taking 60 seconds without a rerun. Entries are the lower triangle,
# by columns, padded with NA for d = 2. Errors are recorded and the next case runs.
# These are method comparisons: see docs/src/r.md, notes 16-18, for the SAMSE
# functional ordering, the different default 3-D pilot, and SCV pilot and LSCV
# optimizer differences. pilot = "samse" alone does not remove those differences.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: Rscript binned.R SAMPLEDIR OUT.csv")
if (!requireNamespace("ks", quietly = TRUE)) stop("ks 1.15.3 is required")
ks_version <- as.character(utils::packageVersion("ks"))
if (ks_version != "1.15.3") stop("expected ks 1.15.3; found ", ks_version)

timeit <- function(f, budget = 2, maxreps = 5) {
  warmtime <- system.time(warm <- f())[["elapsed"]]
  if (warmtime >= 60) return(list(H = warm, seconds = warmtime, runs = 1L))
  best <- Inf; total <- 0; selection <- warm; reps <- 0L
  for (r in seq_len(maxreps)) {
    t <- system.time(H <- f())[["elapsed"]]
    if (t < best) { best <- t; selection <- H }
    total <- total + t; reps <- r
    if (t >= 60 || total >= 60 || (r >= 3 && total >= budget)) break
  }
  list(H = selection, seconds = best, runs = reps)
}

con <- file(args[2], "w")
first <- TRUE
for (d in c(2L, 3L)) for (n in c(10000L, 100000L)) {
  path <- file.path(args[1], sprintf("sample-d%d-n%d.csv", d, n))
  X <- as.matrix(read.csv(path, check.names = FALSE))
  if (!identical(dim(X), c(n, d)) || !all(is.finite(X))) stop("invalid sample: ", path)
  cases <- list(
    list(rule = "PI", fun = ks::Hpi, args = list(), settings = "default"),
    list(rule = "SCV", fun = ks::Hscv, args = list(), settings = "default"),
    list(rule = "LSCV", fun = ks::Hlscv, args = list(), settings = "default")
  )
  if (d == 3L) cases <- c(cases, list(
    list(rule = "PI", fun = ks::Hpi, args = list(pilot = "samse"), settings = "pilot=samse"),
    list(rule = "SCV", fun = ks::Hscv, args = list(pilot = "samse"), settings = "pilot=samse")
  ))
  for (cs in cases) {
    result <- tryCatch({
      fit <- timeit(function() do.call(cs$fun, c(list(x = X), cs$args)))
      if (!identical(dim(fit$H), c(d, d)) || !all(is.finite(fit$H))) stop("non-finite or invalid matrix")
      c(fit, list(error = ""))
    }, error = function(e) {
      message(cs$rule, " d=", d, " n=", n, ": ", conditionMessage(e))
      list(H = matrix(NA_real_, d, d), seconds = NA_real_, runs = 0L,
           error = gsub("[\r\n]", " ", conditionMessage(e)))
    })
    indices <- list(c(1, 1), c(2, 1), c(3, 1), c(2, 2), c(3, 2), c(3, 3))
    entries <- vapply(indices, function(ij) if (all(ij <= d)) result$H[ij[1], ij[2]] else NA_real_, numeric(1))
    row <- data.frame(rule = cs$rule, d = d, n = n, settings = cs$settings,
                      seconds = result$seconds, runs = result$runs, ks_version = ks_version,
                      h11 = entries[1], h21 = entries[2], h31 = entries[3],
                      h22 = entries[4], h32 = entries[5], h33 = entries[6], error = result$error)
    write.table(row, con, sep = ",", row.names = FALSE, col.names = first, quote = TRUE,
                qmethod = "double", na = "NA")
    first <- FALSE
    flush(con)
    message(sprintf("%s d=%d n=%d %s: %.6g s", cs$rule, d, n, cs$settings, result$seconds))
  }
}
close(con)
