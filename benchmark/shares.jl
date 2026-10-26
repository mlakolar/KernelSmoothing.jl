# Where the time of a default call goes: preparing the data (SmoothingData, which
# sorts it), choosing the bandwidth, and estimating at the chosen bandwidth, on one
# thread, at n = 10⁴ and 10⁵. Writes a CSV to the file given as the first argument.
#
#     julia --project=. -t 1 shares.jl shares.csv
using KernelSmoothing
using Random

best(f; reps = 5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

function main()
  open(ARGS[1], "w") do io
    println(io, "call,n,prepare,select,estimate,total")
    for n in (10_000, 100_000)
      rng = Xoshiro(n)
      x = rand(rng, n) .^ 1.5
      y = sin.(2π .* x) .+ 0.5 .* randn(rng, n)
      d = SmoothingData(x, y)
      dx = SmoothingData(x)
      k, tk = EpanechnikovKernel(), TriangularKernel()
      prep = best(() -> SmoothingData(x, y))
      prepx = best(() -> SmoothingData(x))
      rows = Any[]
      # local polynomial regression with the MSE-optimal bandwidths at its 30 points
      pts = regression_points(d)
      h = select_bandwidth(MSEOptimal(), d, k; points = pts).h
      push!(rows, ("LocalPolynomialRegression", prep, best(() -> select_bandwidth(MSEOptimal(), d, k; points = pts)),
                   best(() -> fit(LocalPolynomialRegression, d; points = pts, bandwidth = h)),
                   best(() -> fit(LocalPolynomialRegression, x, y))))
      # kernel density estimate with the bandwidth of Sheather and Jones, on 512 points
      g = collect(range(0, 1; length = 512))
      hk = select_bandwidth(SheatherJones(), dx, k).h
      push!(rows, ("KDE", prepx, best(() -> select_bandwidth(SheatherJones(), dx, k)),
                   best(() -> pdf.(Ref(fit(KDE, dx; bandwidth = hk)), g)), best(() -> pdf.(Ref(fit(KDE, x)), g))))
      # pointwise density intervals
      pd = collect(range(0.1, 0.9; length = 30))
      hp = select_bandwidth(MSEOptimal(), dx, k; points = pd).h
      push!(rows, ("PointwiseDensity", prepx, best(() -> select_bandwidth(MSEOptimal(), dx, k; points = pd)),
                   best(() -> fit(PointwiseDensity, dx; points = pd, bandwidth = hp)),
                   best(() -> fit(PointwiseDensity, x; points = pd))))
      # regression discontinuity at 0.5 with rdrobust's MSE-optimal bandwidth
      xr = x .- 0.5
      dr = SmoothingData(xr, y)
      m = fit(RegressionDiscontinuity, xr, y)
      push!(rows, ("RegressionDiscontinuity", prep, best(() -> fit(RegressionDiscontinuity, xr, y)) -
                   best(() -> fit(RegressionDiscontinuity, xr, y; bandwidth = m.h)),
                   best(() -> fit(RegressionDiscontinuity, xr, y; bandwidth = m.h)), best(() -> fit(RegressionDiscontinuity, xr, y))))
      for r in rows
        println(io, r[1], ",", n, ",", join(r[2:end], ","))
        println(rpad(r[1], 26), lpad(n, 8), "  prepare ", round(r[2]; sigdigits = 2), "  select ", round(r[3]; sigdigits = 2),
                "  estimate ", round(r[4]; sigdigits = 2), "  total ", round(r[5]; sigdigits = 2))
      end
    end
  end
end

main()
