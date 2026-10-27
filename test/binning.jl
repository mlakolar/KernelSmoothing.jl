module BinningTest

using Test
using KernelSmoothing
using KernelSmoothing: _MatrixBins, _linearbinning, _lagcorrelations, _matrixbins, _usebinning,
  _binningdetails, _mvdata, _pretransform, _multiindices, _psihat, _gausspairs, _lscvcriterion, _scvcriterion
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "data", "ks_golden.jl"))

# The signed correlations by their definition, independent of the FFT.
function direct_correlations(c, e = c)
  M = size(c)
  B = zeros(Tuple(2 .* M .- 1))
  for I in CartesianIndices(c), J in CartesianIndices(e)
    K = CartesianIndex(Tuple(J - I) .+ M)
    B[K] += c[I] * e[J]
  end
  return B
end

# One observation on a fixed grid, for the smeared self-pair oracle.
function observation_bins(x, origin, spacing, M)
  D = length(x)
  c = zeros(ntuple(_ -> M, D))
  t = (x .- collect(origin)) ./ collect(spacing)
  pos = min.(floor.(Int, t), M - 2)
  f = t .- pos
  for I in CartesianIndices(ntuple(_ -> 2, D))
    c[CartesianIndex(Tuple(pos .+ collect(Tuple(I))))] =
      prod(I[a] == 1 ? 1 - f[a] : f[a] for a in 1:D)
  end
  return c
end

@testset "multivariate binning" begin
  @testset "FFT correlations and binning moments" begin
    rng = Xoshiro(86)
    for dims in ((3, 5), (3, 4, 5))
      c, e = rand(rng, dims...), rand(rng, dims...)
      for v in (c, e)
        ref = direct_correlations(c, v)
        @test _lagcorrelations(c, v) ≈ ref atol = 1e-13 * maximum(abs, ref) rtol = 1e-13
      end
    end
    @test_throws DimensionMismatch _lagcorrelations(zeros(3, 3), zeros(3, 4))
    for D in (2, 3)
      y = randn(rng, D, 19)
      w = rand(rng, 19) .+ 0.2
      W = sum(w)
      b = _linearbinning(y, w, W, 7; loo = true)
      @test sum(b.counts) ≈ W rtol = 1e-14
      @test sum(b.loo_counts) ≈ sum(w ./ (W .- w)) rtol = 1e-14
      for a in 1:D
        first_moment = sum(b.counts[I] * (b.origin[a] + (I[a] - 1) * b.spacing[a]) for I in CartesianIndices(b.counts))
        @test first_moment ≈ sum(w .* y[a, :]) atol = 1e-13 * W
      end
      bins = _MatrixBins(y, w, W, 7; loo = true)
      @test sum(bins.ww) ≈ (W^2 - sum(abs2, w)) / 2 rtol = 1e-13
      @test sum(bins.loo) ≈ 1 rtol = 1e-13
      @test bins.ww == _MatrixBins(y, w, W, 7; loo = true).ww
    end
  end

  @testset "lattice oracle" begin
    lattice = [0.0 2.0 0.5 0.5 1.5 2.0 0.0;
               0.0 2.0 1.0 1.0 0.5 0.0 2.0;
               0.0 2.0 1.5 1.5 0.5 2.0 2.0]
    for D in (2, 3), w in (nothing, ones(7), [0.2, 3.0, 0.5, 1.5, 2.0, 0.8, 1.0])
      y = lattice[1:D, :]
      W = w === nothing ? 7.0 : sum(w)
      n = w === nothing ? 7.0 : W^2 / sum(abs2, w)
      bins = _MatrixBins(y, w, W, 5; loo = true)
      for j in (4, 6)
        R = _multiindices(D, j)
        exact = _psihat(y, w, W, n, 0.9, R)
        binned = _psihat(y, w, W, n, 0.9, R, bins)
        @test [binned[r] for r in eachcol(R)] ≈ [exact[r] for r in eachcol(R)] atol = 1e-12 rtol = 1e-12
      end
      H = [0.7 0.13 -0.07; 0.13 0.6 0.09; -0.07 0.09 0.8][1:D, 1:D]
      for grad in (false, true)
        s, M = _gausspairs(y, w, W, [2H, H], [false, true], grad)
        sb, Mb = _gausspairs(y, w, W, [2H, H], [false, true], grad, bins)
        @test sb ≈ s atol = 1e-12 rtol = 1e-12
        for k in eachindex(M)
          @test Mb[k] ≈ M[k] atol = 1e-12 rtol = 1e-12
        end
      end
      @test _lscvcriterion(y, w, W, n, H, bins)[1] ≈ _lscvcriterion(y, w, W, n, H)[1] atol = 1e-12 rtol = 1e-12
      G = 0.3 .* Matrix(1.0I, D, D)
      @test _scvcriterion(y, w, W, n, H, G, 0.0, bins)[1] ≈
            _scvcriterion(y, w, W, n, H, G, 0.0)[1] atol = 1e-12 rtol = 1e-12
    end
  end

  @testset "smeared self-pairs" begin
    for D in (2, 3)
      y = [0.0 2.0 0.37 0.37; 0.0 2.0 1.23 1.23; 0.0 2.0 0.79 0.79][1:D, :]
      w = [0.5, 2.0, 1.5, 0.8]
      W = sum(w)
      b = _linearbinning(y, w, W, 3; loo = true)
      sww, sab = zeros(ntuple(_ -> 5, D)), zeros(ntuple(_ -> 5, D))
      for i in eachindex(w)
        c = observation_bins(y[:, i], b.origin, b.spacing, 3)
        self = direct_correlations(c)
        sww .+= w[i]^2 .* self
        sab .+= w[i]^2 / (W - w[i]) .* self
      end
      center = CartesianIndices(ntuple(_ -> 2:4, D))
      @test b.self_ww ≈ sww[center] atol = 1e-14
      @test b.self_ab ≈ sab[center] atol = 1e-14
      offdiag = direct_correlations(b.counts) .- sww
      loo = (direct_correlations(b.loo_counts, b.counts) .+
             direct_correlations(b.counts, b.loo_counts) .- 2 .* sab) ./ W
      H = [0.6 0.12 -0.08; 0.12 0.5 0.03; -0.08 0.03 0.7][1:D, 1:D]
      φ(u) = (2π)^(-D / 2) / sqrt(det(H)) * exp(-dot(u, H \ u) / 2)
      ref = zeros(2)
      for I in CartesianIndices(offdiag)
        u = collect(Tuple(I) .- 3) .* collect(b.spacing)
        ref[1] += offdiag[I] * φ(u) / 2
        ref[2] += loo[I] * φ(u) / 2
      end
      bins = _MatrixBins(y, w, W, 3; loo = true)
      @test _gausspairs(y, w, W, [H, H], [false, true], true, bins)[1] ≈ ref atol = 1e-12 rtol = 1e-12
    end
  end

  @testset "gradients and refinement" begin
    for D in (2, 3)
      x = D == 2 ? X2 : X3
      y = x[:, 1:17]
      w = 1.0 .+ mod.(1:size(y, 2), 4)
      W = sum(w)
      n = W^2 / sum(abs2, w)
      bins = _MatrixBins(y, w, W, 7; loo = true)
      H = [0.5 0.07 -0.04; 0.07 0.4 0.06; -0.04 0.06 0.6][1:D, 1:D]
      G = 0.3 .* Matrix(1.0I, D, D)
      for fg in (h -> _lscvcriterion(y, w, W, n, h, bins), h -> _scvcriterion(y, w, W, n, h, G, 0.0, bins))
        _, grad = fg(H)
        for a in 1:D, c in 1:a
          E = zeros(D, D)
          E[a, c] = E[c, a] = 1
          num = (fg(H .+ 1e-6 .* E)[1] - fg(H .- 1e-6 .* E)[1]) / 2e-6
          @test sum(grad .* E) ≈ num rtol = 1e-5 atol = 1e-10
        end
      end
      # The whole golden sample, at fixed H and G in the transformed coordinates.
      data = _mvdata(x, nothing)
      _, y, _ = _pretransform(data, :sphere)
      W = data.sumw
      errors = map((9, 17)) do M
        bins = _MatrixBins(y, nothing, W, M; loo = true)
        lscv = abs(_lscvcriterion(y, nothing, W, W, H, bins)[1] - _lscvcriterion(y, nothing, W, W, H)[1])
        s0 = _gausspairs(y, nothing, W, [2G], [false], false, bins)[1][1]
        s0exact = _gausspairs(y, nothing, W, [2G], [false], false)[1][1]
        c0 = (2π)^(-D / 2) / sqrt(det(2G)) / W
        scv = abs(_scvcriterion(y, nothing, W, W, H, G, c0 + 2s0 / W^2, bins)[1] -
                  _scvcriterion(y, nothing, W, W, H, G, c0 + 2s0exact / W^2)[1])
        R = _multiindices(D, 4)
        ψ = _psihat(y, nothing, W, W, 0.8, R, bins)
        ψexact = _psihat(y, nothing, W, W, 0.8, R)
        pi = norm([ψ[r] - ψexact[r] for r in eachcol(R)])
        (lscv, scv, pi)
      end
      @test all(errors[2] .< errors[1])
    end
  end

  @testset "default binning by rule" begin
    # Use the selection's cache and details without minimizing exact sums at large n.
    for (D, n) in ((2, 500), (2, 501), (3, 1001)), diagonal in (false, true)
      x = D == 2 ? X2 : X3
      data = _mvdata(repeat(x[:, 1:10], 1, cld(n, 10))[:, 1:n], nothing)
      H = 100.0 .* Matrix(1.0I, D, D)
      for R in (PlugInMatrix, SCVMatrix, LSCVMatrix), binned in (nothing, false, true)
        rule = R(; diagonal, binned, gridsize = 3)
        bins = _matrixbins(rule, data.x, data; loo = R == LSCVMatrix)
        details = _binningdetails(bins, H)
        expected = binned === nothing ? (D == 2 && n > 500 && R != LSCVMatrix) : binned
        @test details.binned == expected
        @test details.gridsize === (expected ? 3 : nothing)
      end
    end
  end

  @testset "matrix line search domain errors" begin
    for diagonal in (false, true)
      rejected = Ref(0)
      fg = H -> begin
        D = det(H)
        if D < 0.04
          rejected[] += 1
          throw(DomainError(D, "the trial determinant is too small"))
        end
        tr(H), Matrix(1.0I, 2, 2)
      end
      H, f, _, _ = KernelSmoothing._minimizematrix(fg, Matrix(1.0I, 2, 2), diagonal)
      @test rejected[] > 0
      @test all(isfinite, H) && isposdef(H)
      @test isfinite(f) && det(H) >= 0.04
    end
    @test_throws ArgumentError KernelSmoothing._minimizematrix(H -> throw(DomainError(H)), fill(1.0, 1, 1), true)
  end

  @testset "collapsed matrix resolution" begin
    bins = _MatrixBins([0.0 2.0; 0.0 2.0], nothing, 2.0, 3)
    for H in ([1.0 0.0; 0.0 0.0], [1.0 0.0; 0.0 -2e-15])
      details = @test_logs (:warn, r"coarse") _binningdetails(bins, H)
      @test details.resolution.selected == 0.0
    end
  end

  @testset "defaults, validation and details" begin
    for (D, n) in ((2, 501), (3, 1001))
      @test !_usebinning(nothing, D, n - 1)
      @test _usebinning(nothing, D, n) == (D == 2)
      @test !_usebinning(false, D, n)
      @test _usebinning(true, D, n)
      x = repeat((D == 2 ? X2 : X3)[:, 1:10], 1, cld(n, 10))[:, 1:n]
      # A zero weight is dropped; the effective sample size does not set the threshold.
      w = [100.0; ones(n - 2); 0.0]
      d = _mvdata(x, w)
      @test _matrixbins(d.x, d, nothing, 3) === nothing
      w[end] = 1.0
      d = _mvdata(x, w)
      @test (_matrixbins(d.x, d, nothing, 3) !== nothing) == (D == 2)
      d = _mvdata(x[:, 1:10], nothing)
      b = _matrixbins(d.x, d, true, nothing)
      @test b.gridsize == (D == 2 ? 151 : 51)
    end
    for D in (1, 4)
      @test !_usebinning(nothing, D, 2000)
      @test_throws ArgumentError _usebinning(true, D, 10)
    end
    for R in (PlugInMatrix, SCVMatrix, LSCVMatrix)
      @test R().binned === nothing && R().gridsize === nothing
      @test R(binned = false, gridsize = 3).gridsize == 3
      @test_throws ArgumentError R(gridsize = 2)
      @test_throws ArgumentError R(binned = false, gridsize = 0)
      @test_throws ArgumentError select_bandwidth(R(binned = true), X2[1:1, :])
      x = X2[:, 1:30]
      sel = select_bandwidth(R(binned = true, gridsize = 17), x)
      @test sel.details.binned && sel.details.gridsize == 17
      @test length(sel.details.spacing) == 2
      A, _, _ = _pretransform(_mvdata(x, nothing), :sphere)
      Hy = A * bandwidth(sel) * A'
      @test sel.details.resolution.selected ≈ sqrt(eigmin(Symmetric(Hy))) / maximum(sel.details.spacing)
      pilot = R == PlugInMatrix ? minimum(values(sel.details.pilots)) : R == SCVMatrix ? sel.details.pilot : nothing
      @test sel.details.resolution.pilot === nothing ? pilot === nothing :
            sel.details.resolution.pilot ≈ pilot / maximum(sel.details.spacing)
      exact = select_bandwidth(R(binned = false, gridsize = 17), x)
      @test !exact.details.binned && exact.details.gridsize === nothing && exact.details.spacing === nothing
      @test exact.details.resolution === nothing
      @test bandwidth(exact) == bandwidth(select_bandwidth(R(), x))
    end
    # The one-stage SCV branch estimates orders 6 and 4 separately on the same grid.
    x = X2[:, 1:30]
    data = _mvdata(x, nothing)
    A, y, Σ = _pretransform(data, :sphere)
    bins = _MatrixBins(y, nothing, data.sumw, 17)
    R4, R6 = _multiindices(2, 4), _multiindices(2, 6)
    g6 = KernelSmoothing._samse(R6, KernelSmoothing._psinormal(Σ, _multiindices(2, 8)), data.neff)
    g4 = KernelSmoothing._samse(R4, KernelSmoothing._psinormal(Σ, R6), data.neff)
    ψ6 = _psihat(y, nothing, data.sumw, data.neff, g6, R6, bins)
    ψ4 = _psihat(y, nothing, data.sumw, data.neff, g4, R4, bins)
    T = KernelSmoothing._psitensor(ψ4, 2)
    Hpi = KernelSmoothing._minimizematrix(H -> KernelSmoothing._picriterion(H, T, (4π)^(-1) / data.neff),
                                        KernelSmoothing._normalscale(Σ, data.neff, false), false)[1]
    g = KernelSmoothing._scvpilot(KernelSmoothing._theta6(ψ6, 2), Hpi, data.neff)
    sel = select_bandwidth(SCVMatrix(stages = 1, binned = true, gridsize = 17), x)
    @test sel.details.pilot ≈ g rtol = 1e-13
    G = g^2 .* Matrix(1.0I, 2, 2)
    s0 = _gausspairs(y, nothing, data.sumw, [2G], [false], false, bins)[1][1]
    const0 = (2π)^(-1) / sqrt(det(2G)) / data.neff + 2s0 / data.sumw^2
    Hy = A * bandwidth(sel) * A'
    @test sel.details.criterion ≈ _scvcriterion(y, nothing, data.sumw, data.neff, Hy, G, const0, bins)[1] * abs(det(A)) rtol = 1e-12
    pi = select_bandwidth(PlugInMatrix(stages = 1, binned = true, gridsize = 17), x)
    pic = select_bandwidth(PlugInMatrix(stages = 1, compat = true, binned = true, gridsize = 17), x)
    @test pi.details.pilots[4] == g4
    @test bandwidth(pi) == bandwidth(pic)
    @test_throws ArgumentError PlugInMatrix(false, 2, :sphere, false, true, 2)
    @test_throws ArgumentError SCVMatrix(false, 2, :sphere, true, 2)
    @test_throws ArgumentError LSCVMatrix(false, true, 2)
    bins = _MatrixBins([0.0 2.0; 0.0 2.0], nothing, 2.0, 3)
    @test_logs (:warn, r"increase gridsize or use binned = false") _binningdetails(bins, 0.1 .* Matrix(1.0I, 2, 2))
    @test_logs (:warn, r"increase gridsize or use binned = false") _binningdetails(bins, 4.0 .* Matrix(1.0I, 2, 2), 0.5)
  end
end

end
