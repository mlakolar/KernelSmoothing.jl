module NNLSUpdateTest

using Test
using KernelSmoothing
using KernelSmoothing: _nnls, _NNLSQR, _nnls_qr_insert!, _nnls_qr_delete!, _nnls_qr_rebuild!
using LinearAlgebra
using Random

# The solver before QR updating, kept as an independent test oracle.
function _nnls_reference(A, b; maxiter = 3 * size(A, 2) + 50, tol = nothing)
  m, n = size(A)
  nb = LinearAlgebra.norm(b)
  nb > 0 || return zeros(n)
  s = [LinearAlgebra.norm(view(A, :, j)) for j in 1:n]
  live = findall(>(0), s)
  As = A[:, live] ./ s[live]'
  bs = b ./ nb
  k = length(live)
  x = zeros(k)
  passive = falses(k)
  rejected = falses(k)   # entering indices refused since x last changed
  tol = tol === nothing ? 10 * eps() * max(m, k) : tol
  w = As' * bs
  converged = false
  for _ in 1:maxiter
    j = 0
    best = tol
    for i in 1:k
      !passive[i] && !rejected[i] && w[i] > best && ((j, best) = (i, w[i]))
    end
    if j == 0
      converged = true
      break
    end
    passive[j] = true
    P = findall(passive)
    z = zeros(k)
    z[P] = As[:, P] \ bs
    if z[j] <= 0
      passive[j] = false
      rejected[j] = true
      continue
    end
    for _ in 1:k + 1
      if all(i -> z[i] > 0, P)
        x = z
        break
      end
      # the largest step toward z that keeps x ≥ 0; the indices that block it leave
      α, blocking = Inf, 0
      for i in P
        z[i] <= 0 || continue
        αi = x[i] / (x[i] - z[i])
        αi < α && ((α, blocking) = (αi, i))
      end
      x .+= α .* (z .- x)
      x[blocking] = 0.0
      for i in P
        x[i] <= 0 && (passive[i] = false; x[i] = 0.0)
      end
      P = findall(passive)
      z = zeros(k)
      isempty(P) || (z[P] = As[:, P] \ bs)
    end
    fill!(rejected, false)
    w = As' * (bs .- As * x)
  end
  converged || @warn "the nonnegative least-squares solver reached its iteration limit; the projection may be inexact"
  out = zeros(n)
  out[live] = x .* nb ./ s[live]
  return out
end

function check_qr(factor, A, b)
  p = length(factor.P)
  Q = factor.Q[:, 1:p]
  R = factor.R[1:p, 1:p]
  @test Q * R ≈ A[:, factor.P] atol = 2e-12
  @test Q' * Q ≈ Matrix{Float64}(I, p, p) atol = 2e-12
  @test factor.h[1:p] ≈ Q' * b atol = 2e-12
  @test istriu(R)
end

function check_optimality(A, b, x)
  s = [norm(view(A, :, j)) for j in axes(A, 2)]
  live = findall(>(0), s)
  As = A[:, live] ./ s[live]'
  bs = b ./ norm(b)
  xs = x[live] .* s[live] ./ norm(b)
  w = As' * (bs - As * xs)
  τ = 10 * eps() * max(size(A, 1), length(live))
  @test all(>=(0), x)
  @test all(i -> xs[i] > 0 || w[i] <= τ, eachindex(xs))
  @test all(i -> xs[i] == 0 || abs(w[i]) <= τ, eachindex(xs))
  @test maximum(abs, xs .* w; init = 0.0) <= τ
end

function check_reference(A, b; rtol = 1e-9, atol = 2e-12, optimality = true)
  x = @test_logs _nnls(A, b)
  reference = @test_logs _nnls_reference(A, b)
  @test A * x ≈ A * reference rtol = rtol atol = atol
  @test sum(abs2, A * x - b) ≈ sum(abs2, A * reference - b) rtol = 1e-10 atol = atol * max(1, sum(abs2, b))
  optimality && check_optimality(A, b, x)
  return x
end

@testset "passive QR insertions, deletions and rebuilds" begin
  rng = Xoshiro(912)
  A, b = randn(rng, 24, 10), randn(rng, 24)
  A ./= [norm(view(A, :, j)) for j in axes(A, 2)]'
  factor = _NNLSQR(size(A)...)
  check_qr(factor, A, b)
  for j in (4, 1, 10, 2, 9, 6, 3, 8, 7, 5)
    @test _nnls_qr_insert!(factor, A, b, j)
    check_qr(factor, A, b)
  end
  @test factor.updates == 10 && factor.rebuilds == 0
  # The first, a middle and the last column exercise every part of the bulge chase.
  for position in (1, 4, 8)
    _nnls_qr_delete!(factor, A, b, position)
    check_qr(factor, A, b)
  end
  @test factor.updates == 13
  P = copy(factor.P)
  _nnls_qr_rebuild!(factor, A, b)
  @test factor.P == P
  @test factor.updates == 13 && factor.rebuilds == 1 && factor.since_rebuild == 0
  check_qr(factor, A, b)
  for position in length(factor.P):-1:1
    _nnls_qr_delete!(factor, A, b, position)
    check_qr(factor, A, b)
  end
  @test isempty(factor.P)
  # A refusal, including the coefficient guard, is a transaction with no changes.
  A = [1.0 1.0 -1.0 -1.0 0.0; 0.0 0.0 0.0 1e-12 0.0; 0.0 0.0 0.0 0.0 0.0]
  b = [1.0, 2.0, 3.0]
  factor = _NNLSQR(size(A)...)
  @test !_nnls_qr_insert!(factor, A, -b, 1; positive = true)
  @test isempty(factor.P) && factor.updates == 0
  @test _nnls_qr_insert!(factor, A, b, 1)
  before = deepcopy(factor)
  for j in 2:5
    @test !_nnls_qr_insert!(factor, A, b, j)
    @test factor.P == before.P && factor.Q == before.Q && factor.R == before.R && factor.h == before.h
    @test (factor.updates, factor.rebuilds, factor.since_rebuild) ==
          (before.updates, before.rebuilds, before.since_rebuild)
  end
  # The small diagonal triggers the accuracy rebuild above the rank cutoff.
  A = [1.0 1.0; 0.0 5e-9; 0.0 0.0]
  factor = _NNLSQR(size(A)...)
  @test _nnls_qr_insert!(factor, A, b, 1)
  @test _nnls_qr_insert!(factor, A, b, 2)
  @test factor.updates == 2 && factor.rebuilds == 1 && factor.since_rebuild == 0
  check_qr(factor, A, b)
end

@testset "passive QR storage growth" begin
  rng = Xoshiro(915)
  A = randn(rng, 64, 45)
  A = hcat(A, A[:, 1])
  b = randn(rng, 64)
  factor = _NNLSQR(size(A)...)
  @test size(factor.Q) == (64, 16)
  @test size(factor.R) == (16, 16) && length(factor.h) == 16
  @test factor.max_capacity == 46
  for j in 1:45
    @test _nnls_qr_insert!(factor, A, b, j)
    capacity = j <= 16 ? 16 : j <= 32 ? 32 : 46
    @test size(factor.Q, 2) == capacity
    @test size(factor.R) == (capacity, capacity) && length(factor.h) == capacity
    @test factor.updates == j && factor.rebuilds == (j >= 16)
    check_qr(factor, A, b)
    if j in (16, 32)
      before = deepcopy(factor)
      Q, R, h = factor.Q, factor.R, factor.h
      @test !_nnls_qr_insert!(factor, A, b, 46)
      @test factor.Q === Q && factor.R === R && factor.h === h
      @test factor.P == before.P && factor.Q == before.Q && factor.R == before.R && factor.h == before.h
      @test (factor.updates, factor.rebuilds, factor.since_rebuild) ==
            (before.updates, before.rebuilds, before.since_rebuild)
    end
  end
  for position in (1, 22, 43)
    _nnls_qr_delete!(factor, A, b, position)
    check_qr(factor, A, b)
  end
  _nnls_qr_rebuild!(factor, A, b)
  check_qr(factor, A, b)
  @test size(factor.Q, 2) == 46
  @test size(_NNLSQR(7, 100).Q) == (7, 7)
  @test size(_NNLSQR(100, 0).Q) == (100, 0)
end

@testset "NNLS allocation with few passive columns" begin
  m, n = 400, 399
  A = Matrix{Float64}(I, m, n)
  b = zeros(m)
  b[1:8] .= 1:8
  x = _nnls(A, b)   # warm the exact call measured below, outside @allocated
  @test count(>(0), x) == 8
  check_optimality(A, b, x)
  bytes = @allocated _nnls(A, b)
  # A[:, live] and the normalized As each allocate a dense m × n matrix. Above
  # those current work arrays, allow one third of a matrix for vectors, headers
  # and the small QR buffers. The former full Q and R alone cost nearly two
  # matrices, so they exceed this bound even before the other solver allocations.
  matrix_bytes = sizeof(Float64) * m * n
  workspace_bytes = 2 * matrix_bytes
  @test bytes - workspace_bytes < matrix_bytes ÷ 3
end

@testset "amortized rebuild counts" begin
  A = Matrix{Float64}(I, 48, 32)
  b = ones(48)
  factor = _NNLSQR(size(A)...)
  for j in 1:32
    @test _nnls_qr_insert!(factor, A, b, j)
    @test factor.since_rebuild < max(length(factor.P), 16)
  end
  @test factor.updates == 32 && factor.rebuilds == 1 && factor.since_rebuild == 16
  # Hold p above 16 while enough insertions and deletions accumulate for a rebuild.
  for _ in 1:8
    _nnls_qr_delete!(factor, A, b, 32)
    @test _nnls_qr_insert!(factor, A, b, 32)
    check_qr(factor, A, b)
  end
  @test factor.updates == 48 && factor.rebuilds == 2 && factor.since_rebuild == 1
  @test factor.rebuilds <= factor.updates / 16 + 1
  # Below 16 passive columns the interval stays at 16 updates.
  A = Matrix{Float64}(I, 8, 8)
  b = ones(8)
  factor = _NNLSQR(size(A)...)
  for i in 1:8
    @test _nnls_qr_insert!(factor, A, b, 1)
    _nnls_qr_delete!(factor, A, b, 1)
    @test factor.rebuilds == (2 * i) ÷ 16
    check_qr(factor, A, b)
  end
  stats = Dict{Symbol, Int}()
  x = @test_logs _nnls(Matrix{Float64}(I, 40, 40), ones(40); stats)
  @test x ≈ ones(40)
  @test stats[:updates] == 40 && stats[:rebuilds] == 1 && stats[:fallbacks] == 0
  @test stats[:rebuilds] <= stats[:updates] / 16 + 1
end

@testset "fallback rejection and stopping" begin
  A = [1.0 0.0; 0.0 1.0]
  b = [1.0, -1.0]
  stats = Dict{Symbol, Int}()
  x = @test_logs _nnls(A, b; tol = -1.0, stats)
  @test x ≈ [1.0, 0.0] atol = 1e-14
  @test x ≈ (@test_logs _nnls_reference(A, b; tol = -1.0)) atol = 1e-14
  @test stats[:fallbacks] == 1 && stats[:updates] == 1
  # The same candidate is refused once by QR and once by the fallback solve.
  @test stats[:rejections] == 2 && stats[:reconsiderations] == 1
  @test stats[:deletions] == 0
  # Stopping at the guard leaves x as it was before the final iteration.
  @test (@test_logs (:warn, r"iteration limit") _nnls(A, b; tol = -1.0, maxiter = 4)) == x
end

@testset "backtracking in fallback" begin
  δ = 2.0^-32
  A = [1.0 0.0 0.0 100.0; 0.0 1.0 1.0 1.0; 0.0 1.0 -1.0 0.0; 0.0 0.0 0.0 δ]
  b = [1000.0, 1.0, 0.0, 2δ]
  stats = Dict{Symbol, Int}()
  # The first three columns enter by QR. The fourth fails the rank cutoff;
  # in fallback its positive coefficient makes both middle coefficients negative.
  x = @test_logs _nnls(A, b; tol = -1.0, stats)
  reference = @test_logs _nnls_reference(A, b; tol = -1.0)
  @test x ≈ [900.0, 0.0, 0.0, 1.0] atol = 1e-12
  @test A * x ≈ A * reference rtol = 1e-12 atol = 1e-12
  @test stats[:fallbacks] == 1 && stats[:updates] == 3
  @test stats[:deletions] == 2 && stats[:max_blockers] == 2
end

@testset "deterministic NNLS updates" begin
  # Two symmetric passive coefficients hit zero in the same backtracking step.
  A = [1.0 0.0 0.0 100.0; 0.0 1.0 1.0 1.0; 0.0 1.0 -1.0 0.0; 0.0 0.0 0.0 1.0]
  b = [1000.0, 1.0, 0.0, 2.0]
  stats = Dict{Symbol, Int}()
  x = @test_logs _nnls(A, b; stats)
  @test x ≈ [850.0, 0.0, 0.0, 1.5] atol = 2e-11
  @test stats[:deletions] == 2 && stats[:max_blockers] == 2
  @test stats[:updates] == 6 && stats[:fallbacks] == 0
  check_reference(A, b)
  # A nearly opposite column is rejected, another column enters, and the first
  # is reconsidered. The final dual check must then invoke the fallback.
  δ = 2.0^-33
  A = [1.0 -1.0 0.0; 0.0 δ 0.0; 0.0 0.0 1.0]
  b = [1.0, 2.0^-8, δ * 2.0^-9]
  x = @test_logs _nnls(A, b; stats)
  @test stats[:rejections] >= 2 && stats[:reconsiderations] >= 1 && stats[:fallbacks] == 1
  @test A * x ≈ b atol = 1e-8
  # Cancellation in nearly opposite columns limits the accuracy of either solver.
  check_reference(A, b; atol = 1e-8, optimality = false)
  for δ in (5e-9, 1e-12)
    A = [1.0 1.0 -1.0; 0.0 δ δ; 0.0 0.0 0.0; 0.0 0.0 0.0]
    b = [1.0, 0.01, 0.0, 0.0]
    check_reference(A, b; atol = 1e-6, optimality = false)
  end
  A = [1.0 1.0 -1.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 0.0 0.0]
  for b in ([1.0, 2.0, -1.0], [-1.0, 2.0, 0.0])
    check_reference(A, b)
  end
  @test (@test_logs _nnls(A, zeros(3); stats)) == zeros(5)
  @test all(iszero, values(stats))
  @test (@test_logs _nnls(zeros(5, 3), ones(5); stats)) == zeros(3)
  @test all(iszero, values(stats))
  @test (@test_logs _nnls(zeros(5, 0), ones(5))) == Float64[]
  rng = Xoshiro(911)
  A = randn(rng, 20, 8)
  coefficients = 0.5 .+ rand(rng, 8)
  b = A * coefficients
  @test check_reference(A, b) ≈ coefficients atol = 2e-12
  for _ in 1:3
    A, b = randn(rng, 10, 6), randn(rng, 10)
    @test (@test_logs _nnls(A, b; tol = -1.0)) ≈ _nnls_reference(A, b; tol = -1.0) atol = 2e-12
  end
  @test_logs (:warn, r"iteration limit") _nnls(A, b; maxiter = 0)
  @test_logs (:warn, r"iteration limit") _nnls(-ones(3, 1), ones(3); maxiter = 0, tol = -1.0)
end

@testset "NNLS optimality and reference fits" begin
  rng = Xoshiro(913)
  for (m, n) in ((12, 5), (30, 18), (60, 55), (35, 35)), trial in 1:5
    A, b = randn(rng, m, n), randn(rng, m)
    trial == 5 && (A .*= exp10.(range(-5, 5; length = n))')
    check_reference(A, b)
  end
  for m in (12, 30, 60), _ in 1:3
    B = randn(rng, m, 8)
    A = hcat(B, B[:, 1], -B[:, 2], B[:, 3] + B[:, 4], zeros(m))
    check_reference(A, randn(rng, m))
  end
end

@testset "passive residual and inactive dual against dense reference" begin
  rng = Xoshiro(916)
  for (m, n) in ((12, 6), (32, 20), (40, 36)), _ in 1:3
    A, b = randn(rng, m, n), randn(rng, m)
    check_reference(A, b; rtol = 1e-12, atol = 0.0)
  end
  for _ in 1:3
    B = randn(rng, 24, 8)
    b = randn(rng, 24)
    # A linearly dependent set and exact duplicate columns need only agree in fit.
    for A in (hcat(B, B[:, 1] + B[:, 2], -B[:, 3], zeros(24)),
              hcat(B, B[:, 1], B[:, 1], B[:, 2]))
      check_reference(A, b; rtol = 1e-12, atol = 0.0)
    end
  end
end

@testset "reference fits for every shape cone" begin
  rng = Xoshiro(914)
  for m in (12, 40, 60), shape in KernelSmoothing._SHAPES, weighted in (false, true)
    t = cumsum(0.5 .+ rand(rng, m))
    t = (t .- t[1]) ./ (t[end] - t[1])
    F, E = KernelSmoothing._conegenerators(shape, t)
    v = weighted ? 0.5 .+ rand(rng, m) : ones(m)
    sw = sqrt.(v)
    Fw, Ew = sw .* F, sw .* E
    Q = Matrix(qr(Fw).Q)[:, 1:size(F, 2)]
    Er = Ew - Q * (Q' * Ew)
    for y in (randn(rng, m), F * randn(rng, size(F, 2)) + E * (0.5 .+ rand(rng, size(E, 2))))
      ys, scale = KernelSmoothing._scaled_values(y)
      yw = sw .* ys
      yr = yw - Q * (Q' * yw)
      c = check_reference(Er, yr)
      reference = _nnls_reference(Er, yr)
      a = Fw \ (yw - Ew * reference)
      θ = scale .* (F * a + E * reference)
      @test (@test_logs KernelSmoothing._coneproject(y, F, E, v)) ≈ θ rtol = 1e-9 atol = 2e-11
      @test (@test_logs shape_projection(y, t; shape, weights = v)) ≈ θ rtol = 1e-9 atol = 2e-11
      @test norm(Er * c - yr) ≈ norm(Er * reference - yr) rtol = 1e-10 atol = 2e-12
    end
  end
end

end
