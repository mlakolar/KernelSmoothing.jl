module MultivariateKDETiesTest

using Test
using KernelSmoothing
using LinearAlgebra
using Logging: current_logger
using Random

include(joinpath(@__DIR__, "data", "ks_golden.jl"))

# Counts on the original sample, with both signs of zero identified.
function tiecounts(x)
  z = x .+ 0.0
  n = size(z, 2)
  return n - size(unique(z; dims = 2), 2), [1 - length(unique(row)) / n for row in eachrow(z)]
end

@testset "LSCV relative eigenvalues" begin
  H0 = Matrix(2.0I, 2, 2)
  Hy = [2.0 0.0; 0.0 8.0]
  @test all(isnan, KernelSmoothing._lscveigenratio([1.0 0.0; 0.0 -1.0], Hy))
  @test all(isnan, KernelSmoothing._lscveigenratio(H0, [Inf 0.0; 0.0 8.0]))
  ρmin, ρmax = KernelSmoothing._lscveigenratio(H0, Hy)
  @test ρmin ≈ 1.0
  @test ρmax ≈ 4.0
end

@testset "LSCV duplicate runaway" begin
  for duplicates in (50, 100)
    @testset "$duplicates duplicate pairs" begin
      x = randn(Xoshiro(7), 2, 200)
      x[:, 201-duplicates:200] .= x[:, 1:duplicates]
      sel = @test_logs (:warn, r"tied or duplicated") (:warn, r"degenerate bandwidth matrix") match_mode = :any begin
        result = select_bandwidth(LSCVMatrix(binned = false), x)
        messages = [log.message for log in current_logger().logs]
        for phrase in (r"tied or duplicated", r"degenerate bandwidth matrix")
          @test count(message -> occursin(phrase, message), messages) == 1
          @test all(message -> startswith(message, "LSCVMatrix: "), filter(message -> occursin(phrase, message), messages))
        end
        result
      end
      @test sel isa BandwidthSelection
      @test size(bandwidth(sel)) == (2, 2)
      @test sel.details.duplicates === duplicates
      @test sel.details.ties == last(tiecounts(x))
      @test sel.details.collapsed
      @test !sel.details.converged
      ρmin, ρmax = sel.details.eigenratio
      @test !(ρmin > 1e-8) || !(ρmax < 1e8)
    end
  end
end

@testset "LSCV binned collapse" begin
  for d in (2, 3), diagonal in (false, true)
    @testset "d = $d, diagonal = $diagonal" begin
      x = randn(Xoshiro(102), d, 60)
      sel = @test_logs (:warn, r"degenerate bandwidth matrix") match_mode = :any begin
        result = select_bandwidth(LSCVMatrix(; diagonal, binned = true, gridsize = 3), x)
        @test !any(log -> occursin(r"tied or duplicated", log.message), current_logger().logs)
        result
      end
      @test sel.details.binned
      @test sel.details.duplicates === 0
      @test all(iszero, sel.details.ties)
      @test sel.details.collapsed
      @test !sel.details.converged
    end
  end
end

@testset "LSCV rounded observations" begin
  for d in (2, 3), digits in (1, 2)
    @testset "d = $d, digits = $digits" begin
      x = round.(randn(Xoshiro(102), d, 100); digits)
      duplicates, ties = tiecounts(x)
      @test maximum(ties) > 0.1
      sel = @test_logs (:warn, r"tied or duplicated") match_mode = :any select_bandwidth(LSCVMatrix(binned = false), x)
      @test sel isa BandwidthSelection
      @test size(bandwidth(sel)) == (d, d)
      @test sel.details.duplicates === duplicates
      @test sel.details.ties isa Vector{Float64}
      @test sel.details.ties == ties
    end
  end
  for d in (2, 3)
    x = round.(randn(Xoshiro(102), d, 100); digits = 3)
    duplicates, ties = tiecounts(x)
    @test duplicates == 0
    @test 0 < maximum(ties) < 0.1
    sel = @test_logs select_bandwidth(LSCVMatrix(binned = false), x)
    @test sel.details.duplicates === duplicates
    @test sel.details.ties == ties
  end
end

@testset "LSCV tie data contract" begin
  x = randn(Xoshiro(102), 2, 100)
  x[:, 91:100] .= x[:, 1:10]
  duplicates, ties = tiecounts(x)
  @test duplicates == 10
  @test maximum(ties) <= 0.1
  for diagonal in (false, true), weights in (nothing, 1.0 .+ mod.(1:100, 3))
    sel = @test_logs (:warn, r"tied or duplicated") match_mode = :any select_bandwidth(LSCVMatrix(; diagonal, binned = false), x; weights)
    @test sel isa BandwidthSelection
    @test sel.details.duplicates === duplicates
    @test sel.details.ties == ties
  end
  # A zero-weight copy is absent from both the counts and their denominator.
  kept = randn(Xoshiro(102), 2, 60)
  xzero = hcat(kept, kept[:, 1])
  sel = @test_logs select_bandwidth(LSCVMatrix(binned = false), xzero; weights = [ones(60); 0.0])
  @test sel.details.duplicates === first(tiecounts(kept)) === 0
  @test sel.details.ties == last(tiecounts(kept))
  @test all(iszero, sel.details.ties)
  # The first two columns differ only in the sign of their zero coordinate.
  signed = randn(Xoshiro(102), 2, 60)
  signed[:, 1] .= [0.0, 1.0]
  signed[:, 2] .= [-0.0, 1.0]
  signed[1, 3] = -0.0
  original = copy(signed)
  duplicates, ties = tiecounts(signed)
  @test duplicates == 1
  @test ties == [1 - 58 / 60, 1 - 59 / 60]
  for diagonal in (false, true)
    sel = @test_logs (:warn, r"tied or duplicated") match_mode = :any select_bandwidth(LSCVMatrix(; diagonal, binned = false), signed)
    @test sel.details.duplicates === duplicates
    @test sel.details.ties == ties
    @test isequal(signed, original)
  end
  for rule in (PlugInMatrix(binned = false), SCVMatrix(binned = false))
    sel = @test_logs select_bandwidth(rule, x)
    @test !hasproperty(sel.details, :duplicates)
  end
end

@testset "LSCV continuous golden samples" begin
  for x in (X2, X3), diagonal in (false, true)
    sel = @test_logs select_bandwidth(LSCVMatrix(; diagonal, binned = false), x)
    @test sel.details.converged
    @test !sel.details.collapsed
    @test sel.details.duplicates === 0
    @test all(iszero, sel.details.ties)
    @test sel.details.eigenratio isa Tuple{Float64,Float64}
    @test all(isfinite, sel.details.eigenratio)
    @test all(ρ -> 1e-8 < ρ < 1e8, sel.details.eigenratio)
    @test all(isfinite, bandwidth(sel)) && isposdef(bandwidth(sel))
  end
  sel = @test_logs (:warn, r"coarse") select_bandwidth(LSCVMatrix(binned = true), X3)
  @test sel.details.binned
  @test sel.details.converged
  @test !sel.details.collapsed
  @test sel.details.duplicates === 0
  @test all(iszero, sel.details.ties)
  @test all(isfinite, sel.details.eigenratio)
  @test all(ρ -> 1e-8 < ρ < 1e8, sel.details.eigenratio)
end

end
