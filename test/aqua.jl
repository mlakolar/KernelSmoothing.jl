module AquaTest

using Test
using KernelSmoothing
using Aqua

@testset "Aqua" begin
  Aqua.test_all(KernelSmoothing)
end

end
