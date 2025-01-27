using Reactant
using Reactant: Ops

using Test

# Jax on Github CI dislikes X86 macos
@static if !Sys.isapple() || Sys.ARCH != :x86_64
    using PythonCall

    @testset "PythonCall" begin
        jax = pyimport("jax")

        result = @jit jax.numpy.sum(Reactant.to_rarray(Float32[1, 2, 3]))
        @test typeof(result) == ConcreteRNumber{Float32}
        @test result ≈ 6
    end
end
