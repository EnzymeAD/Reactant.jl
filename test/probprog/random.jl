using Reactant, Test
using Reactant: TracedRArray, TracedRNumber, MLIR, TracedUtils, ConcreteRArray
using Reactant.MLIR: IR
using Reactant.MLIR.Dialects: enzyme

# `enzyme.randomSplit` op is not intended to be emitted directly in Reactant-land.
# It is solely an intermediate representation within the `enzyme.mcmc` op lowering.
function random_split(rng_state::TracedRArray{UInt64,1}, ::Val{N}) where {N}
    rng_mlir = TracedUtils.get_mlir_data(rng_state)
    rng_state_type = IR.TensorType([2], IR.Type(UInt64))
    output_types = [rng_state_type for _ in 1:N]
    op = enzyme.randomSplit(rng_mlir; output_rng_states=output_types)
    return ntuple(i -> TracedRArray{UInt64,1}((), IR.result(op, i), (2,)), Val(N))
end

@testset "enzyme.randomSplit op" begin
    @testset "N=2, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2 = @jit optimize = :probprog random_split(seed, Val(2))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
    end

    @testset "N=2, Seed [42, 0]" begin
        seed = ConcreteRArray(UInt64[42, 0])
        k1, k2 = @jit optimize = :probprog random_split(seed, Val(2))

        @test Array(k1) == [0x4f6cc618de79f4b9, 0x99ba4efe6b200159]
        @test Array(k2) == [0xf67a601be6bdada3, 0xcddb151d375f238f]
    end

    @testset "N=3, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2, k3 = @jit optimize = :probprog random_split(seed, Val(3))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
        @test Array(k3) == [0xa20e4081f71f4ea9, 0x2f36b83d4e83f1ba]
    end

    @testset "N=4, Seed [0, 42]" begin
        seed = ConcreteRArray(UInt64[0, 42])
        k1, k2, k3, k4 = @jit optimize = :probprog random_split(seed, Val(4))

        @test Array(k1) == [0x99ba4efe6b200159, 0x4f6cc618de79f4b9]
        @test Array(k2) == [0xcddb151d375f238f, 0xf67a601be6bdada3]
        @test Array(k3) == [0xa20e4081f71f4ea9, 0x2f36b83d4e83f1ba]
        @test Array(k4) == [0xe4e8dfbe9312778b, 0x982ff5502e6ccb51]
    end
end
