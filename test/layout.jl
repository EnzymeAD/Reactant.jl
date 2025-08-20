using Reactant
using Test

@testset "Layout" begin
    client = Reactant.XLA.default_backend()
    device = Reactant.XLA.default_device()
    sharding = Sharding.NoSharding()
    idx = 0

    @test ConcreteRArray{Float32}(undef, (100, 10)) isa ConcreteRArray{Float32,2}

    @test ConcreteRArray{Float32}(
        undef, (100, 10); idx=idx, client=client, device=device
    ) isa ConcreteRArray{Float32,2}

    @test ConcreteRArray{Float32}(
        undef, Int32(100), Int16(10); idx=idx, client=client, device=device
    ) isa ConcreteRArray{Float32,2}

    @test_deprecated ConcreteRArray(
        undef, Float32, (100, 10); idx=idx, client=client, device=device
    ) isa ConcreteRArray{Float32,2}

    x = reshape([1.0, 2.0, 3.0, 4.0], (2, 2))

    y = Reactant.to_rarray(x)

    y2 = convert(Array{Float64,2}, y)

    @test x == y2

    @allowscalar begin
        @test y[1, 1] == x[1, 1]
        @test y[1, 2] == x[1, 2]
        @test y[2, 1] == x[2, 1]
        @test y[2, 2] == x[2, 2]
    end
end
