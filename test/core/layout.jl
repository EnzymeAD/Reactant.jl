using Reactant
using Test

@testset "Layout" begin
    client = Reactant.XLA.default_backend()
    device = Reactant.XLA.default_device()
    sharding = Sharding.NoSharding()
    idx = 0

    @test ConcreteRArray{Float32}(undef, (100, 10)) isa ConcreteRArray{Float32,2}

    @test ConcreteRArray{Float32}(
        undef, (100, 10); client=client, idx=idx, device=device
    ) isa ConcreteRArray{Float32,2}

    @test ConcreteRArray{Float32}(
        undef, Int32(100), Int16(10); client=client, idx=idx, device=device
    ) isa ConcreteRArray{Float32,2}

    @test ConcreteRNumber(Float32(4.2)) isa ConcreteRNumber{Float32}

    @test ConcreteRNumber(Float16(4.2); client=client, idx=idx, device=device) isa
        ConcreteRNumber{Float16}

    @test ConcreteRNumber{Float32}(Float32(4.2); client=client, idx=idx, device=device) isa
        ConcreteRNumber{Float32}

    @test ConcreteRNumber{Float16}(Float32(4.2)) isa ConcreteRNumber{Float16}

    @test ConcreteRNumber{Float32}(Float16(4.2); client=client, idx=idx, device=device) isa
        ConcreteRNumber{Float32}

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
