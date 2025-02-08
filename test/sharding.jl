# Currently an extremely simple test
using Reactant, Test

const addressable_devices = Reactant.addressable_devices()

function fn_test1(x)
    y = x .+ x
    x .+= 1
    z = x .* y
    return y, x, z
end

@testset "Sharding Across 2 Devices" begin
    if length(addressable_devices) ≥ 2
        mesh = Sharding.Mesh([0 1;], ("x", "y"))
    else
        @warn "Not enough addressable devices to run sharding tests; we are running a \
               pretend test for testing purposes"
        mesh = Sharding.Mesh(reshape([0], 1, 1), ("x", "y"))
    end

    data_sharding = Sharding.NamedSharding(mesh, ("y", nothing, "x"))
    data_sharding2 = Sharding.NamedSharding(mesh, (nothing, "x", nothing))
    data_sharding3 = Sharding.NamedSharding(mesh, (nothing, nothing, nothing)) # fully replicated data

    data = reshape(collect(1:(16 * 4 * 12)) ./ (16 * 4 * 12), 16, 4, 12)

    cdata = Reactant.to_rarray(data)
    cdata_sharded = Reactant.to_rarray(data; sharding=data_sharding)
    cdata_sharded2 = Reactant.to_rarray(data; sharding=data_sharding2)
    cdata_sharded3 = Reactant.to_rarray(data; sharding=data_sharding3)

    @test data ≈
        Array(cdata) ≈
        Array(cdata_sharded) ≈
        Array(cdata_sharded2) ≈
        Array(cdata_sharded3)

    @test cdata_sharded.sharding isa Sharding.FinalizedNamedSharding
    @test cdata_sharded2.sharding isa Sharding.FinalizedNamedSharding
    @test cdata_sharded3.sharding isa Sharding.FinalizedNamedSharding
    @test cdata.sharding isa Sharding.FinalizedNoSharding

    true_res_y, true_res_x, true_res_z = fn_test1(data)

    for cd in (cdata, cdata_sharded, cdata_sharded2, cdata_sharded3)
        local res_y, res_x, res_z = @jit fn_test1(cd)
        @test Array(cd) ≈ Array(res_x)
        @test Array(res_y) ≈ true_res_y
        @test Array(res_z) ≈ true_res_z
        @test Array(res_x) ≈ true_res_x
    end
end

predict(samples, w1, w2) = sin.(w2 * (w1 * tanh.(samples)))

@testset "Sharding Across 8 Devices" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), (4, 2)), ("data", "model"))
    else
        @warn "Not enough addressable devices to run sharding tests; we are running a \
               pretend test for testing purposes"
        mesh = Sharding.Mesh(reshape([0], 1, 1), ("data", "model"))
    end

    samples_sharding = Sharding.NamedSharding(mesh, (nothing, "data"))
    w1_sharding = Sharding.NamedSharding(mesh, ("model", nothing))

    samples = ConcreteRArray(rand(Float32, 3, 12); sharding=samples_sharding)
    w1 = ConcreteRArray(rand(Float32, 4, 3); sharding=w1_sharding)
    w2 = ConcreteRArray(rand(Float32, 2, 4))

    @test Array(@jit(predict(samples, w1, w2))) ≈
        predict(Array(samples), Array(w1), Array(w2))
end
