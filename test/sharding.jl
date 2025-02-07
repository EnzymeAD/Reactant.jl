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
    else
        @warn "Not enough addressable devices to run sharding tests"
    end
end
