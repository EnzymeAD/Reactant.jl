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

        @test cdata_sharded.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
        @test cdata_sharded2.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
        @test cdata_sharded3.sharding isa Sharding.ShardInfo{<:Sharding.NamedSharding}
        @test cdata.sharding isa Sharding.NoShardInfo

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

predict(samples, w1, w2) = sin.(w2 * (w1 * tanh.(samples)))

fn_test2(x) = x .+ x'

fn_test3(x) = sum(x; dims=1)

@testset "Sharding Across 8 Devices" begin
    if length(addressable_devices) ≥ 8
        mesh = Sharding.Mesh(reshape(collect(Int64, 0:7), (4, 2)), ("data", "model"))

        x = reshape(collect(Float32, 1:16), 4, 4)
        x_ra = Reactant.to_rarray(
            x; sharding=Sharding.NamedSharding(mesh, ("data", "model"))
        )

        @test Array(@jit fn_test2(x_ra)) ≈ fn_test2(x)

        @test Array(@jit fn_test3(x_ra)) ≈ fn_test3(x)

        samples = reshape(collect(Float32, 1:48), 4, 12)
        w1 = reshape(collect(Float32, 1:16), 4, 4)
        w2 = reshape(collect(Float32, 1:32), 8, 4)

        for (samples_sharding, w1_sharding, w2_sharding) in zip(
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NamedSharding(mesh, ("model", nothing)),
                Sharding.NamedSharding(mesh, (nothing, "data")),
            ),
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NamedSharding(mesh, (nothing, "data")),
                Sharding.NoSharding(),
            ),
            (
                Sharding.NamedSharding(mesh, ("model", "data")),
                Sharding.NoSharding(),
                Sharding.NoSharding(),
            ),
        )
            samples_ra = Reactant.to_rarray(samples; sharding=samples_sharding)
            w1_ra = Reactant.to_rarray(w1; sharding=w1_sharding)
            w2_ra = Reactant.to_rarray(w2; sharding=w2_sharding)

            @test Array(@jit(predict(samples_ra, w1_ra, w2_ra))) ≈ predict(samples, w1, w2)
        end
    end
end

# Tests from the examples in
# https://github.com/openxla/xla/blob/96d6678053d867099a42be9001c49b2ed7111afd/xla/hlo/ir/tile_assignment.h#L53-L68
@testset "Device List from Iota Tile" begin
    @test Reactant.XLA.generate_device_list(
        Reactant.XLA.OpSharding(
            Reactant.XLA.OpShardingType.Other,
            Int64[],
            Int64[],
            true,
            Reactant.XLA.OpShardingType.T[],
            [4, 4, 1],
            Int64[],
            [4, 2, 2],
            Int32[1, 2, 3],
            false,
            0,
            Reactant.XLA.ShardGroupType.As,
        ),
    ) == collect(0:15)

    @test Reactant.XLA.generate_device_list(
        Reactant.XLA.OpSharding(
            Reactant.XLA.OpShardingType.Other,
            Int64[],
            Int64[],
            true,
            Reactant.XLA.OpShardingType.T[],
            [4, 4, 1],
            Int64[],
            [4, 2, 2],
            Int32[2, 1, 3],
            false,
            0,
            Reactant.XLA.ShardGroupType.As,
        ),
    ) == [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15]

    @test Reactant.XLA.generate_device_list(
        Reactant.XLA.OpSharding(
            Reactant.XLA.OpShardingType.Other,
            Int64[],
            Int64[],
            true,
            Reactant.XLA.OpShardingType.T[],
            [2, 4],
            Int64[],
            [4, 2],
            Int32[2, 1],
            false,
            0,
            Reactant.XLA.ShardGroupType.As,
        ),
    ) == [0, 2, 4, 6, 1, 3, 5, 7]
end
